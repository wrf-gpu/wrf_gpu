#!/usr/bin/env python3
"""Pointwise cross-case mean and sample SD from complete hourly band receipts."""
import argparse
import html
import json
import os
import math
from pathlib import Path

import numpy as np

from hourly_deviation_bands import FIELDS, quiet, validate_metrics

def aggregate(cases, domain, field, metric='rmse'):
    entries = []
    names = []
    for label, js in cases.items():
        if domain not in js['domains']:
            continue
        rows = [r for r in js['domains'][domain]['fields'][field]['by_lead'] if metric in r]
        expected = list(range(1 if field == 'RAINNC_hourly' else 0, 73))
        if [r['lead_h'] for r in rows] != expected:
            raise ValueError(f'incomplete hours {label}/{domain}/{field}')
        vals = np.array([r[metric] for r in rows])
        if not np.isfinite(vals).all():
            raise ValueError(f'nonfinite entire case metric {label}/{domain}/{field}')
        names.append(label)
        entries.append(vals)
    if not entries:
        return None
    matrix = np.stack(entries)
    return {'leads': expected, 'cases': names, 'per_case': matrix.tolist(),
            'n_cases': len(names), 'mean': matrix.mean(axis=0).tolist(),
            'sample_sd': matrix.std(axis=0, ddof=1).tolist() if len(names) > 1 else None,
            'definition': 'equal case weight at each forecast hour; one sample SD(ddof=1), not a confidence interval'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', required=True, help='LABEL=metrics.json')
    p.add_argument('--expected-cases', nargs='+', required=True)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--title', default='v0.3.3 hourly deviation summary')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('CPU-only')
    cases = {}
    for arg in a.case:
        label, path = arg.split('=', 1)
        if label in cases:
            raise ValueError(f'duplicate case {label}')
        cases[label] = validate_metrics(json.loads(Path(path).read_text()))
    unknown = set(cases) - set(a.expected_cases)
    missing = sorted(set(a.expected_cases) - set(cases))
    if unknown or (missing and not a.allow_partial):
        raise ValueError(f'case set mismatch unknown={unknown} missing={missing}')
    domains = sorted(set().union(*(set(js['domains']) for js in cases.values())))
    quiet()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.out.mkdir(parents=True, exist_ok=True)
    result = {'case_list': list(cases), 'expected_cases': a.expected_cases,
              'missing_cases': missing, 'complete_case_set': not missing,
              'scope': a.title, 'fields': {}}
    images = []
    domain_images = {dom: [] for dom in domains}
    def finish_axis(ax, stats, limit, informational):
        upper = np.asarray(stats['mean'])
        if stats['sample_sd'] is not None:
            upper = upper + np.asarray(stats['sample_sd'])
        ymax = max(float(np.max(stats['per_case'])), float(upper.max())) * 1.15
        ymax = ymax if ymax > 0 else limit*.05
        ax.set_ylim(0,ymax)
        label='accumulated D6 reference only' if informational else 'frozen D6 RMSE limit'
        if limit<=ymax:
            ax.axhline(limit,color='#b63231',ls='--',lw=1,label=label)
        else:
            ax.annotate(f'D6 {limit:g} above'+(' (reference)' if informational else ''),
                        xy=(.97,1),xycoords='axes fraction',xytext=(.97,1.035),textcoords='axes fraction',
                        ha='right',va='bottom',fontsize=8,color='#b63231',
                        arrowprops={'arrowstyle':'->','color':'#b63231'})
    for name, unit, _ in FIELDS:
        quiet()
        fig, axes = plt.subplots(1, len(domains), figsize=(5.3 * len(domains), 5.6), squeeze=False)
        result['fields'][name] = {}
        for i, dom in enumerate(domains):
            stats = aggregate(cases, dom, name)
            if stats is None:
                continue
            result['fields'][name][dom] = stats
            ax = axes[0, i]
            x = np.array(stats['leads']);mean = np.array(stats['mean'])
            for label, values in zip(stats['cases'], stats['per_case']):
                ax.plot(x, values, lw=.8, alpha=.65, label=label)
            if stats['sample_sd'] is not None:
                sd = np.array(stats['sample_sd'])
                ax.fill_between(x, np.maximum(0,mean-sd), mean+sd, color='#4383b9', alpha=.22, label='cross-case mean ± 1 sample SD (lower clipped at 0)')
            ax.plot(x, mean, color='#083e74', lw=2.2, label='equal-case mean')
            limits = {js['domains'][dom]['fields'][name]['limit'] for js in cases.values() if dom in js['domains']}
            if len(limits) != 1:
                raise ValueError(f'inconsistent frozen D6 limits for {dom}/{name}')
            finish_axis(ax,stats,next(iter(limits)),name=='RAINNC_hourly')
            ax.set(title=f'{dom} · n={stats["n_cases"]}', xlabel='forecast lead [h]', ylabel=f'RMSE [{unit}]', xlim=(0, 72))
            ax.grid(alpha=.2)
            # Separate all-cases version per domain for the binding release page.
            one, domain_ax = plt.subplots(figsize=(9.2, 5.8))
            for label, values in zip(stats['cases'], stats['per_case']):
                domain_ax.plot(x, values, lw=.9, alpha=.65, label=label)
            if stats['sample_sd'] is not None:
                domain_ax.fill_between(x, np.maximum(0,mean-sd), mean+sd, color='#4383b9', alpha=.22,
                                       label='cross-case mean ± 1 sample SD (lower clipped at 0)')
            domain_ax.plot(x, mean, color='#083e74', lw=2.2, label='equal-case mean')
            finish_axis(domain_ax,stats,next(iter(limits)),name=='RAINNC_hourly')
            domain_ax.set(xlabel='forecast lead [h]', ylabel=f'RMSE [{unit}]', xlim=(0,72))
            domain_ax.grid(alpha=.2)
            one.suptitle(f'{a.title} · {dom} · {name} · all {stats["n_cases"]} available cases', x=.02, ha='left', fontsize=12)
            one.legend(*domain_ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.94),
                       ncol=4, fontsize=7, frameon=False)
            legend_rows=math.ceil(len(domain_ax.get_legend_handles_labels()[0])/4)
            one.tight_layout(rect=(0,.06,1,.94-.025*legend_rows-.01))
            one.text(.02,.015,'Mean ± 1 sample SD; display lower band clipped at 0 for RMSE (stored mean/SD unchanged). '+
                     ('PARTIAL case set.' if missing else 'Complete declared case set.'),fontsize=8)
            individual = a.out / f'{dom}_{name}_all_cases.png'
            one.savefig(individual,dpi=140);plt.close(one);domain_images[dom].append(individual.name)
        status = 'PARTIAL — missing ' + ', '.join(missing) if missing else 'complete declared case set'
        fig.suptitle(f'{a.title} · {name} · {status}', x=.02, ha='left', fontsize=12)
        legend = {}
        for ax in axes.flat:
            for handle,label in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(label,handle)
        fig.legend(list(legend.values()),list(legend),loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,fontsize=7,frameon=False)
        legend_rows=math.ceil(len(legend)/5)
        fig.tight_layout(rect=(0, .06, 1, .94-.025*legend_rows-.01))
        fig.text(.02, .01, 'One sample SD across cases at each hour; lower display band clipped at 0 for RMSE. Stored mean/SD unchanged; not a confidence interval.', fontsize=8)
        target = a.out / f'{name}_cross_case.png'
        fig.savefig(target, dpi=140);plt.close(fig);images.append(target.name)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    pics = ''.join(f'<h2>{html.escape(p)}</h2><a href="{p}"><img src="{p}" loading="lazy" style="width:100%;height:auto" alt="{p}"></a>' for p in images)
    domains_html = ''.join(f'<a href="{dom}_all_cases.html">{dom}: all cases</a> · ' for dom in domains)
    (a.out / 'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(a.title)}</title><body style="font:16px system-ui;margin:2rem"><h1>{html.escape(a.title)}</h1><p>Cases: {html.escape(", ".join(cases))}. Missing: {html.escape(", ".join(missing) or "none")}. Pointwise equal-case mean ± 1 sample SD(ddof=1). n=1 has no estimable SD.</p><a href="summary.json">Plotted data</a><p>{domains_html}</p>{pics}</body>')
    for dom,pngs in domain_images.items():
        body=''.join(f'<h2>{html.escape(p)}</h2><img src="{p}" loading="lazy" style="width:100%;height:auto" alt="{p}">' for p in pngs)
        (a.out/f'{dom}_all_cases.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{dom} all cases</title><body style="font:16px system-ui;margin:2rem"><h1>{html.escape(a.title)} · {dom}: all cases</h1><p>Every included case on each axis; equal-case mean ± 1 sample SD. Missing overall cases: {html.escape(", ".join(missing) or "none")}.</p><a href="index.html">All domains</a>{body}</body>')
    print(f'{len(images)} summary PNGs + HTML; missing_cases={missing}')

if __name__ == '__main__':
    main()
