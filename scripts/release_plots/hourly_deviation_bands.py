#!/usr/bin/env python3
"""Full-array hourly GPU/CPU deviations, spatial bands, spread and PNG/HTML."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import time
import textwrap

import numpy as np
from netCDF4 import Dataset

from wrfout_pairs import pairs, frames, _valid

FIELDS = [('T2', 'K', 1.), ('PSFC', 'Pa', 1.), ('U10', 'm/s', 1.),
          ('V10', 'm/s', 1.), ('RAINNC', 'mm', 1.),
          ('RAINNC_hourly', 'mm', 1.), ('T', 'K', 1.), ('U', 'm/s', 1.),
          ('V', 'm/s', 1.), ('W', 'm/s', 1.), ('QVAPOR', 'g/kg', 1000.)]

def quiet():
    while Path('/tmp/wrf_gpu2_quiet').exists():
        print('QUIET: extraction/render paused', flush=True)
        time.sleep(20)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def metrics(cpu, gpu):
    """Validate both entire operands before any reduction; no masks."""
    if cpu.shape != gpu.shape:
        raise ValueError('incompatible complete field shapes')
    if not (np.isfinite(cpu).all() and np.isfinite(gpu).all()):
        raise ValueError('nonfinite complete operand')
    diff = gpu.astype(np.float64) - cpu.astype(np.float64)
    if not np.isfinite(diff).all():
        raise ValueError('nonfinite complete difference')
    q = np.quantile(np.abs(diff), [.05, .25, .75, .95], method='linear')
    return {'n': diff.size, 'rmse': float(np.sqrt(np.mean(diff * diff))),
            'bias': float(diff.mean()), 'p05_abs': float(q[0]),
            'p25_abs': float(q[1]), 'p75_abs': float(q[2]), 'p95_abs': float(q[3])}

def spread_ratio(rmse, pair_rmse):
    """No artificial floor; a zero/missing pair leaves the ratio undefined."""
    if pair_rmse is None:
        return None, 'pair unavailable'
    if pair_rmse == 0:
        return None, 'zero pair; both zero' if rmse == 0 else 'zero pair; nonzero GPU error'
    return rmse / pair_rmse, 'defined'

def validate_metrics(result):
    """Validate the complete stored payload before rendering/reducing any metric.

    JSON can contain NaN/Inf even when the original extractor rejected them.
    None is reserved for absent/zero-denominator pair statistics, and the
    initial hourly-rain row intentionally has no numeric statistics.
    """
    import math
    if not isinstance(result, dict) or not isinstance(result.get('domains'), dict) or not result['domains']:
        raise ValueError('missing metric domains')
    declared = result.get('expected_domains')
    if declared is not None and set(declared) != set(result['domains']):
        raise ValueError('incomplete declared metric domains')
    expected_hours = result.get('expected_hours', 73)
    if type(expected_hours) is not int or expected_hours not in (25, 73):
        raise ValueError('declared metric coverage must be 25 or 73 hours')
    required = {name for name, _, _ in FIELDS}
    core = ('rmse', 'bias', 'p05_abs', 'p25_abs', 'p75_abs', 'p95_abs')
    checked = []
    def finite(value, label):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'missing/nonfinite metric {label}')
    for dom, entry in result['domains'].items():
        fields = entry.get('fields')
        if not isinstance(fields, dict) or not required <= set(fields):
            raise ValueError(f'missing required metric field: {dom}')
        for name in required:
            field = fields[name]
            for key in ('limit', 'scale'):
                finite(field.get(key), f'{dom}/{name}/{key}')
            if field['limit'] <= 0 or field['scale'] <= 0:
                raise ValueError(f'nonpositive metric scale/limit {dom}/{name}')
            rows = field.get('by_lead')
            if not isinstance(rows, list) or len(rows) != expected_hours:
                raise ValueError(f'incomplete {expected_hours}-hour metrics {dom}/{name}')
            if any(type(r.get('lead_h')) is not int or r['lead_h'] != h for h, r in enumerate(rows)):
                raise ValueError(f'missing/duplicate/out-of-order metric hours {dom}/{name}')
            for h, row in enumerate(rows):
                label = f'{dom}/{name}/{h}'
                if name == 'RAINNC_hourly' and h == 0:
                    if not str(row.get('status', '')).startswith('undefined') or any(k in row for k in (*core, 'n', 'pair_rmse', 'spread_ratio')):
                        raise ValueError('tau0 hourly rain must remain explicitly undefined')
                    continue
                for key in core:
                    finite(row.get(key), f'{label}/{key}')
                if type(row.get('n')) is not int or row['n'] <= 0:
                    raise ValueError(f'invalid cell count {label}')
                if 'pair_rmse' not in row:
                    raise ValueError(f'missing pair availability {label}')
                if row['pair_rmse'] is not None:
                    finite(row['pair_rmse'], f'{label}/pair_rmse')
                if 'spread_ratio' in row and row['spread_ratio'] is not None:
                    finite(row['spread_ratio'], f'{label}/spread_ratio')
                checked.append((label, row))
    # All complete operands are finite before consistency comparisons/reductions.
    for label, row in checked:
        if row['rmse'] < 0 or not (0 <= row['p05_abs'] <= row['p25_abs'] <= row['p75_abs'] <= row['p95_abs']):
            raise ValueError(f'invalid error distribution {label}')
        if row['pair_rmse'] is not None and row['pair_rmse'] < 0:
            raise ValueError(f'negative pair RMSE {label}')
        if 'spread_ratio' in row:
            expected, _ = spread_ratio(row['rmse'], row['pair_rmse'])
            if expected is None:
                if row['spread_ratio'] is not None:
                    raise ValueError(f'ratio must be undefined {label}')
            elif row['spread_ratio'] is None or not math.isclose(row['spread_ratio'], expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'inconsistent stored pair ratio {label}')
    return result

def check_time(ds, path):
    ds.set_auto_mask(False)
    times = [''.join(row.astype('U1')).strip('\x00') for row in ds['Times'][:]]
    if len(times) != 1:
        raise ValueError(f'one internal Time required: {path}')
    raw = dt.datetime.strptime(times[0], '%Y-%m-%d_%H:%M:%S')
    internal = (raw + dt.timedelta(minutes=30)).replace(minute=0, second=0)
    if internal != _valid(path.name)[1]:
        raise ValueError(f'filename/internal valid time mismatch: {path}')
    return times[0]

def strict_pairs(cpu, gpu, dom, expected_hours=73):
    for directory in (cpu, gpu):
        matched = [p for p in Path(directory).iterdir() if _valid(p.name)[0] == dom]
        if len(matched) != len(frames(directory, dom)):
            raise ValueError(f'duplicate nominal hour in {directory}/{dom}')
    result = pairs(cpu, gpu, dom)
    if len(result) != expected_hours:
        raise ValueError(f'{dom}: require all{expected_hours} pairs, got {len(result)}')
    start = result[0][0]
    if [int((r[0] - start).total_seconds()) for r in result] != list(range(0, expected_hours * 3600, 3600)):
        raise ValueError(f'{dom}: incomplete hourly sequence')
    return result

def extract(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    tolerance = json.loads(a.manifest.read_text())
    result = {'schema': 'hourly-deviation-spatial-bands-v1', 'case': a.case,
              'gpu_tree': a.gpu_tree, 'scope': a.scope,
              'expected_domains': a.domains, 'expected_hours': getattr(a, 'expected_hours', 73),
              'finite_first': True, 'mask': 'none', 'quantile_method': 'linear',
              'manifest_sha256': sha(a.manifest), 'tool_sha256': sha(Path(__file__)),
              'pair_kind': a.pair_label if a.pair else None,
              'pair_reference': a.pair_reference if a.pair else None, 'domains': {}}
    start_all = None
    for dom in a.domains:
        group = strict_pairs(a.cpu, a.gpu, dom, result['expected_hours'])
        if start_all is None:
            start_all = group[0][0]
        assert group[0][0] == start_all, 'domains start at different valid hours'
        pairmap = frames(a.pair, dom) if a.pair else {}
        previous = None
        fields = {n: {'unit': unit, 'scale': scale,
                      'limit': tolerance['fields'][n if n != 'RAINNC_hourly' else 'RAINNC']['rmse'] * scale,
                      'limit_kind': 'D6 accumulated reference only' if n == 'RAINNC_hourly' else 'frozen D6 RMSE',
                      'by_lead': []} for n, unit, scale in FIELDS}
        sources = []
        reference = json.loads((a.binding_report / f'{dom}.json').read_text()) if a.binding_report else None
        binding = {name: {r['lead_h']: r for r in fs['by_lead']}
                   for name, fs in reference['field_summaries'].items()
                   if name in fields and name != 'RAINNC_hourly'} if reference else {}
        for lead, (valid, cpath, gpath) in enumerate(group):
            quiet()
            pairpath = pairmap.get(valid)
            pds = Dataset(pairpath) if pairpath else None
            try:
                with Dataset(cpath) as cds, Dataset(gpath) as gds:
                    ct, gt = check_time(cds, cpath), check_time(gds, gpath)
                    if pds:
                        check_time(pds, pairpath)
                    vals = {}
                    for name, _, scale in FIELDS:
                        quiet()
                        key = 'RAINNC' if name == 'RAINNC_hourly' else name
                        if cds[key].dimensions != gds[key].dimensions:
                            raise ValueError(f'{dom}/{name}: native dimensions differ')
                        if name != 'RAINNC_hourly':
                            c = np.asarray(cds[key][:], dtype=np.float64)
                            g = np.asarray(gds[key][:], dtype=np.float64)
                            p = np.asarray(pds[key][:], dtype=np.float64) if pds else None
                            # E200 must cover pair operands too, before any statistics.
                            if p is not None and (p.shape != c.shape or not np.isfinite(p).all()):
                                raise ValueError(f'nonfinite/incompatible full predictability operand {dom}/{name}/{lead}')
                            vals[name] = (c, g, p)
                        else:
                            if previous is None:
                                fields[name]['by_lead'].append({'lead_h': 0, 'status': 'undefined: no preceding hour'})
                                continue
                            c, g, p = vals['RAINNC']
                            c0, g0, p0 = previous
                            c, g = c - c0, g - g0
                            p = p - p0 if p is not None and p0 is not None else None
                        row = metrics(c * scale, g * scale)
                        row.update(lead_h=lead, valid_time=valid.isoformat(), pair_rmse=None)
                        if p is not None:
                            base = c if a.pair_reference == 'cpu' else g
                            row['pair_rmse'] = metrics(base * scale, p * scale)['rmse']
                        row['spread_ratio'], row['spread_ratio_status'] = spread_ratio(row['rmse'], row['pair_rmse'])
                        if binding and name != 'RAINNC_hourly':
                            base = binding[name][lead]
                            # Independent raw-cell computation must reproduce binding whole-grid reductions.
                            for k in ('rmse', 'bias'):
                                if not np.isclose(row[k], base[k] * scale, rtol=2e-10, atol=2e-10):
                                    raise ValueError(f'binding statistic mismatch {dom}/{name}/{lead}/{k}: {row[k]} vs {base[k]*scale}')
                        fields[name]['by_lead'].append(row)
                    previous = vals['RAINNC']
                    sources.append({'lead_h': lead, 'cpu': str(cpath), 'gpu': str(gpath),
                                    'cpu_sha256': sha(cpath), 'gpu_sha256': sha(gpath),
                                    'pair': str(pairpath) if pairpath else None,
                                    'pair_sha256': sha(pairpath) if pairpath else None,
                                    'cpu_internal_time': ct, 'gpu_internal_time': gt})
            finally:
                if pds:
                    pds.close()
            result['domains'][dom] = {'fields': fields, 'sources': sources,
                                      'pair_missing_leads': [i for i, r in enumerate(sources) if r['pair'] is None]}
            (out / 'metrics.private.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
            if lead % 12 == 0 or lead == 72:
                print(f'{dom}: {lead}/72 complete, finite-first, binding metrics match', flush=True)
    # Public numerical data retain source hashes, not private filesystem paths.
    public = json.loads(json.dumps(result))
    for entry in public['domains'].values():
        for source in entry['sources']:
            for key in ('cpu', 'gpu', 'pair'):
                source[key] = Path(source[key]).name if source[key] else None
    (out / 'metrics.json').write_text(json.dumps(public, indent=1, allow_nan=False) + '\n')
    return result

def render(result, out):
    validate_metrics(result)
    quiet()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    domains = list(result['domains'])
    expected_hours = result.get('expected_hours', 73)
    last_hour = expected_hours - 1
    tick_step = 12 if last_hour == 72 else 6
    has_pair = any(r.get('pair_rmse') is not None for entry in result['domains'].values() for field in entry['fields'].values() for r in field['by_lead'])
    near_zero_pair = 'GPU IC' if result.get('pair_reference') == 'gpu' else 'CPU'
    ratio_note = result.get('ratio_note', 'Ratio 2 = 50% pair/RC rule; a single pair is not a confidence interval.')
    ratio_scope = result.get('ratio_scope', f'Ratio > 2 at small absolute error is expected where the {near_zero_pair} pair is near zero (early leads, d01); the 50% rule is applied only to D6 exceedance frames.')
    ratio_guide_label = result.get('ratio_guide_label', '(the 50% predictability rule)')
    pair_name = 'GPU IC-pair' if result.get('pair_reference') == 'gpu' else 'CPU IC-pair'
    images = []
    for name, unit, _ in FIELDS:
        quiet()
        fig, axes = plt.subplots(3, len(domains), figsize=(5.4 * len(domains), 10.2), squeeze=False, sharex=True)
        for i, dom in enumerate(domains):
            f = result['domains'][dom]['fields'][name]
            rows = [r for r in f['by_lead'] if 'rmse' in r]
            x = [r['lead_h'] for r in rows]
            upper, lower, ratio_ax = axes[0, i], axes[1, i], axes[2, i]
            upper.fill_between(x, [r['p05_abs'] for r in rows], [r['p95_abs'] for r in rows], color='#417fb5', alpha=.17, label='spatial |error| p5–p95')
            upper.fill_between(x, [r['p25_abs'] for r in rows], [r['p75_abs'] for r in rows], color='#417fb5', alpha=.35, label='spatial |error| p25–p75')
            px = [r['lead_h'] for r in rows if r['pair_rmse'] is not None]
            py = [r['pair_rmse'] for r in rows if r['pair_rmse'] is not None]
            if px:
                upper.plot(px, py, color='#666666', lw=2, label=f'{pair_name} RMSE (empirical spread)')
            else:
                ratio_ax.text(.02, .98, 'IC pair unavailable', transform=ratio_ax.transAxes, va='top', fontsize=8)
            upper.plot(x, [r['rmse'] for r in rows], color='#075297', lw=2, label='GPU − CPU RMSE')
            ymax = max([r['p95_abs'] for r in rows] + [r['rmse'] for r in rows] + py) * 1.15
            ymax = ymax if ymax > 0 else f['limit'] * .05
            upper.set_ylim(0, ymax)
            if f['limit'] <= ymax:
                upper.axhline(f['limit'], ls='--', color='#b63231', label=f['limit_kind'])
            else:
                upper.annotate(f'D6 {f["limit"]:g} {unit} above' + (' (reference)' if name == 'RAINNC_hourly' else ''),
                               xy=(.97, 1.), xycoords='axes fraction', xytext=(.97, 1.04), textcoords='axes fraction',
                               ha='right', va='bottom', fontsize=8, color='#b63231',
                               arrowprops={'arrowstyle': '->', 'color': '#b63231'})
            if name != 'RAINNC_hourly':
                breaches = [r for r in rows if r['rmse'] > f['limit']]
                if breaches:
                    upper.scatter([r['lead_h'] for r in breaches], [r['rmse'] for r in breaches],
                                  marker='o', s=28, facecolors='none', edgecolors='#b63231', lw=1.3,
                                  zorder=5, label='D6 exceedance (classification in evidence)')
            upper.set_title(dom, loc='left', fontweight='bold')
            upper.set_ylabel(f'RMSE / absolute spatial error [{unit}]')
            lower.plot(x, [r['bias'] for r in rows], color='#c16f16', lw=2, label='GPU − CPU signed mean bias')
            lower.axhline(0, color='#777777', lw=.8)
            lower.set_ylabel(f'mean bias [{unit}]')
            ratios = [spread_ratio(r['rmse'], r['pair_rmse']) for r in rows]
            ratio_ax.plot(x, [v if v is not None else np.nan for v, _ in ratios], color='#704798', lw=1.7)
            ratio_ax.axhline(1., color='#777777', lw=.9, ls=':')
            ratio_ax.axhline(2., color='#b63231', lw=1., ls='--')
            if any(status == 'zero pair; nonzero GPU error' for _, status in ratios):
                ratio_ax.text(.02, .98, f'Some ratios undefined: {pair_name} RMSE = 0', transform=ratio_ax.transAxes,
                              va='top', fontsize=7)
            ratio_ax.set_ylabel(f'RMSE / {pair_name} RMSE\nreferences 1 and 2')
            ratio_ax.set_xlabel('forecast lead [h]')
            ratio_ax.set_xlim(0, last_hour)
            ratio_ax.set_xticks(range(0, expected_hours, tick_step))
            ratio_ax.set_ylim(bottom=0)
            for ax in (upper, lower, ratio_ax):
                ax.grid(alpha=.18)
        note = 'Hourly increment: τ0 undefined; dashed accumulated D6 reference is informational.' if name == 'RAINNC_hourly' else 'All native cells; spatial bands are distributions, not confidence intervals.'
        fig.suptitle(f'{result["case"]} · {name} · measured {result["gpu_tree"][:12]}\nvs original CPU-WRF [M]', fontsize=14, x=.02, ha='left', y=.995)
        legend = {}
        for ax in axes.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(label, handle)
        fig.legend(list(legend.values()), list(legend), loc='upper center', ncol=3, fontsize=8,
                   bbox_to_anchor=(.5, .962), frameon=False)
        caption_parts = [note + ' ' + ratio_note, ratio_scope]
        if result.get('validation_classification'):
            caption_parts.append(result['validation_classification'])
        caption = '\n'.join(textwrap.fill(part, width=int(fig.get_figwidth()*14)) for part in caption_parts)
        fig.text(.02, .005, caption, fontsize=7, va='bottom')
        fig.tight_layout(rect=(0, .105, 1, .88))
        target = out / f'{name}_bands.png'
        fig.savefig(target, dpi=140)
        plt.close(fig)
        images.append(target.name)
    quiet()
    bounded = [n for n, _, _ in FIELDS if n != 'RAINNC_hourly']
    fig, axes = plt.subplots(len(domains), 1, figsize=(15, 3.8 * len(domains)), squeeze=False)
    for i, dom in enumerate(domains):
        data = np.asarray([[r['rmse'] / result['domains'][dom]['fields'][n]['limit'] for r in result['domains'][dom]['fields'][n]['by_lead']] for n in bounded])
        ax = axes[i, 0]
        m = ax.imshow(data, aspect='auto', origin='upper', cmap='YlOrRd', vmin=0, vmax=max(1., float(data.max())))
        y, x = np.where(data > 1)
        ax.scatter(x, y, marker='x', color='#111111', s=10)
        ax.set_yticks(range(len(bounded)), bounded)
        ax.set_xticks(range(0, expected_hours, tick_step))
        ax.set_title(dom, loc='left')
        ax.set_xlabel('forecast lead [h]')
        fig.colorbar(m, ax=ax, label='RMSE / frozen D6 limit; × = exceedance')
    fig.suptitle(f'{result["case"]} · hourly deviation heatmap · all {expected_hours} hours', x=.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out / 'heatmap.png', dpi=140)
    plt.close(fig)
    images.insert(0, 'heatmap.png')
    # One-case prototype summary: show every hour; SD is not estimable at n=1.
    for name, unit, _ in FIELDS:
        quiet()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for dom in domains:
            rows = [r for r in result['domains'][dom]['fields'][name]['by_lead'] if 'rmse' in r]
            ax.plot([r['lead_h'] for r in rows], [r['rmse'] for r in rows], label=f'{dom}: case RMSE = cross-case mean (n=1)')
        limit = result['domains'][domains[0]]['fields'][name]['limit']
        ax.axhline(limit, color='#b63231', ls='--', lw=1,
                   label='accumulated D6 reference only' if name == 'RAINNC_hourly' else 'frozen D6 RMSE limit')
        ax.set(xlabel='forecast lead [h]', ylabel=f'RMSE [{unit}]', title=f'{name} · per-case summary (n=1; cross-case SD unavailable here)', xlim=(0, last_hour))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        path = out / f'{name}_summary.png';fig.savefig(path, dpi=140);plt.close(fig);images.append(path.name)
    body = '\n'.join(f'<section><h2>{html.escape(p.removesuffix(".png"))}</h2><a href="{p}"><img loading="lazy" src="{p}" alt="{html.escape(p)}"></a></section>' for p in images)
    pair_caption = (f'Grey line: {pair_name} empirical RMSE where available. Third row: GPU/CPU RMSE divided by pair RMSE, references 1 and 2 {html.escape(ratio_guide_label)}. Missing/zero pair denominators are undefined; no artificial floor. Pair spread is context, not a confidence interval or gate waiver.' if has_pair else 'No paired comparator is available for this standalone diagnostic GPU-versus-original-CPU comparison. No grey pair-spread curve or pair-RMSE ratio is reported.')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(result['case'])} hourly deviation bands</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1600px}}img{{width:100%;height:auto}}section{{margin:3rem 0}}a{{color:#075297}}</style>
<h1>{html.escape(result['case'])}: all {expected_hours} hours, spatial deviation bands</h1>
<p>{html.escape(result['scope'])}; measured tree {html.escape(result['gpu_tree'])}. All stored cells, no masks, finite-first. Blue line: GPU-versus-original-CPU RMSE. Orange: signed mean bias. Dark/light blue: spatial |error| p25–p75/p5–p95. {pair_caption} Dashed/off-scale arrow: frozen D6 RMSE limit; hourly rain uses the accumulated limit as an informational reference.</p>
<p>{html.escape(result.get('validation_classification', ''))}</p>
<p>{html.escape(ratio_note)} {html.escape(ratio_scope)}</p>
<p>Summary n=1: cross-case sample SD is unavailable. Final summaries will show all cases and pointwise mean ± one sample SD per domain.</p>
<p><a href="metrics.json">Plotted numerical data and source hashes</a></p>{body}</html>'''
    (out / 'index.html').write_text(page)
    print(f'rendered {len(images)} PNGs + HTML + public/private JSON', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu', type=Path)
    p.add_argument('--gpu', type=Path)
    p.add_argument('--pair', type=Path)
    p.add_argument('--pair-reference', choices=['cpu', 'gpu'], default='cpu',
                   help='spread = original CPU vs CPU IC member, or main GPU vs GPU IC member')
    p.add_argument('--pair-label', default='original CPU-WRF vs P0227, theta +/-2^-15 K')
    p.add_argument('--case', default='0227')
    p.add_argument('--gpu-tree', help='required measured model revision for extraction')
    p.add_argument('--scope', default='Validation candidate; gate verdict reported separately')
    p.add_argument('--domains', nargs='+', default=['d01', 'd02', 'd03'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--binding-report', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('CPU-only: set JAX_PLATFORMS=cpu')
    if not a.render_only and not all((a.cpu, a.gpu, a.gpu_tree)):
        raise ValueError('extraction requires --cpu, --gpu and the measured --gpu-tree')
    result = json.loads((a.out / 'metrics.private.json').read_text()) if a.render_only else extract(a)
    render(result, a.out)

if __name__ == '__main__':
    main()
