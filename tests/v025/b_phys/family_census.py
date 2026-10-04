"""Reduce node-level nsys windows by source-anchored XLA operation scope.

No kernel-name heuristics. Unknown work stays unknown; sums include graph nodes,
stream kernels, copies and memsets. Run with LABEL=SQLITE positional arguments.
Reads only allowlisted performance tables; raw captures remain private.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3

FAMILIES = (
    ('KF', r'kf_eta_para|kf_.*'),
    ('MYNN', r'step_mynn_pbl.*|.*mynn.*'),
    ('THOMPSON', r'step_thompson.*'),
    ('NOAHMP', r'noahmp_.*'),
    ('RRTMG', r'.*rrtmg.*'),
    ('DYCORE', r'advance_w|advance_uv|advance_mu|acoustic.*|small_step.*'),
)


def classify(path):
    if path is None:
        return 'UNKNOWN'
    for fn in reversed(re.findall(r'(?:jit|pjit)\(([^)]*)\)', path)):
        for family, pattern in FAMILIES:
            if re.fullmatch(pattern, fn):
                return family
    return 'UNKNOWN'


def reduce_window(db, kernel_map=None):
    conn = sqlite3.connect(f'file:{Path(db).resolve()}?mode=ro', uri=True)
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    strings = dict(conn.execute('select id,value from StringIds'))
    ranges = defaultdict(list)
    for start, end, tid, text, text_id in conn.execute(
        'select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'
    ):
        text = text if text is not None else strings.get(text_id, '')
        if not text or ('hlo_op=' not in text and 'hlo_module=' not in text):
            continue
        match = re.search(r'[#,]name=([^,#]*)', text)
        ranges[tid].append((start, end, match.group(1) if match else None))
    for values in ranges.values():
        values.sort()
    starts = {tid: [r[0] for r in values] for tid, values in ranges.items()}
    api = {}
    for table in ('CUPTI_ACTIVITY_KIND_DRIVER', 'CUPTI_ACTIVITY_KIND_RUNTIME'):
        if table in tables:
            for start, corr, tid in conn.execute(f'select start,correlationId,globalTid from {table}'):
                api[corr] = start, tid

    def owner(corr):
        if corr not in api:
            return 'UNKNOWN'
        when, tid = api[corr]
        values = ranges.get(tid, ())
        index = bisect.bisect_right(starts.get(tid, ()), when) - 1
        while index >= 0:
            start, end, path = values[index]
            if start <= when <= end:
                return classify(path)
            index -= 1
        return 'UNKNOWN'

    families = defaultdict(Counter)
    totals = Counter()
    kernel_columns = {r[1] for r in conn.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    symbol_col = 'shortName' if 'shortName' in kernel_columns else 'NULL'
    mapped = Counter()
    disagreements = Counter()
    scope_by_symbol = defaultdict(set)
    unscoped_by_symbol = defaultdict(Counter)
    for start, end, corr, graph, symbol in conn.execute(
        f'select start,end,correlationId,graphNodeId,{symbol_col} from CUPTI_ACTIVITY_KIND_KERNEL'
    ):
        if end < start:
            raise ValueError('inverted kernel row')
        scope_family = owner(corr)
        kernel_name = strings.get(symbol, '')
        source_family = (kernel_map or {}).get(kernel_name)
        if scope_family != 'UNKNOWN' and kernel_name:
            scope_by_symbol[kernel_name].add(scope_family)
        if scope_family == 'UNKNOWN' and kernel_name:
            counter = unscoped_by_symbol[kernel_name]
            counter['kernels'] += 1
            counter['kernel_ns'] += end - start
            counter['graph_kernels'] += graph is not None
            counter['ops'] += 1
            counter['device_ns'] += end - start
        if source_family and scope_family != 'UNKNOWN' and source_family != scope_family:
            disagreements[f'{source_family}->{scope_family}'] += 1
        selected = scope_family if scope_family != 'UNKNOWN' else (source_family or 'UNKNOWN')
        if source_family:
            mapped[source_family] += 1
        family = families[selected]
        for counter in (family, totals):
            counter['kernels'] += 1
            counter['kernel_ns'] += end - start
            counter['graph_kernels'] += graph is not None
            counter['ops'] += 1
            counter['device_ns'] += end - start
    for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
        if table not in tables:
            continue
        cols = 'start,end,correlationId' + (',copyKind' if kind == 'memcpy' else '')
        for row in conn.execute(f'select {cols} from {table}'):
            start, end, corr = row[:3]
            if end < start:
                raise ValueError(f'inverted {kind} row')
            for counter in (families[owner(corr)], totals):
                counter[kind] += 1
                counter['ops'] += 1
                counter['device_ns'] += end - start
                if kind == 'memcpy':
                    counter[f'copy_kind_{row[3]}'] += 1
    conn.close()
    for key, value in totals.items():
        assert sum(f[key] for f in families.values()) == value, key
    if not totals['kernels']:
        raise ValueError('capture has no kernel rows')
    return dict(totals=totals, families=dict(families), source=str(db), conservation=True,
                hlo_mapped_kernels=dict(mapped), hlo_scope_disagreement=dict(disagreements),
                scope_by_symbol={k:sorted(v) for k,v in scope_by_symbol.items()},
                unscoped_by_symbol=dict(unscoped_by_symbol))


def mark_shared_symbols(windows, maps):
    # A single emitted kernel can serve several HLO call sites. Known runtime
    # scopes identify the call site; its symbol alone does not.
    observed = defaultdict(lambda: defaultdict(set))
    for label, window in windows.items():
        domain = label.split('_')[0]
        for symbol, families in window['scope_by_symbol'].items():
            observed[domain][symbol].update(families)
    aliases = {}
    for domain, symbols in observed.items():
        aliases[domain] = {}
        for symbol, families in symbols.items():
            canonical = maps.get(domain, {}).get(symbol)
            if canonical and canonical not in ('UNKNOWN', 'MIXED', 'AMBIGUOUS'):
                families.add(canonical)
            if len(families) > 1:
                aliases[domain][symbol] = sorted(families)
    for label, window in windows.items():
        domain = label.split('_')[0]
        for symbol in aliases.get(domain, {}):
            counts = window['unscoped_by_symbol'].get(symbol)
            if not counts:
                continue
            previous = maps.get(domain, {}).get(symbol, 'UNKNOWN')
            target = window['families'].setdefault('AMBIGUOUS', Counter())
            for key, value in counts.items():
                window['families'][previous][key] -= value
                target[key] += value
        for key, total in window['totals'].items():
            assert sum(c[key] for c in window['families'].values()) == total, key
        for counter in window['families'].values():
            assert all(value >= 0 for value in counter.values())
    return aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('windows', nargs='+')
    parser.add_argument('--hlo-map', type=Path)
    args = parser.parse_args()
    output = dict(schema='wrf_gpu2.b_phys.family_census.v1',
                  rule='innermost physics jit scope enclosing launching API; UNKNOWN disclosed',
                  duration='sum of GPU row durations (not wall time)', windows={})
    maps = {}
    if args.hlo_map:
        modules = json.loads(args.hlo_map.read_text())['modules']
        output['hlo_map'] = str(args.hlo_map)
        for domain in ('d01', 'd02'):
            combined = defaultdict(set)
            for module in modules:
                if module.get('domain') == domain:
                    for name, family in module['kernels'].items():
                        combined[name].add(family)
            maps[domain] = {k: next(iter(v)) if len(v) == 1 else 'AMBIGUOUS'
                            for k, v in combined.items()}
    for window in args.windows:
        label, db = window.split('=', 1)
        domain = label.split('_')[0]
        output['windows'][label] = reduce_window(db, maps.get(domain))
    output['shared_symbol_origins'] = mark_shared_symbols(output['windows'], maps)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
