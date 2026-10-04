"""Sum device work inside named NVTX ranges of exported nsys sqlite reports.

Each range must enclose a synchronised call (block_until_ready inside the range),
so every GPU activity launched by an API call inside the range belongs to it.
Counts graph-node kernels, stream kernels, memcpy rows by kind and memsets;
reports per-range sum and busy union. Usage: LABEL=SQLITE ... --range NAME.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import statistics

COPY_KIND = {1: 'HtoD', 2: 'DtoH', 8: 'DtoD', 10: 'PtoP'}


def union_ns(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def reduce_db(db, name):
    conn = sqlite3.connect(f'file:{Path(db).resolve()}?mode=ro', uri=True)
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    strings = dict(conn.execute('select id,value from StringIds'))
    ranges = sorted(
        (start, end) for start, end, text, text_id in conn.execute(
            'select start,end,text,textId from NVTX_EVENTS where end is not null')
        if (text if text is not None else strings.get(text_id)) == name)
    api = []
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            api.extend(conn.execute(f'select start,end,correlationId from {table}'))
    corr_range = {}
    for start, end, corr in api:
        for i, (rs, re_) in enumerate(ranges):
            if rs <= start and end <= re_:
                corr_range[corr] = i
                break
    rows = [[] for _ in ranges]
    if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
        for s, e, corr, graph in conn.execute(
                'select start,end,correlationId,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL'):
            if corr in corr_range:
                rows[corr_range[corr]].append(('kernel_graph' if graph else 'kernel_stream', s, e, 0))
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for s, e, corr, kind, nbytes in conn.execute(
                'select start,end,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
            if corr in corr_range:
                rows[corr_range[corr]].append((COPY_KIND.get(kind, f'copy{kind}'), s, e, nbytes))
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        for s, e, corr, nbytes in conn.execute(
                'select start,end,correlationId,bytes from CUPTI_ACTIVITY_KIND_MEMSET'):
            if corr in corr_range:
                rows[corr_range[corr]].append(('memset', s, e, nbytes))
    per = []
    for (rs, re_), items in zip(ranges, rows):
        kinds = {}
        for kind, s, e, nbytes in items:
            k = kinds.setdefault(kind, {'count': 0, 'ns': 0, 'bytes': 0})
            k['count'] += 1
            k['ns'] += e - s
            k['bytes'] += nbytes
        per.append({
            'range_ns': re_ - rs,
            'ops': len(items),
            'device_sum_ns': sum(e - s for _, s, e, _ in items),
            'device_busy_ns': union_ns([(s, e) for _, s, e, _ in items]),
            'kernel_ns': sum(e - s for k, s, e, _ in items if k.startswith('kernel')),
            'max_kernel_ns': max([e - s for k, s, e, _ in items if k.startswith('kernel')], default=0),
            'kinds': kinds,
        })
    med = {key: statistics.median(p[key] for p in per) for key in
           ('range_ns', 'ops', 'device_sum_ns', 'device_busy_ns', 'kernel_ns', 'max_kernel_ns')} if per else {}
    return {'db': str(db), 'range': name, 'n_ranges': len(ranges), 'median': med, 'per_range': per}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', help='LABEL=SQLITE')
    parser.add_argument('--range', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = {}
    for item in args.reports:
        label, db = item.split('=', 1)
        result[label] = reduce_db(db, args.range)
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(json.dumps({k: {'n': v['n_ranges'], **v['median']} for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
