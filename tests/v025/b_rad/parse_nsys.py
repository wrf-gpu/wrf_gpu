"""Sanitized per-call counts from node-traced nsys warm NVTX ranges."""

import argparse
import json
from pathlib import Path
import sqlite3


def parse(path, repeats):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges = db.execute(
        "SELECT e.start, e.end, COALESCE(e.text, s.value) FROM NVTX_EVENTS e "
        "LEFT JOIN StringIds s ON e.textId=s.id WHERE e.end IS NOT NULL "
        "AND COALESCE(e.text,s.value) LIKE 'BR_%'").fetchall()
    if not ranges:
        raise ValueError('No warm BR_ NVTX ranges: cannot report per-call device metrics')
    rows = []
    for start, end, label in ranges:
        row = dict(range=label, repeats=repeats, kernels=0, copies=0, memsets=0,
                   device_ms=0, transfers={})
        for table, key in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernels'),
                           ('CUPTI_ACTIVITY_KIND_MEMCPY', 'copies'),
                           ('CUPTI_ACTIVITY_KIND_MEMSET', 'memsets')]:
            if table not in tables:
                continue
            count, duration = db.execute(
                f'SELECT count(*), coalesce(sum(end-start),0) FROM {table} '
                'WHERE start>=? AND end<=?', (start, end)).fetchone()
            row[key] = count / repeats
            row['device_ms'] += duration / repeats / 1e6
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            for kind, count, size in db.execute(
                    'SELECT copyKind,count(*),sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                    'WHERE start>=? AND end<=? GROUP BY copyKind', (start, end)):
                row['transfers'][str(kind)] = dict(count=count / repeats, bytes=size / repeats)
        row['ops'] = row['kernels'] + row['copies'] + row['memsets']
        if not row['kernels']:
            raise ValueError(f'No node-traced kernels within {label}')
        rows.append(row)
    db.close()
    return dict(schema='b-rad.nsys-warm-node.v1', raw_path=str(path), ranges=rows,
                denominator='one isolated full radiation family call; cadence excluded',
                memcpy_kind='CUPTI enum: 1 HtoD, 2 DtoH, 8 DtoD; numeric otherwise')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(parse(args.sqlite, args.repeats), indent=2) + '\n')
