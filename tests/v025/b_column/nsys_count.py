"""Count device ops (kernels + memcpy + memset rows, graph nodes expanded) in an nsys capture.

usage: nsys_count.py report.nsys-rep out.json
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from collections import Counter


def main():
    rep, out = sys.argv[1], sys.argv[2]
    text = subprocess.run(
        ['/usr/local/cuda/bin/nsys', 'stats', '--report', 'cuda_gpu_trace', '--format', 'csv',
         '--force-export=true', '--quiet', rep],
        capture_output=True, text=True, timeout=600, check=True).stdout
    start = text.find('"Start (ns)"')
    if start < 0:
        start = text.find('Start (ns)')
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    kinds = Counter()
    dur = Counter()
    names = Counter()
    for r in rows:
        name = r.get('Name', '')
        kind = 'memcpy' if '[CUDA memcpy' in name else 'memset' if '[CUDA memset' in name else 'kernel'
        kinds[kind] += 1
        dur[kind] += int(float(r.get('Duration (ns)', 0) or 0))
        if kind == 'kernel':
            names[name[:60]] += 1
    t0 = min(int(float(r['Start (ns)'])) for r in rows) if rows else 0
    t1 = max(int(float(r['Start (ns)'])) + int(float(r['Duration (ns)'])) for r in rows) if rows else 0
    res = dict(rows=len(rows), kinds=dict(kinds), device_ns=dict(dur), span_ns=t1 - t0,
               device_busy_us=sum(dur.values()) / 1e3, top_kernels=names.most_common(15))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
