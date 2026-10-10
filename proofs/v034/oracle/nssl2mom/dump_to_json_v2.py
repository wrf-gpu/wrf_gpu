#!/usr/bin/env python3
"""Parse the v034 instrumented NSSL oracle dump into a JSON savepoint.

schema wrf-v034-o1-nssl2mom-column-savepoint-v2:
  scalars  : CASE KX ITIMESTEP DT CCN_IS_CCNA QCCN RAINNCV SNOWNCV GRPLNCV HAILNCV SR
  columns  : <NAME> -> list over k=1..KX (inputs *_IN, PII P DZ W DN, outputs *_OUT, DBZ/RE)
  stages   : S0..S6 -> {"an": [na][KX], "<col>": [KX], "xfall": [na]}  (density-scaled #/m3 units
             inside the driver; S0 = after pack + denscale, S1 = after calcnfromq, S2 = after
             sediment1d, S3 = after nssl_2mom_gs, S4 = after NUCOND, S5 = after smallvalues,
             S6 = radardd02 dbz2d)
  module_vars (case 1 only): name -> list of values after nssl_2mom_init
"""
import json, re, sys

SCHEMA = 'wrf-v034-o1-nssl2mom-column-savepoint-v2'
SCALARS = {'CASE', 'KX', 'ITIMESTEP', 'DT', 'CCN_IS_CCNA', 'QCCN', 'RAINNCV', 'SNOWNCV', 'GRPLNCV', 'HAILNCV', 'SR'}
INT_SCALARS = {'CASE', 'KX', 'ITIMESTEP', 'CCN_IS_CCNA'}
COL_RE = re.compile(r'^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$')
SCAL_RE = re.compile(r'^([A-Z0-9_]+)=\s*(.+)$')
AN_RE = re.compile(r'^(S\d):AN:(\d+):(\d+)=\s*(.+)$')
XF_RE = re.compile(r'^(S\d):XFALL:(\d+)=\s*(.+)$')
SC_RE = re.compile(r'^(S\d):([a-z0-9]+):(\d+)=\s*(.+)$')
MV_RE = re.compile(r'^MV:([A-Za-z0-9_]+)=(.*)$')
MVS_RE = re.compile(r'^MVS:([A-Za-z0-9_]+)=(.*)$')

def num(s):
    s = s.strip()
    if s in ('T', 'F'):
        return s == 'T'
    try:
        return int(s)
    except ValueError:
        return float(s)

def main(infile, outfile, mode):
    scalars, cols, stages, mv, mvs = {}, {}, {}, {}, {}
    for line in open(infile):
        line = line.rstrip('\n')
        if line.startswith(('INIT_FATAL', 'RUN_FATAL', 'FATAL', 'WRF_ERROR_FATAL')):
            raise SystemExit(f'oracle reported fatal: {line}')
        m = MVS_RE.match(line)
        if m:
            vals = m.group(2).split()
            mvs[m.group(1)] = {'size': int(vals[0]), 'sum': float(vals[1]), 'stride997': [float(x) for x in vals[2:]]}
            continue
        m = MV_RE.match(line)
        if m:
            mv[m.group(1)] = [num(x) for x in m.group(2).split()]
            continue
        m = AN_RE.match(line)
        if m:
            st = stages.setdefault(m.group(1), {})
            st.setdefault('an', {}).setdefault(int(m.group(2)), {})[int(m.group(3))] = float(m.group(4))
            continue
        m = XF_RE.match(line)
        if m:
            stages.setdefault(m.group(1), {}).setdefault('xfall', {})[int(m.group(2))] = float(m.group(3))
            continue
        m = SC_RE.match(line)
        if m:
            stages.setdefault(m.group(1), {}).setdefault(m.group(2), {})[int(m.group(3))] = float(m.group(4))
            continue
        m = COL_RE.match(line)
        if m:
            cols.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
            continue
        m = SCAL_RE.match(line)
        if m and m.group(1) in SCALARS:
            scalars[m.group(1)] = int(m.group(2)) if m.group(1) in INT_SCALARS else float(m.group(2))
    kx = scalars['KX']
    out = {'schema': SCHEMA, 'mode': mode, 'scalars': scalars, 'columns': {}, 'stages': {}}
    for name, d in cols.items():
        assert len(d) == kx, name
        out['columns'][name] = [d[k] for k in range(1, kx + 1)]
    for st, d in sorted(stages.items()):
        o = {}
        for key, v in d.items():
            if key == 'an':
                na = max(v)
                o['an'] = [[v[il][k] for k in range(1, kx + 1)] for il in range(1, na + 1)]
            elif key == 'xfall':
                o['xfall'] = [v[il] for il in range(1, max(v) + 1)]
            else:
                o[key] = [v[k] for k in range(1, kx + 1)]
        out['stages'][st] = o
    if mv:
        out['module_vars'] = mv
        out['module_vars_large'] = mvs
    json.dump(out, open(outfile, 'w'))
    print(f'wrote {outfile}: stages={sorted(out["stages"])} mv={len(mv)}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
