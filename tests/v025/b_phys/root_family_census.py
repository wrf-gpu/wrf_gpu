"""Family census of one profiled root from rich optimized HLO (CPU only).

Kernel rows inside an NVTX root range are assigned to an executable by the
XlaModule range enclosing their launching API call, then to the HLO
fusion/custom-call of the same name in that executable's metadata-rich
optimized module (e.g. extracted from the JAX cache), then to a family by the
source-file stack of the fused instructions (majority; MIXED below 70 %).
Kernels of other executables are classified by their NVTX thunk name.
Usage: --db S --root NAME --module PROGRAM_ID=RICH_HLO ... --out J
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
    ('Thompson', ('thompson',)),
    ('MYNN', ('mynn', 'edmf', 'boulac')),
    ('KF', ('cumulus_kf', 'kf_')),
    ('RRTMG', ('rrtmg', 'rad_mcica', 'clwrf', 'radiation', 'cam_ozone', 'solar')),
    ('Noah-MP', ('noahmp', 'noah_mp')),
    ('sfclay', ('surface_layer', 'sfclay')),
    ('GWDO', ('gwdo',)),
    ('boundary', ('boundary', 'nesting/', 'interp_sint', 'field_sides')),
    ('dycore', ('dynamics/', 'kernels/dyn_', 'b_core', 'acoustic', 'small_step', 'advance_', 'flux_advection',
                'explicit_diffusion', 'rk_', 'fused_vertical')),
    ('coupling', ('physics_couplers', 'scan_adapters', 'coupling/')),
    ('runtime', ('runtime/', 'operational_mode')),
)
INSTR = re.compile(r'^\s*(?:ROOT )?%?([\w.\-]+) = .*? (fusion|custom-call)\((.*)$')


def classify(text):
    text = text.lower()
    for family, tokens in FAMILIES:
        if any(t in text for t in tokens):
            return family
    return None


def load_module(path):
    """Return {kernel_name: (family, share, n_instr, top_sources)} for one rich HLO."""
    lines = Path(path).read_text().split('\n')
    files, funcs, locs, frames = {}, {}, {}, {}
    sec = None
    comps = defaultdict(list)
    cur = None
    for line in lines:
        if line in ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'):
            sec = line
            continue
        if sec:
            if re.match(r'^\d+ ', line):
                if sec == 'FileNames':
                    i, v = re.match(r'^(\d+) "(.*)"', line).groups()
                    files[i] = v
                elif sec == 'FunctionNames':
                    i, v = re.match(r'^(\d+) "(.*)"', line).groups()
                    funcs[i] = v
                elif sec == 'FileLocations':
                    m = re.match(r'^(\d+) \{file_name_id=(\d+) function_name_id=(\d+) line=(\d+)', line)
                    locs[m.group(1)] = (files[m.group(2)], funcs[m.group(3)], int(m.group(4)))
                else:
                    m = re.match(r'^(\d+) \{file_location_id=(\d+)(?: parent_frame_id=(\d+))?', line)
                    frames[m.group(1)] = (m.group(2), m.group(3))
                continue
            if not line.strip():
                continue
            sec = None
        m = re.match(r'^(?:ENTRY )?%?([\w.\-]+) .*\{$', line)
        if m and not line.startswith(' '):
            cur = m.group(1)
            continue
        if line.startswith(' ') and cur:
            comps[cur].append(line.strip())

    def chain(fid):
        out, seen = [], set()
        while fid and fid in frames and fid not in seen:
            seen.add(fid)
            loc, parent = frames[fid]
            if loc in locs:
                out.append(locs[loc])
            fid = parent
        return out

    def instr_family(line):
        fam = None
        sf = re.search(r'stack_frame_id=(\d+)', line)
        stack = chain(sf.group(1)) if sf else []
        for file, func, lineno in stack:  # innermost first
            fam = classify(file) or classify(func)
            if fam and fam not in ('coupling', 'runtime'):
                return fam, f'{Path(file).name}:{func}'
        op = re.search(r'op_name="([^"]*)"', line)
        if op:
            for scope in reversed(re.findall(r'jit\(([^)]*)\)', op.group(1))):
                fam2 = classify(scope)
                if fam2:
                    return fam2, scope
        if stack:
            file, func, _ = stack[0]
            return (classify(file) or 'other'), f'{Path(file).name}:{func}'
        return 'unknown', None

    table = {}
    for comp, body in comps.items():
        for line in body:
            m = INSTR.match(line)
            if not m:
                continue
            name, op, rest = m.groups()
            members = []
            called = re.search(r'calls=%?([\w.\-]+)', rest)
            if op == 'fusion' and called and called.group(1) in comps:
                members = [l for l in comps[called.group(1)] if 'parameter(' not in l]
            if not members:
                members = [line]
            fams, srcs = Counter(), Counter()
            for member in members:
                fam, src = instr_family(member)
                if fam == 'unknown':
                    continue
                fams[fam] += 1
                if src:
                    srcs[(fam, src)] += 1
            if not fams:
                fams['unknown'] += 1
            top, n = fams.most_common(1)[0]
            share = n / sum(fams.values())
            kernel = name.replace('.', '_')
            entry = dict(family=top if share >= 0.7 else 'MIXED:' + '+'.join(f for f, _ in fams.most_common(3)),
                         share=round(share, 3), n=sum(fams.values()), op=op,
                         sources=[f'{f}|{s}|{c}' for (f, s), c in srcs.most_common(3)])
            target = re.search(r'custom_call_target="([^"]*)"', rest)
            if target:
                entry['target'] = target.group(1)
            table[kernel] = entry
            # Pallas/Triton kernels are launched under their Python kernel name.
            opname = re.search(r'op_name="([^"]*)/pallas_call"', rest)
            if op == 'custom-call' and opname:
                base = re.sub(r'\W', '_', opname.group(1).split('/')[-1])
                table.setdefault(base, entry)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', required=True)
    ap.add_argument('--root', action='append', required=True)
    ap.add_argument('--module', action='append', required=True, help='PROGRAM_ID=LABEL=RICH_HLO')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    modules = {}
    for item in args.module:
        pid, label, path = item.split('=', 2)
        modules[pid] = (label, load_module(path))
    conn = sqlite3.connect(f'file:{Path(args.db).resolve()}?mode=ro', uri=True)
    strings = dict(conn.execute('select id,value from StringIds'))
    nvtx = [(s, e, t if t is not None else strings.get(tid, ''))
            for s, e, t, tid in conn.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    mod_ranges = sorted((s, e, re.search(r'program_id=(\d+)', t).group(1)) for s, e, t in nvtx
                        if t.startswith('XlaModule:') and 'program_id=' in t)
    thunks = sorted((s, e, t) for s, e, t in nvtx if t.startswith('Thunk:') or t.startswith('XlaModule:'))
    thunk_starts = [r[0] for r in thunks]
    mod_starts = [r[0] for r in mod_ranges]
    api = {corr: s for s, corr in conn.execute('select start,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}

    def module_of(t):
        i = bisect.bisect_right(mod_starts, t) - 1
        while i >= 0:
            s, e, pid = mod_ranges[i]
            if s <= t <= e:
                return pid
            if e < t and i < len(mod_ranges) - 1:
                break
            i -= 1
        return None

    def thunk_of(t):
        i = bisect.bisect_right(thunk_starts, t) - 1
        best = None
        while i >= 0 and i > len(thunks) - 10**9:
            s, e, txt = thunks[i]
            if s <= t <= e:
                best = txt
                if txt.startswith('Thunk:'):
                    return txt
            if t - s > 5e9:
                break
            i -= 1
        return best

    result = {}
    for root in args.root:
        rs, re_ = next((s, e) for s, e, t in nvtx if t == root)
        fam = defaultdict(Counter)
        kern = defaultdict(Counter)
        unmapped = Counter()
        rows = conn.execute('select start,end,correlationId,shortName from CUPTI_ACTIVITY_KIND_KERNEL '
                            'where start>=? and start<=?', (rs, re_ + int(5e8)))
        for s, e, corr, sid in rows:
            t = api.get(corr)
            if t is None or not (rs <= t <= re_):
                continue
            name = strings.get(sid, '')
            pid = module_of(t)
            if pid in modules:
                label, table = modules[pid]
                info = table.get(name) or table.get(re.sub(r'_+\d+$', '', name))
                family = info['family'] if info else 'UNMAPPED'
                if not info:
                    unmapped[f'{label}:{name}'] += e - s
            else:
                label = 'other_exec'
                txt = thunk_of(t) or ''
                family = classify(txt) or 'other_exec'
            fam[family]['ns'] += e - s
            fam[family]['n'] += 1
            fam[family][f'{label}_ns'] += e - s
            kern[(label, family, name)]['ns'] += e - s
            kern[(label, family, name)]['n'] += 1
        for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
            try:
                rows = list(conn.execute(f'select start,end,correlationId from {table} where start>=? and start<=?',
                                         (rs, re_ + int(5e8))))
            except sqlite3.OperationalError:
                continue
            for s, e, corr in rows:
                t = api.get(corr)
                if t is not None and rs <= t <= re_:
                    fam[kind]['ns'] += e - s
                    fam[kind]['n'] += 1
        total = sum(v['ns'] for v in fam.values())
        result[root] = dict(
            total_ms=total / 1e6,
            families={k: dict(ms=v['ns'] / 1e6, n=v['n'], share=round(v['ns'] / total, 4),
                              **{kk: vv / 1e6 for kk, vv in v.items() if kk.endswith('_ns')})
                      for k, v in sorted(fam.items(), key=lambda x: -x[1]['ns'])},
            top_kernels=[dict(module=l, family=f, kernel=n, ms=v['ns'] / 1e6, n=v['n'])
                         for (l, f, n), v in sorted(kern.items(), key=lambda x: -x[1]['ns'])[:60]],
            unmapped_ms={k: v / 1e6 for k, v in unmapped.most_common(20)})
    for label, table in modules.values():
        pass
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for root, r in result.items():
        print(root, f"{r['total_ms']:.2f} ms")
        for k, v in r['families'].items():
            print(f"  {k:28s} {v['ms']:9.3f} ms  {v['share']*100:5.1f} %  n={v['n']}")


if __name__ == '__main__':
    main()
