"""Attribute every f64-typed instruction of an optimized HLO dump to source (stack frames)."""
import json, re, sys
from collections import Counter, defaultdict
path = sys.argv[1]
lines = open(path).read().split('\n')
sec = None; fn = {}; fun = {}; locs = {}; frames = {}
comps = defaultdict(list); cur = None
for l in lines:
    if l in ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'):
        sec = l; continue
    if sec:
        if re.match(r'^\d+ ', l):
            if sec == 'FileNames':
                i, v = re.match(r'^(\d+) "(.*)"', l).groups(); fn[i] = v.split('/')[-1]
            elif sec == 'FunctionNames':
                i, v = re.match(r'^(\d+) "(.*)"', l).groups(); fun[i] = v
            elif sec == 'FileLocations':
                m = re.match(r'^(\d+) \{file_name_id=(\d+) function_name_id=(\d+) line=(\d+)', l)
                locs[m.group(1)] = f'{fn[m.group(2)]}:{fun[m.group(3)]}:{m.group(4)}'
            else:
                m = re.match(r'^(\d+) \{file_location_id=(\d+)(?: parent_frame_id=(\d+))?', l)
                frames[m.group(1)] = (m.group(2), m.group(3))
            continue
        if l.strip() == '':
            continue
        sec = None
    m = re.match(r'^(?:ENTRY )?%?([\w.\-]+) .*\{$', l)
    if m and not l.startswith(' '):
        cur = m.group(1); continue
    if l.startswith('  ') and cur:
        comps[cur].append(l.strip())

def top(fid):
    out = []; seen = set()
    while fid and fid in frames and fid not in seen:
        seen.add(fid); loc, p = frames[fid]; out.append(locs.get(loc)); fid = p
    return out[0] if out else None

ARITH = {'add','subtract','multiply','divide','maximum','minimum','select','compare','power','exponential','log','sqrt','rsqrt','negate','abs','floor','ceil','sine','cosine','tanh','remainder','clamp','atan2','reduce','dot','exponential-minus-one','log-plus-one','round-nearest-even','round-nearest-afz','sign'}
INSTR = re.compile(r'^(?:ROOT )?%([\w.\-]+) = (\S+?)(?:\{[^}]*\})? ([\w\-]+)\((.*)')
arith = Counter(); by_src = Counter(); fusions = {}
for comp, body in comps.items():
    types = {}
    for ins in body:
        m = INSTR.match(ins)
        if m: types[m.group(1)] = m.group(2)
    for ins in body:
        m = INSTR.match(ins)
        if not m: continue
        name, typ, op, rest = m.groups()
        f64_res = typ.startswith('f64')
        opnds = re.findall(r'%([\w.\-]+)', rest.split(')')[0])
        f64_opnd = op == 'compare' and any(types.get(o, '').startswith('f64') for o in opnds)
        sf = re.search(r'stack_frame_id=(\d+)', ins)
        src = top(sf.group(1)) if sf else None
        if op in ARITH and (f64_res or f64_opnd):
            arith[op] += 1; by_src[(src, op)] += 1
for (src, op), n in sorted(by_src.items(), key=lambda x: str(x)):
    print(n, op, src)
print('total', dict(arith))
