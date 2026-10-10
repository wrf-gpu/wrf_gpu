#!/usr/bin/env python3
"""List module-level variable names declared in module_mp_nssl_2mom.F (spec part).

Used only to generate the print-only constants dump of the instrumented oracle copy.
"""
import re, sys

def split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([': depth += 1
        elif ch in ')]': depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    return out

def main(path, end_line, start_line=1):
    lines = open(path).read().split('\n')[start_line - 1:end_line]
    # join continuation lines
    stmts, cur, skip = [], '', 0
    for raw in lines:
        l = raw.split('!')[0].rstrip()
        if l.lstrip().startswith('#'):
            continue
        if cur:
            cur += ' ' + l.strip().lstrip('&')
        else:
            cur = l
        if cur.rstrip().endswith('&'):
            cur = cur.rstrip()[:-1]
            continue
        stmts.append(cur); cur = ''
    names = []
    for s in stmts:
        t = s.strip()
        m = re.match(r'^(real|integer|logical|double\s+precision)\b(.*?)::(.*)$', t, re.I)
        if not m:
            m2 = re.match(r'^(real|integer|logical|double\s+precision)\s+([A-Za-z_].*)$', t, re.I)
            if not m2 or re.match(r'^(function|subroutine)\b', m2.group(2), re.I):
                continue
            class _M:
                def __init__(s_, a, b): s_.g = (None, a, '', b)
                def group(s_, i): return s_.g[i]
            m = _M(m2.group(1), m2.group(2))
        attrs = m.group(2).lower()
        if 'allocatable' in attrs or 'pointer' in attrs:
            kind = 'alloc'
        else:
            kind = 'plain'
        typ = m.group(1).lower().replace(' ', '')
        for item in split_top(m.group(3)):
            nm = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\()?', item)
            if nm:
                isarr = bool(nm.group(2)) or 'dimension' in attrs
                dims = ''
                if nm.group(2):
                    rest = item[nm.end() - 1:]
                    depth = 0
                    for j, ch in enumerate(rest):
                        depth += ch == '('
                        depth -= ch == ')'
                        if depth == 0:
                            dims = rest[1:j]
                            break
                elif 'dimension' in attrs:
                    dm = re.search(r'dimension\s*\((.*)\)', m.group(2), re.I)
                    dims = dm.group(1) if dm else ''
                names.append((nm.group(1), typ, 'array' if isarr else 'scalar', 'parameter' in attrs, dims.replace(' ', '')))
    seen = set()
    for n, typ, kind, par, dims in names:
        if n.lower() in seen: continue
        seen.add(n.lower())
        if DIMS:
            print(n, typ, kind, int(par), dims or '-')
        else:
            print(n, typ, kind, int(par))

DIMS = '--dims' in sys.argv
if DIMS:
    sys.argv.remove('--dims')
if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
