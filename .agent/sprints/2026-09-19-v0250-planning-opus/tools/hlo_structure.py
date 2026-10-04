"""CPU-only structural walk of an XLA optimized HLO dump (planning evidence tool).

Parses computations, resolves stack_frame_id -> deepest gpuwrf file:line, and prints the
ordered structure of a computation: runs of fusions (command-buffer candidates), whiles
(with known_trip_count), conditionals, custom-calls and copies, recursively.
Usage: python3 hlo_structure.py <hlo.txt> [--root COMP] [--depth N]
"""
from __future__ import annotations
import re, sys, collections

def parse(path):
    comps = collections.OrderedDict(); cur = None; entry = None
    files, funcs, locs, frames = {}, {}, {}, {}
    sec = None
    for ln in open(path):
        s = ln.rstrip("\n")
        if s in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            sec = s; continue
        if sec:
            m = re.match(r"^(\d+) (.*)$", s)
            if m:
                i = int(m.group(1)); body = m.group(2)
                if sec == "FileNames": files[i] = body.strip().strip('"')
                elif sec == "FunctionNames": funcs[i] = body.strip().strip('"')
                elif sec == "FileLocations": locs[i] = dict(re.findall(r"(\w+)=(-?\d+)", body))
                elif sec == "StackFrames": frames[i] = dict(re.findall(r"(\w+)=(-?\d+)", body))
                continue
            elif s.strip():
                sec = None
        m = re.match(r"^(ENTRY )?%?([\w.\-]+)(?: \(.*\))?[^{]*\{\s*$", s)
        if m and not s.startswith("  "):
            cur = m.group(2); comps[cur] = []
            if m.group(1): entry = cur
            continue
        if s == "}":
            cur = None; continue
        if cur is not None and s.startswith("  "):
            comps[cur].append(s.strip())
    return comps, entry, (files, funcs, locs, frames)

def frame_str(tabs, fid, want="gpuwrf"):
    files, funcs, locs, frames = tabs
    cur = fid; out = []
    d = 0
    while cur in frames and d < 60:
        fr = frames[cur]; loc = locs[int(fr["file_location_id"])]
        fn = files.get(int(loc["file_name_id"]), "?")
        out.append((fn, loc.get("line"), funcs.get(int(loc["function_name_id"]), "?")))
        cur = int(fr.get("parent_frame_id", 0)); d += 1
    # out[0] = innermost
    for fn, line, f in out:
        if want in fn:
            return f"{fn.split('src/')[-1]}:{line}[{f}]"
    return f"{out[0][0].split('/')[-1]}:{out[0][1]}" if out else "?"

INS = re.compile(r"^(ROOT )?%([\w.\-]+) = (.+?) (\w[\w\-]*)\((.*)$")

def classify(line):
    m = INS.match(line)
    if not m: return None
    name, typ, op, rest = m.group(2), m.group(3), m.group(4), m.group(5)
    info = {"name": name, "op": op, "type": typ, "line": line}
    for key in ("condition", "body", "calls", "to_apply"):
        mm = re.search(key + r"=%([\w.\-]+)", line)
        if mm: info[key] = mm.group(1)
    mm = re.search(r"branch_computations=\{([^}]*)\}", line)
    if mm: info["branches"] = [x.strip().lstrip("%") for x in mm.group(1).split(",")]
    mm = re.search(r"true_computation=%([\w.\-]+), false_computation=%([\w.\-]+)", line)
    if mm: info["branches"] = [mm.group(1), mm.group(2)]
    mm = re.search(r'known_trip_count":\{"n":"(\d+)"', line)
    if mm: info["trip"] = int(mm.group(1))
    mm = re.search(r"stack_frame_id=(\d+)", line)
    if mm: info["frame"] = int(mm.group(1))
    mm = re.search(r'op_name="([^"]*)"', line)
    if mm: info["op_name"] = mm.group(1)
    mm = re.search(r'custom_call_target="([^"]*)"', line)
    if mm: info["target"] = mm.group(1)
    return info

TRIVIAL = {"parameter", "get-tuple-element", "tuple", "constant", "bitcast", "after-all", "partition-id", "replica-id"}

def walk(comps, tabs, comp, depth, maxdepth, indent=0, mult=1, out=None):
    run = []  # consecutive kernel-like ops
    def flush():
        if run:
            srcs = collections.Counter(r[1] for r in run)
            top = "; ".join(f"{k} x{v}" for k, v in srcs.most_common(3))
            print("  " * indent + f"[{len(run)} kernels] {top}")
            run.clear()
    for ln in comps.get(comp, []):
        i = classify(ln)
        if not i or i["op"] in TRIVIAL: continue
        src = frame_str(tabs, i["frame"]) if "frame" in i else "?"
        if i["op"] == "while":
            flush()
            print("  " * indent + f"WHILE %{i['name']} trip={i.get('trip','DYNAMIC')} src={src}")
            if depth < maxdepth:
                walk(comps, tabs, i["body"], depth + 1, maxdepth, indent + 1)
        elif i["op"] == "conditional":
            flush()
            print("  " * indent + f"COND %{i['name']} src={src} branches={i.get('branches')}")
            if depth < maxdepth:
                for b in i.get("branches", []):
                    print("  " * (indent + 1) + f"branch {b}:")
                    walk(comps, tabs, b, depth + 1, maxdepth, indent + 2)
        elif i["op"] == "custom-call":
            flush()
            print("  " * indent + f"CUSTOM {i.get('target')} src={src}")
        elif i["op"] in ("fusion", "copy", "copy-start", "copy-done", "dynamic-update-slice", "dynamic-slice", "reduce", "select", "add", "sort", "scatter", "gather", "concatenate", "transpose", "slice", "pad", "convert", "broadcast", "iota", "compare", "multiply", "subtract", "divide", "maximum", "minimum", "and", "or", "not", "exponential", "log", "sqrt", "rsqrt", "power", "abs", "negate", "clamp", "reverse", "reshape"):
            tag = i["op"] if i["op"] != "fusion" else "f"
            run.append((tag, src))
        elif i["op"] == "call":
            flush()
            print("  " * indent + f"CALL %{i.get('to_apply') or i.get('calls')} src={src}")
            if depth < maxdepth and (i.get("to_apply") or i.get("calls")):
                walk(comps, tabs, i.get("to_apply") or i.get("calls"), depth + 1, maxdepth, indent + 1)
        else:
            run.append((i["op"], src))
    flush()

if __name__ == "__main__":
    path = sys.argv[1]
    root = None; maxd = 3
    if "--root" in sys.argv: root = sys.argv[sys.argv.index("--root") + 1]
    if "--depth" in sys.argv: maxd = int(sys.argv[sys.argv.index("--depth") + 1])
    comps, entry, tabs = parse(path)
    print(f"computations={len(comps)} entry={entry}")
    walk(comps, tabs, root or entry, 0, maxd)
