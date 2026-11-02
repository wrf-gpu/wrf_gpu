"""Source (file:function:line) census of one fusion's body in an optimized HLO text dump."""
import re, sys, collections
text = open(sys.argv[1]).read(); want = sys.argv[2]; lines = text.split("\n")
heads = {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}
def table(name):
    s = lines.index(name) + 1; out = {}
    for l in lines[s:]:
        if l in heads or not l.strip(): break
        k, v = l.split(" ", 1); out[int(k)] = v
    return out
files = {k: v.strip('"') for k, v in table("FileNames").items()}
funcs = {k: v.strip('"') for k, v in table("FunctionNames").items()}
locs = {k: dict(re.findall(r"(\w+)=(\d+)", v)) for k, v in table("FileLocations").items()}
frames = {k: dict(re.findall(r"(\w+)=(\d+)", v)) for k, v in table("StackFrames").items()}
def chain(fid):
    out, seen = [], set()
    while fid and fid not in seen:
        seen.add(fid); fr = frames[fid]; loc = locs[int(fr["file_location_id"])]
        f = files[int(loc["file_name_id"])]
        if "gpuwrf" in f: out.append(f"{f.split('src/gpuwrf/')[-1]}:{funcs[int(loc['function_name_id'])]}:{loc.get('line')}")
        fid = int(fr["parent_frame_id"])
    return out
call = next(l for l in lines if re.search(rf"%{re.escape(want)} = .* fusion\(", l))
comp = re.search(r"calls=%?([\w.\-]+)", call).group(1)
start = next(i for i, l in enumerate(lines) if re.match(rf"^%?{re.escape(comp)} \(", l))
end = next(i for i in range(start, len(lines)) if lines[i].startswith("}"))
inner, outer = collections.Counter(), collections.Counter()
for b in lines[start + 1:end]:
    m = re.search(r"stack_frame_id=(\d+)", b)
    if m:
        c = chain(int(m.group(1)))
        if c: inner[c[0]] += 1; outer[" <- ".join(c[:4])] += 1
print("instructions:", end - start - 1)
for k, v in inner.most_common(25): print(v, k)
print("--- chains")
for k, v in outer.most_common(8): print(v, k)
