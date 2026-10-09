"""Julian ON/OFF lowered-HLO verdict (manager 23:02:55Z gate: delta only the traced julian arithmetic, <= ~20 scalar
ops, no new loops/transposes; OFF identical).
OFF: raw lowered HLO sha256, main vs branch (same export path). ON vs branch OFF: multiset of (opcode, result shape)
over every instruction (tuple shapes incl. /*index=N*/ comments, E124) + direct counts of while/conditional/
transpose/custom-call/call. (The receipts' 'ops' counter regex '=\\s*\\S+\\s+op(' skips tuple-shaped results, e.g.
while; it is kept for reference only.)  usage: score_hlo_delta.py [case ...]"""
import gzip, json, re, sys
from collections import Counter
from pathlib import Path
J = Path("<USER_HOME>/wrf_gpu2_lanes/b-thompson/JUL/hlo")
INSTR = re.compile(r"^\s*(?:ROOT\s+)?[\w.\-]+\s*=\s*(\(.*?\)|\w+\[[^\]]*\](?:\{[^}]*\})?)\s+([\w\-]+)\(")
STRUCT = ("while", "conditional", "transpose", "custom-call", "call", "fusion", "scatter", "gather", "sort")


def instrs(text):
    out = Counter()
    for ln in text.splitlines():
        m = INSTR.match(re.sub(r"/\*index=\d+\*/", "", ln))
        if m:
            out[(m.group(2), re.sub(r"\{[^}]*\}", "", m.group(1)))] += 1
    return out


def rank(shape):
    if shape.startswith("("):
        return 99
    dims = shape[shape.index("[") + 1:shape.index("]")]
    return 0 if not dims else sum(int(d) > 1 for d in dims.split(","))


out = {}
for case in (sys.argv[1:] or ["prod", "wn3"]):
    m, b = (json.load(open(J / f"{arm}_{case}/receipt.json")) for arm in ("main_off", "br_off"))
    for prog in b["domains"]:
        rd = lambda arm: gzip.decompress((J / f"{arm}_{case}" / (prog.replace("/", "_") + ".hlo.gz")).read_bytes()).decode()  # noqa: E731
        t_off, t_on = rd("br_off"), rd("br_on")
        off_same = m["domains"][prog]["raw_lower_hlo_sha256"] == b["domains"][prog]["raw_lower_hlo_sha256"]
        i_off, i_on = instrs(t_off), instrs(t_on)
        added, removed = i_on - i_off, i_off - i_on
        struct = {op: [sum(v for (o, _s), v in i_off.items() if o == op), sum(v for (o, _s), v in i_on.items() if o == op)]
                  for op in STRUCT}
        nonscalar = {f"{o} {s}": v for (o, s), v in added.items() if rank(s) > 0}
        out[f"{case}/{prog}"] = dict(
            off_raw_hlo_identical_main_vs_branch=off_same, off_sha256=b["domains"][prog]["raw_lower_hlo_sha256"],
            on_sha256=json.load(open(J / f"br_on_{case}/receipt.json"))["domains"][prog]["raw_lower_hlo_sha256"],
            instructions_off_on=[sum(i_off.values()), sum(i_on.values())],
            added=sum(added.values()), removed=sum(removed.values()),
            added_by_op_shape={f"{o} {s}": v for (o, s), v in sorted(added.items())},
            removed_by_op_shape={f"{o} {s}": v for (o, s), v in sorted(removed.items())},
            added_not_scalar=nonscalar, structural_off_on=struct,
            structural_changed={k: v for k, v in struct.items() if v[0] != v[1]})
        r = out[f"{case}/{prog}"]
        print(case, prog, "OFF identical:", off_same, "| instr", r["instructions_off_on"], "added", r["added"],
              "removed", r["removed"], "| not scalar:", nonscalar, "| structural changed:", r["structural_changed"])
name = "verdict.json" if not sys.argv[1:] else f"verdict_{'_'.join(sys.argv[1:])}.json"
(J / name).write_text(json.dumps(out, indent=1) + "\n")
