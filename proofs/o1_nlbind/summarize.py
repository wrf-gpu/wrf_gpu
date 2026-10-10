"""Summarise the o1-nlbind probe JSONs into the committed CLI support matrix.

usage: python summarize.py <lane_dir> <out.json> <out.md>
Evidence eras (each compared with the release program of ITS base tree):
  * pre-rebase matrix (probe_cand1.json, matrix_*.json): candidate tree on main 643416607, base probe_base.json;
  * final (final_cand_swiss.json / final_cand_prod.json): rebased candidate on main dee37ef2a, base
    final_base_swiss.json / final_base_prod.json.
For each arm: requested edits, whether every requested value reached the bound OperationalNamelist, whether the
traced production-step program differs from the release program, and the trace/load error (if any).  The last
row per arm (final era wins) is the verdict.
"""
import glob
import json
import sys
from pathlib import Path

LANE, OUT_JSON, OUT_MD = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def load(name):
    path = LANE / name
    return json.loads(path.read_text()) if path.exists() else None


def release_sha(data):
    return (data or {}).get("arms", {}).get("release", {}).get("sha")


ERAS = [  # (candidate files, base json, era label)
    (["probe_cand1.json"] + sorted(Path(p).name for p in glob.glob(str(LANE / "matrix_*.json"))),
     "probe_base.json", "pre-rebase (main 643416607)"),
    (["final_cand_swiss_v1_islandbug.json"], "final_base_swiss.json",
     "final v1 (candidate c82b9b75b; island arms superseded)"),
    (["final_cand_swiss.json"], "final_base_swiss.json", "final (main dee37ef2a)"),
    (["final_cand_prod.json"], "final_base_prod.json", "final PROD d01 (main dee37ef2a)"),
    (["head_cand_swiss.json"], "final_base_swiss.json", "HEAD e90c1d4f4 (main dee37ef2a)"),
    (["head_cand_prod.json"], "final_base_prod.json", "HEAD e90c1d4f4 PROD d01 (main dee37ef2a)"),
    (["r2_cand_swiss.json"], "r2_base_swiss.json", "rebased 1188aba98 (main b5100f705)"),
    (["r2_cand_prod.json"], "r2_base_prod.json", "rebased 1188aba98 PROD d01 (main b5100f705)"),
    (["r3_cand_swiss.json"], "r3_base_swiss.json", "rebased 2f9004938 (main 14253f8ca, +o1-camrad)"),
    (["r3_cand_prod.json"], "r3_base_prod.json", "rebased 2f9004938 PROD d01 (main 14253f8ca)"),
]


def bound_ok(rec):
    b = rec.get("bound")
    if not b:
        return None
    for key, val in rec["edits"].items():
        got = b.get(key)
        if got is None:
            continue  # not a recorded field
        try:
            if float(got) != float(val):
                return False
        except (TypeError, ValueError):
            if str(got) != str(val):
                return False
    return True


rows, identity = [], {}
for files, base_name, era in ERAS:
    base = load(base_name)
    ref = release_sha(base)
    for name in files:
        data = load(name)
        if data is None:
            continue
        for arm, rec in data["arms"].items():
            err = rec.get("error")
            row = {"arm": arm, "era": era, "source": name, "case": data.get("case"),
                   "bound_ok": bound_ok(rec),
                   "program_differs_from_release": (None if rec.get("sha") is None or ref is None
                                                    else rec["sha"] != ref),
                   "sha": rec.get("sha"), "error": err.splitlines()[0][:300] if err else None}
            for key in ("load_s", "trace_s", "wall_s"):
                if key in rec:
                    row[key] = rec[key]
            if arm == "release":
                identity[f"{era} :: {Path(data.get('case', '')).name or 'case'}"] = {
                    "base_src": (base or {}).get("src"), "base_sha": ref,
                    "candidate_src": data.get("src"), "candidate_sha": rec.get("sha"),
                    "identical": rec.get("sha") == ref}
            rows.append(row)

verdict = {}
for r in rows:  # later eras override earlier rows of the same arm on the Swiss case
    if "PROD" in r["era"]:
        continue
    verdict[r["arm"]] = r
OUT_JSON.write_text(json.dumps({"release_identity": identity, "verdict_per_arm": verdict, "rows": rows},
                               indent=1))

lines = ["| namelist edit (Swiss RD11 d01, release defaults) | bound | program changes | result | evidence |",
         "|---|---|---|---|---|"]
for arm, r in verdict.items():
    if arm == "release":
        continue
    res = "traces" if not r["error"] else f"refused/crash: {r['error'][:150]}"
    lines.append(f"| `{arm.replace(',', ', ')}` | {r['bound_ok']} | {r['program_differs_from_release']} | "
                 f"{res} | {r['era']} |")
lines.append("")
lines.append("| release identity | base sha | candidate sha | identical |")
lines.append("|---|---|---|---|")
for k, v in identity.items():
    lines.append(f"| {k} | `{(v['base_sha'] or '')[:12]}` | `{(v['candidate_sha'] or '')[:12]}` | {v['identical']} |")
OUT_MD.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
