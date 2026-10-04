#!/usr/bin/env bash
# ADR-031 S0+S1 Tier-S bit-identity gate -- DETERMINISTIC harness.
#
# The default (non-deterministic) cross-process AB-hash forecast on this 5090 is
# NOT bit-reproducible run-to-run: an identical-code A/B (streamA_vs_streamA)
# already shows ~99/168 leaf mismatches, the same signature as a base-vs-s0s1
# diff. That noise floor comes from non-deterministic GPU atomics (scatter /
# index-add in advection + physics). To make bit-identity meaningful we pin
# XLA_FLAGS=--xla_gpu_deterministic_ops=true and FIRST prove identical code -> 0
# mismatches (the determinism baseline), THEN diff base vs S0+S1.
set -euo pipefail

ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
BASE_REF="499d3724"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"
mkdir -p "$OUT"
cd "$ROOT"

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
echo "[abdet] XLA_FLAGS=$XLA_FLAGS" >&2

BASE_SRC="$(mktemp -d /tmp/v016_base_src.XXXXXX)"
echo "[abdet] materializing base ($BASE_REF) src -> $BASE_SRC" >&2
git archive "$BASE_REF" src | tar -x -C "$BASE_SRC"
for entry in "$ROOT"/* "$ROOT"/.[!.]*; do
  name="$(basename "$entry")"
  case "$name" in src|.git|.git*) continue;; esac
  [ -e "$entry" ] && ln -sfn "$entry" "$BASE_SRC/$name"
done

run_probe () {  # tag, srcdir
  local tag="$1" srcdir="$2"
  echo "[abdet] === run tag=$tag (src=$srcdir) ===" >&2
  PYTHONPATH="$srcdir" python "$PROBE" --tag "$tag" --hours 3
  cp -f "proofs/perf/v015/ab_${tag}.json" "$OUT/ab_${tag}.json"
}

compare () {  # a.json b.json label outfile
  PYTHONPATH="$ROOT/src" python - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
a=json.load(open(sys.argv[1]))["hashes"]; b=json.load(open(sys.argv[2]))["hashes"]
label=sys.argv[3]
mm=[]; total=0
for h in sorted(set(a)|set(b)):
    al=a.get(h,{}); bl=b.get(h,{})
    for leaf in sorted(set(al)|set(bl)):
        total+=1
        if al.get(leaf)!=bl.get(leaf): mm.append({"hour":h,"leaf":leaf,"a":al.get(leaf),"b":bl.get(leaf)})
v={"schema":"ADR031_S0S1_TierS_BitIdentity_deterministic","comparison":label,
   "leaf_hash_comparisons":total,"mismatches":len(mm),"pass":len(mm)==0,
   "xla_deterministic":True,"mismatch_detail":mm[:60]}
json.dump(v, open(sys.argv[4],"w"), indent=2)
print(f"[{label}] {len(mm)}/{total} mismatches -> {'PASS' if not mm else 'FAIL'}")
PY
}

# (1) determinism baseline: identical BASE code, two runs.
run_probe base_det      "$BASE_SRC/src"
run_probe base_det_rep  "$BASE_SRC/src"
compare "$OUT/ab_base_det.json" "$OUT/ab_base_det_rep.json" \
        "determinism-baseline(base-vs-base)" "$OUT/det_baseline_verdict.json"

# (2) the gate: BASE vs S0+S1 HEAD, same deterministic flag.
run_probe s0s1_det "$ROOT/src"
compare "$OUT/ab_base_det.json" "$OUT/ab_s0s1_det.json" \
        "S0S1-vs-base" "$OUT/s0s1_det_identity_verdict.json"

rm -rf "$BASE_SRC"
echo "[abdet] done -- baseline: $OUT/det_baseline_verdict.json ; gate: $OUT/s0s1_det_identity_verdict.json" >&2
