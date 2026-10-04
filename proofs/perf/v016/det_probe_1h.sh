#!/usr/bin/env bash
# Quick determinism validator: run HEAD code twice for 1 production hour with
# XLA deterministic ops, compare leaf hashes. 0 mismatches => the flag makes the
# cross-process forecast bit-reproducible (so a base-vs-s0s1 diff is meaningful).
# Also run ONE non-deterministic repeat as the contrast control.
set -euo pipefail
ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"; mkdir -p "$OUT"; cd "$ROOT"

cmp () { PYTHONPATH="$ROOT/src" python - "$1" "$2" "$3" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["hashes"]; b=json.load(open(sys.argv[2]))["hashes"]; lab=sys.argv[3]
mm=0;tot=0
for h in sorted(set(a)|set(b)):
    al=a.get(h,{});bl=b.get(h,{})
    for k in sorted(set(al)|set(bl)):
        tot+=1; mm+= (al.get(k)!=bl.get(k))
print(f"[{lab}] {mm}/{tot} mismatches")
PY
}

echo "[det1h] CONTROL: non-deterministic, HEAD twice, 1h" >&2
PYTHONPATH="$ROOT/src" python "$PROBE" --tag nd_a --hours 1; cp -f proofs/perf/v015/ab_nd_a.json "$OUT/"
PYTHONPATH="$ROOT/src" python "$PROBE" --tag nd_b --hours 1; cp -f proofs/perf/v015/ab_nd_b.json "$OUT/"
cmp "$OUT/ab_nd_a.json" "$OUT/ab_nd_b.json" "nondeterministic-control(HEAD-vs-HEAD)"

echo "[det1h] TEST: deterministic, HEAD twice, 1h" >&2
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
PYTHONPATH="$ROOT/src" python "$PROBE" --tag det_a --hours 1; cp -f proofs/perf/v015/ab_det_a.json "$OUT/"
PYTHONPATH="$ROOT/src" python "$PROBE" --tag det_b --hours 1; cp -f proofs/perf/v015/ab_det_b.json "$OUT/"
cmp "$OUT/ab_det_a.json" "$OUT/ab_det_b.json" "deterministic-test(HEAD-vs-HEAD)"
echo "[det1h] done" >&2
