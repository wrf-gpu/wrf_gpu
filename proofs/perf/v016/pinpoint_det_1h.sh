#!/usr/bin/env bash
# DETERMINISTIC pinpoint: localize the base-vs-s0s1 102/180 divergence to S0
# (carry leaf) or S1 (seam). XLA_FLAGS=--xla_gpu_deterministic_ops=true makes the
# forecast bit-reproducible (proven: base-vs-base 0/180), so any nonzero here is
# a REAL graph/numerics difference. 1 production hour per run.
#   base = 499d3724, s0 = 12ed00be (carry leaf only), s0s1 = worktree HEAD
set -euo pipefail
ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"; mkdir -p "$OUT"; cd "$ROOT"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"

mktree () { local d; d="$(mktemp -d /tmp/v016_tree.XXXXXX)"; git archive "$1" src | tar -x -C "$d"
  for e in "$ROOT"/* "$ROOT"/.[!.]*; do local n; n="$(basename "$e")"; case "$n" in src|.git|.git*) continue;; esac; [ -e "$e" ] && ln -sfn "$e" "$d/$n"; done; echo "$d"; }
BASE_T="$(mktree 499d3724)"; S0_T="$(mktree 12ed00be)"
run () { PYTHONPATH="$2" python "$PROBE" --tag "$1" --hours 1; cp -f "proofs/perf/v015/ab_$1.json" "$OUT/"; }
cmp () { PYTHONPATH="$ROOT/src" python - "$1" "$2" "$3" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["hashes"]; b=json.load(open(sys.argv[2]))["hashes"]; lab=sys.argv[3]
mm=[];tot=0
for h in sorted(set(a)|set(b)):
    al=a.get(h,{});bl=b.get(h,{})
    for k in sorted(set(al)|set(bl)):
        tot+=1
        if al.get(k)!=bl.get(k): mm.append(f"{h}:{k}")
print(f"[{lab}] {len(mm)}/{tot} mismatches"+("" if not mm else f"  {mm[:10]}"))
PY
}
run base_dpp "$BASE_T/src"; run s0_dpp "$S0_T/src"; run s0s1_dpp "$ROOT/src"
echo "========= DETERMINISTIC PINPOINT (1h) ========="
cmp "$OUT/ab_base_dpp.json" "$OUT/ab_s0_dpp.json"   "S0 carry-leaf  base-vs-s0"
cmp "$OUT/ab_s0_dpp.json"   "$OUT/ab_s0s1_dpp.json" "S1 seam only   s0-vs-s0s1"
cmp "$OUT/ab_base_dpp.json" "$OUT/ab_s0s1_dpp.json" "S0+S1 full     base-vs-s0s1"
rm -rf "$BASE_T" "$S0_T"; echo "========= done ========="
