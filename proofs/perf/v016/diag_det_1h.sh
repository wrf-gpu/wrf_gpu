#!/usr/bin/env bash
# DETERMINISTIC combined diagnostic (1h): localize base-vs-s0s1 to S0/S1 AND
# measure the numeric MAGNITUDE of the deltas (ULP codegen-noise vs algorithmic).
# --xla_gpu_deterministic_ops=true makes the forecast bit-reproducible
# (base-vs-base=0/180 proven), so any nonzero hash diff is a real graph diff;
# delta_magnitude.py then says whether it is fp64-ULP (Tier-P) or large (bug).
set -euo pipefail
ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"; mkdir -p "$OUT"; cd "$ROOT"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"

mktree () { local d; d="$(mktemp -d /tmp/v016_tree.XXXXXX)"; git archive "$1" src | tar -x -C "$d"
  for e in "$ROOT"/* "$ROOT"/.[!.]*; do local n; n="$(basename "$e")"; case "$n" in src|.git|.git*) continue;; esac; [ -e "$e" ] && ln -sfn "$e" "$d/$n"; done; echo "$d"; }
BASE_T="$(mktree 499d3724)"; S0_T="$(mktree 12ed00be)"
run () { PYTHONPATH="$2" python "$PROBE" --tag "$1" --hours 1 --dump-state
         cp -f "proofs/perf/v015/ab_$1.json" "$OUT/"; cp -f "proofs/perf/v015/ab_$1_state.npz" "$OUT/"; }
cmp () { PYTHONPATH="$ROOT/src" python - "$1" "$2" "$3" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["hashes"]; b=json.load(open(sys.argv[2]))["hashes"]; lab=sys.argv[3]
mm=[];tot=0
for h in sorted(set(a)|set(b)):
    al=a.get(h,{});bl=b.get(h,{})
    for k in sorted(set(al)|set(bl)):
        tot+=1
        if al.get(k)!=bl.get(k): mm.append(f"{h}:{k}")
print(f"[hash {lab}] {len(mm)}/{tot} mismatches"+("" if not mm else f"  {mm[:8]}"))
PY
}
run base_dg "$BASE_T/src"; run s0_dg "$S0_T/src"; run s0s1_dg "$ROOT/src"
echo "================ DETERMINISTIC DIAG (1h) — hash localization ================"
cmp "$OUT/ab_base_dg.json" "$OUT/ab_s0_dg.json"   "S0 carry-leaf base-vs-s0"
cmp "$OUT/ab_s0_dg.json"   "$OUT/ab_s0s1_dg.json" "S1 seam only  s0-vs-s0s1"
cmp "$OUT/ab_base_dg.json" "$OUT/ab_s0s1_dg.json" "S0+S1 full    base-vs-s0s1"
echo "================ MAGNITUDE (final-state npz, fp64) ================"
python proofs/perf/v016/delta_magnitude.py "$OUT/ab_base_dg_state.npz" "$OUT/ab_s0_dg_state.npz"   "S0  base-vs-s0"  || true
python proofs/perf/v016/delta_magnitude.py "$OUT/ab_base_dg_state.npz" "$OUT/ab_s0s1_dg_state.npz" "S0S1 base-vs-s0s1" || true
rm -rf "$BASE_T" "$S0_T"; echo "================ done ================"
