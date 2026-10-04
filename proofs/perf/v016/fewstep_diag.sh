#!/usr/bin/env bash
# Measure the PER-STEP delta seed base-vs-s0s1 (deterministic XLA) at a few
# horizons (~2 steps, ~10 steps) to separate a ULP codegen seed (Tier-P; chaos
# amplifies) from a systematic algorithmic error (large from step 1).
set -euo pipefail
ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
P="proofs/perf/v016/fewstep_probe.py"; OUT="$ROOT/proofs/perf/v016"; cd "$ROOT"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
mktree () { local d; d="$(mktemp -d /tmp/v016_tree.XXXXXX)"; git archive "$1" src | tar -x -C "$d"
  for e in "$ROOT"/* "$ROOT"/.[!.]*; do local n; n="$(basename "$e")"; case "$n" in src|.git|.git*) continue;; esac; [ -e "$e" ] && ln -sfn "$e" "$d/$n"; done; echo "$d"; }
BASE_T="$(mktree 499d3724)"
# Step-aligned horizons (probe converts steps->hours exactly from namelist dt).
for N in 1 5 20; do
  echo "===== ${N} step(s) ====="
  PYTHONPATH="$BASE_T/src" python "$P" --tag base_s${N} --steps "$N"
  PYTHONPATH="$ROOT/src"   python "$P" --tag s0s1_s${N} --steps "$N"
  python proofs/perf/v016/delta_magnitude.py "$OUT/fewstep_base_s${N}.npz" "$OUT/fewstep_s0s1_s${N}.npz" "base-vs-s0s1 @ ${N} step(s)" || true
done
rm -rf "$BASE_T"; echo "===== done ====="
