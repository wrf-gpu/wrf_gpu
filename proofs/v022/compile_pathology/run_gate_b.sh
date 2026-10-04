#!/usr/bin/env bash
# v0.22 compile-pathology kill-gate (b): fp32-BouLac mixed fp32/fp64 fusion.
# Runs BASELINE (fp32-boulac OFF) then FP32 (fp32-boulac ON) sequentially under
# ONE GPU lock so the cold-compile measurement is contention-free. Each python
# run is wrapped in `timeout` so a true STALL is caught (not waited on forever).
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
OUT="$(pwd)/proofs/v022/compile_pathology"
SRC="$(pwd)/src"
HARNESS="$OUT/gate_b_harness.py"

# Per-config hard cap. Prior verdict: clean ~80s compile; pathological crawls
# >4min then STALLS. 1500s (25min) is generous headroom: completing well under
# it = bounded; hitting it = stall (caught, rc=124).
PER_CONF_TIMEOUT=1500

run_one () {
  local tag="$1"; shift
  echo "===== CONFIG $tag : env $* ====="
  echo "START $(date -u +%H:%M:%SZ)"
  env "$@" \
    GPUWRF_MYNN_COND_NITER=16 \
    JAX_ENABLE_X64=true GPUWRF_FORCE_FP64=1 \
    PYTHONPATH="$SRC" \
    timeout -k 30 "$PER_CONF_TIMEOUT" \
      python "$HARNESS" --tag "$tag" --hours 3 \
      > "$OUT/${tag}.stdout" 2> "$OUT/${tag}.stderr"
  local rc=$?
  echo "END   $(date -u +%H:%M:%SZ) rc=$rc"
  echo "[$tag] timeout-rc=$rc (124 => STALL/timeout)"
  echo "[$tag] slow-compile-alarms=$(grep -ci 'very slow compile' "$OUT/${tag}.stderr" 2>/dev/null || echo 0)"
  echo "[$tag] stdout-tail:"; tail -6 "$OUT/${tag}.stdout" 2>/dev/null
  echo "[$tag] rc-file write"; echo "$rc" > "$OUT/${tag}.rc"
  return 0
}

# 1) clean baseline (establishes the clean cold-compile wall + steady runtime)
run_one baseline GPUWRF_MYNN_BOULAC_FP32=0

# 2) the kill-gate: fp32 BouLac ON in the full operational pipeline
run_one fp32boulac GPUWRF_MYNN_BOULAC_FP32=1

echo "===== GATE B DONE $(date -u +%H:%M:%SZ) ====="
