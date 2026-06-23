#!/usr/bin/env bash
# v020_ab_nested_async_output.sh -- READY-TO-FIRE GPU A/B for the v0.20 nested
# host-bubble levers (group G_nested_pipeline). DO NOT run while the GPU is held
# by another job; this wraps the shared GPU lock so it will block until free.
#
# Levers measured (both numerics-free; the compiled HLO is identical across arms,
# so the JAX persistent compilation cache is shared and every timed arm is WARM):
#
#   L1  Async history output on the nested path
#       knob: GPUWRF_NESTED_ASYNC_OUTPUT  (default ON; =0 forces the legacy
#       synchronous step-thread NetCDF write). Removes the ~30 s/output-group
#       synchronous write stall by overlapping the write with GPU compute.
#
#   L2  root_sync_cadence sweep (GPUWRF_NESTED_SYNC_MODE=root:K)
#       knob already exists; the default is root:1. This sweep finds the K that
#       maximizes utilization without a peak-VRAM regression. block_until_ready
#       is a pure host wait -> byte-identical wrfout at every K; only the DEFAULT
#       (a one-line change in _nested_sync_mode_from_env) would change after the
#       sweep concludes a better K.
#
# WHAT IS MEASURED: wall_clock_forecast_only_s / hours  (s per forecast hour) for
# each arm, read from the nested run payload JSON. Lower is better. The peak VRAM
# per arm is captured from a 1 Hz nvidia-smi sampler running alongside the run
# (an A/B is only valid if peak VRAM does NOT regress).
#
# Usage:
#   scripts/with_gpu_lock.sh --label v020-async-ab -- \
#       scripts/v020_ab_nested_async_output.sh
#
# (The script re-invokes nothing else under the lock; run it INSIDE the lock as
#  above so the whole A/B holds the GPU exactly once.)
#
# Env (all optional; defaults target the v0.18/v0.19 all-7 9-domain case):
#   INPUT_DIR   case dir with wrfinput_d0N + wrfbdy_d01 + namelist.input
#               (default: the v0.18 max_dom9 all-7 ac1fit case)
#   MAX_DOM     number of nested domains (default 9 = all-7 9-domain nest)
#   HOURS       forecast hours per arm (default 2; >=2 so segment-2+ is warm even
#               without the persistent cache)
#   OUT_ROOT    output/proof root (default: proofs/v020/nested_async_ab/<ts>)
#   SYNC_K_LIST space-separated K values for the L2 sweep (default "1 2 3")
#   SKIP_L2     set to 1 to measure only the async-output lever (L1)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-<DATA_ROOT>/wrf_downscale/nested_canary_training/ac1fit_20260614T220802Z/run_cpu}"
MAX_DOM="${MAX_DOM:-9}"
HOURS="${HOURS:-2}"
SYNC_K_LIST="${SYNC_K_LIST:-1 2 3}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-proofs/v020/nested_async_ab/${TS}}"

# Shared JAX persistent compilation cache so cross-arm runs reuse the SAME
# compiled binaries (the levers do not change the HLO). Makes every timed arm
# warm even though it runs in a fresh process.
CACHE_DIR="${OUT_ROOT}/jax_cache"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT/$CACHE_DIR}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_ROOT" "$JAX_COMPILATION_CACHE_DIR"

if [[ ! -f "$INPUT_DIR/namelist.input" ]]; then
  echo "ERROR: INPUT_DIR has no namelist.input: $INPUT_DIR" >&2
  echo "       Point INPUT_DIR at the canonical all-7 9-domain case." >&2
  exit 2
fi

echo "=== v0.20 nested host-bubble A/B ==="
echo "  root=$ROOT"
echo "  input=$INPUT_DIR  max_dom=$MAX_DOM  hours=$HOURS"
echo "  out=$OUT_ROOT  jax_cache=$JAX_COMPILATION_CACHE_DIR"
echo

# ---------------------------------------------------------------------------
# run_arm <label> -- runs one nested forecast with the current env, samples peak
# VRAM, and prints "<label>  s/fc-h=<v>  peakVRAM_MiB=<v>".
# ---------------------------------------------------------------------------
run_arm() {
  local label="$1"; shift
  local arm_out="$OUT_ROOT/$label"
  rm -rf "$arm_out"; mkdir -p "$arm_out"
  local smi_log="$arm_out/nvidia_smi.csv"

  # 1 Hz peak-VRAM sampler for the duration of this arm.
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
    > "$smi_log" 2>/dev/null &
  local smi_pid=$!

  set +e
  taskset -c "${AB_TASKSET:-0-3}" python -m gpuwrf run \
    --input-dir "$INPUT_DIR" \
    --output-dir "$arm_out/out" \
    --proof-dir "$arm_out/proof" \
    --max-dom "$MAX_DOM" \
    --hours "$HOURS" \
    > "$arm_out/run.log" 2>&1
  local rc=$?
  set -e

  kill "$smi_pid" 2>/dev/null || true
  wait "$smi_pid" 2>/dev/null || true

  if [[ $rc -ne 0 ]]; then
    echo "$label  FAILED rc=$rc (see $arm_out/run.log)"
    return $rc
  fi

  python - "$arm_out/proof/nested_pipeline_run.json" "$smi_log" "$HOURS" "$label" <<'PY'
import json, sys
payload_path, smi_log, hours, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
with open(payload_path) as fh:
    p = json.load(fh)
fc_s = float(p["wall_clock_forecast_only_s"])
s_per_fc_h = fc_s / hours
peak = 0
try:
    with open(smi_log) as fh:
        for line in fh:
            line = line.strip()
            if line:
                peak = max(peak, int(float(line)))
except FileNotFoundError:
    peak = -1
print(f"{label}  s/fc-h={s_per_fc_h:.1f}  forecast_only_s={fc_s:.1f}  "
      f"peakVRAM_MiB={peak}  finite={p.get('all_domains_finite')}  "
      f"outputs_present={p.get('all_outputs_present')}")
PY
}

# ---------------------------------------------------------------------------
# WARM-UP: one throwaway run to populate the persistent compilation cache so
# every TIMED arm below reuses the compiled binaries (identical HLO across arms).
# ---------------------------------------------------------------------------
echo "--- warm-up (populating JAX compile cache; timing discarded) ---"
GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:1 run_arm warmup || {
  echo "warm-up failed; aborting A/B" >&2; exit 1; }
echo

# ---------------------------------------------------------------------------
# L1: async history output ON vs OFF (default sync cadence root:1).
# ---------------------------------------------------------------------------
echo "=== L1  async history output (nested) ==="
GPUWRF_NESTED_SYNC_MODE=root:1 GPUWRF_NESTED_ASYNC_OUTPUT=0 run_arm L1_async_OFF
GPUWRF_NESTED_SYNC_MODE=root:1 GPUWRF_NESTED_ASYNC_OUTPUT=1 run_arm L1_async_ON
echo

# ---------------------------------------------------------------------------
# L2: root_sync_cadence sweep (async output ON, the intended default). Pick the K
# that minimizes s/fc-h WITHOUT a peak-VRAM regression vs K=1; that K becomes the
# new default in _nested_sync_mode_from_env (one-line change), else keep root:1.
# ---------------------------------------------------------------------------
if [[ "${SKIP_L2:-0}" != "1" ]]; then
  echo "=== L2  root_sync_cadence sweep (async output ON) ==="
  for K in $SYNC_K_LIST; do
    GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE="root:$K" run_arm "L2_sync_root_$K"
  done
  echo
fi

echo "=== DONE.  Compare s/fc-h (lower=better); reject any arm whose peakVRAM_MiB"
echo "    regresses vs the root:1 baseline. Byte-identity of wrfout across L1"
echo "    arms is already CPU-proven by tests/test_v014_noahmp_nested_pipeline.py"
echo "    ::test_nested_async_output_byte_identical_to_sync."
echo "    Results under: $OUT_ROOT"
