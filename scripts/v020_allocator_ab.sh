#!/usr/bin/env bash
# v020_allocator_ab.sh — READY-TO-FIRE A/B harness for the v0.20.0 speed lever
# G_allocator_env: nested GPU allocator default cuda_async (lever ON) vs the old
# platform synchronous cudaMalloc allocator (lever OFF / baseline).
#
# THE LEVER (numerics-free): the live-nested path now defaults its XLA allocator
# to GPUWRF_ALLOCATOR=cuda_async (CUDA stream-ordered pool) instead of the old
# platform (raw cudaMalloc/cudaFree, no pool). cuda_async pools per-op churn
# WITHOUT the BFC best-fit arena whose fragmentation caused the production
# "allocate 9.24 GiB" 1km-nest OOM. The allocator only governs WHERE device
# buffers live, NEVER the math -> the two arms must be tolerance-identical; only
# wall-clock (and peak VRAM) may differ.
#
# WHAT IT MEASURES: two WARM all-7 max_dom=9 nested forecasts (warm = persistent
# XLA cache hot, so compile is excluded from the timed window), ms/fc-h derived
# from the nested proof JSON (wall_clock_forecast_only_s / hours), for:
#   ARM A  (LEVER ON ): GPUWRF_ALLOCATOR=cuda_async   <- the new default
#   ARM B  (LEVER OFF): GPUWRF_ALLOCATOR=platform     <- the v0.19.x baseline
# and reports the speedup A_off/A_on. It ALSO records, for the cuda_async arm,
# the OOM-fit verdict (rc + any CUDA_ERROR_OUT_OF_MEMORY in the log) and the
# wrfout d01..d09 finite/present census from the proof JSON -- the MUST-NOT-
# REGRESS gate (the lever's one real risk: a fragmentation/fit regression on the
# 1km nest). To prove the 24h VRAM-flat property, fire with V020_HOURS=24.
#
# GPU SAFETY: every GPU command is wrapped in scripts/with_gpu_lock.sh via
# v020_run_gpu and is SKIPPED entirely under V020_DRYRUN=1 (CPU dry-run exercises
# arg-parse + the JSON reducer on existing artifacts, never touching the GPU).
# This script does NOT run the GPU itself when sourced/inspected; fire it
# explicitly. The GPU is held by the live fp32 1km demo as of authoring -- do NOT
# fire the GPU arms until it is free.
#
# Usage (fire LATER, when the GPU is free):
#   scripts/with_gpu_lock.sh --label opus-alloc-ab --timeout 36000 -- \
#     scripts/v020_allocator_ab.sh
#   # OR let the script take the lock per-arm (recommended; releases between arms):
#   scripts/v020_allocator_ab.sh
#   # CPU dry-run (no GPU; validates structure + reducer):
#   V020_DRYRUN=1 scripts/v020_allocator_ab.sh
#
# Env knobs: V020_ALL7_INPUT (case), V020_HOURS (default 1; use 24 for the VRAM-
# flat gate), V020_ARMS ("on off" default; e.g. "on" to time only the lever).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v020_probe_common.sh"

: "${V020_HOURS:=1}"            # forecast hours; 24 = the VRAM-flat / leak-guard gate
: "${V020_ARMS:=on off}"       # which arms to run (on=cuda_async, off=platform)
INPUT="$V020_ALL7_INPUT"

v020_assert_case "$INPUT" || { v020_log "alloc-ab abort: missing all-7 case"; exit 1; }

RR="$(v020_mk_rundir allocator_ab)"
v020_log "alloc-ab run dir: $RR  (dryrun=$V020_DRYRUN hours=$V020_HOURS arms='$V020_ARMS')"

cat > "$RR/runinfo.txt" <<EOF
probe=G_allocator_env_AB
lever=nested allocator default cuda_async (ON) vs platform (OFF)
numerics_free=true (allocator governs buffer placement, not math)
benchmark=nested_all7_max_dom9
input=$INPUT
metric=ms_per_forecast_hour = 1000 * wall_clock_forecast_only_s / hours
forecast_hours=$V020_HOURS
warm=persistent_xla_cache (JAX_COMPILATION_CACHE_DIR=$V020_JAX_CACHE)
arms=$V020_ARMS
must_not_regress=cuda_async arm fits (no OOM) + d01..d09 finite+present (+ VRAM flat at hours=24)
EOF

# run_arm ARM_NAME GPUWRF_ALLOCATOR_VALUE
#   Runs one warm nested forecast with the given allocator, writes its proof JSON
#   + log under $RR/<arm>/, and records rc. The production code path is exercised
#   via the GPUWRF_ALLOCATOR knob (NOT by forcing XLA_PYTHON_CLIENT_ALLOCATOR
#   directly), so this measures exactly what an operator gets.
run_arm() {
  local arm="${1:?need arm}" alloc="${2:?need allocator}"
  local odir="$RR/$arm"; local gout="$odir/gpu_output"; local proofs="$odir/proofs"
  local scratch="$odir/scratch"; local log="$odir/run.log"
  mkdir -p "$gout" "$proofs" "$scratch"
  v020_log "ARM $arm: GPUWRF_ALLOCATOR=$alloc (allocator default decides XLA_PYTHON_CLIENT_ALLOCATOR)"

  # NOTE: we DO NOT set XLA_PYTHON_CLIENT_ALLOCATOR here -- leaving it UNSET lets
  # cli.py:_resolve_nested_allocator map GPUWRF_ALLOCATOR -> the XLA allocator and
  # re-exec, exactly as in production. (v020_env_block defaults that XLA var to
  # 'platform', which would mask the lever, so we build the arm argv explicitly.)
  local cmd=(
    env
    PYTHONPATH="$V020_ROOT/src"
    JAX_ENABLE_X64=true
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_ENABLE_COMPILATION_CACHE=true
    JAX_COMPILATION_CACHE_DIR="$V020_JAX_CACHE"
    OMP_NUM_THREADS="$V020_OMP"
    GPUWRF_MYNN_BOULAC_ONZ=1
    GPUWRF_NESTED_SYNC_MODE=root
    GPUWRF_ALLOCATOR="$alloc"
    GPUWRF_SCRATCH="$scratch"
    taskset -c "$V020_TASKSET"
    python -m gpuwrf run
      --input-dir "$INPUT"
      --namelist "$INPUT/namelist.input"
      --output-dir "$gout"
      --proof-dir "$proofs"
      --scratch-dir "$scratch"
      --max-dom 9
      --hours "$V020_HOURS"
  )

  local start end wall rc
  start=$(date -u +%s)
  v020_run_gpu "opus-alloc-$arm" "$((V020_HOURS * 3600 + 5400))" -- "${cmd[@]}" \
    > "$log" 2>&1
  rc=$?
  end=$(date -u +%s); wall=$((end - start))
  echo "$rc"   > "$odir/rc.txt"
  echo "$wall" > "$odir/wall_s.txt"
  echo "$alloc" > "$odir/allocator.txt"
  # OOM-fit gate (the lever's one real risk): surface any CUDA OOM the run hit.
  if grep -qi "CUDA_ERROR_OUT_OF_MEMORY\|RESOURCE_EXHAUSTED\|Failed to allocate" "$log" 2>/dev/null; then
    echo "OOM_DETECTED" > "$odir/oom.txt"
  else
    echo "no_oom_strings" > "$odir/oom.txt"
  fi
  v020_log "ARM $arm rc=$rc wall=${wall}s oom=$(cat "$odir/oom.txt") (0 wall in dry-run = skipped)"
}

for arm in $V020_ARMS; do
  case "$arm" in
    on)  run_arm on  cuda_async ;;
    off) run_arm off platform   ;;
    *)   v020_log "unknown arm '$arm' (use on|off)"; ;;
  esac
done

# --- CPU reducer: derive ms/fc-h per arm + the speedup + the fit census --------
# In dry-run, fall back to any existing nested_pipeline_run.json so the reducer is
# genuinely exercised rather than no-op'd.
DRY_FALLBACK=""
if [[ "$V020_DRYRUN" == "1" ]]; then
  DRY_FALLBACK="$(find "$V020_ROOT/proofs" -name 'nested_pipeline_run.json' 2>/dev/null | head -1)"
  [[ -n "$DRY_FALLBACK" ]] && v020_log "dry-run: reducer will read existing $DRY_FALLBACK"
fi
taskset -c "$V020_TASKSET" python "$HERE/v020_allocator_ab_reduce.py" \
  --run-dir "$RR" --arms "$V020_ARMS" --hours "$V020_HOURS" \
  ${DRY_FALLBACK:+--dry-fallback-json "$DRY_FALLBACK"} \
  --out "$RR/allocator_ab_result.json" \
  > "$RR/reduce.log" 2>&1 || v020_log "reducer found no proof JSON yet (expected pre-capture)"

{
  echo "G_allocator_env A/B artifacts ($RR):"
  echo "  result JSON : $RR/allocator_ab_result.json"
  for arm in $V020_ARMS; do
    echo "  arm $arm    : rc=$(cat "$RR/$arm/rc.txt" 2>/dev/null) wall=$(cat "$RR/$arm/wall_s.txt" 2>/dev/null)s oom=$(cat "$RR/$arm/oom.txt" 2>/dev/null)"
  done
  echo "  (speedup A_off/A_on + ms/fc-h + fit census are in the result JSON)"
} | tee "$RR/ALLOC_AB_SUMMARY.txt"
v020_log "alloc-ab DONE -> $RR/ALLOC_AB_SUMMARY.txt"
exit 0
