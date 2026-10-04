#!/usr/bin/env bash
# v020_s0_instrument.sh — S0 INSTRUMENT (roadmap step 0): clean v0.19.1 warm nsys
# profile of the nested all-7 max_dom=9 benchmark + per-domain CFL probe + launch/
# kernel-count extractor. The first thing the manager fires when the GPU frees.
#
# WHAT IT MEASURES (the "how host-bound is v0.19.1 NOW" question OPUS_P34_PLAN §S0):
#   1. a WARM nsys trace of the all-7 1 h forecast (warm = persistent XLA cache hot,
#      so compile is excluded) -> host launch count, GPU kernel instances + time,
#      top kernel families, in-loop memcpy (transfer audit), GPU-active/wall proxy.
#   2. a per-domain realized-CFL probe on the resulting wrfout (the T0 headroom table
#      that sizes the P4 dt/n_sound ladder), via scripts/v020_cfl_probe.py.
#   3. the launch/kernel-count extractor (scripts/v020_nsys_extract.py) reducing the
#      nsys CSVs into one JSON + table.
#
# GPU SAFETY: the ONE nsys+forecast command is wrapped in with_gpu_lock.sh via
# v020_run_gpu, and is SKIPPED entirely under V020_DRYRUN=1 (CPU dry-run validates
# arg-parse + the two CPU extractors on EXISTING artifacts). All non-GPU steps run
# in both modes.
#
# Usage:
#   scripts/v020_s0_instrument.sh                 # FULL (holds GPU lock)
#   V020_DRYRUN=1 scripts/v020_s0_instrument.sh   # CPU dry-run (no GPU)
# Env knobs: V020_ALL7_INPUT, V020_DURATION (nsys capture seconds), V020_HOURS.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v020_probe_common.sh"

: "${V020_DURATION:=900}"   # nsys bounded capture (s): ~7 min warm init + steady fc
: "${V020_HOURS:=1}"        # forecast hours to advance under the profiler
INPUT="$V020_ALL7_INPUT"

v020_assert_case "$INPUT" || { v020_log "S0 abort: missing all-7 case"; exit 1; }

RR="$(v020_mk_rundir s0_instrument)"
GPU_OUT="$RR/gpu_output"; PROOFS="$RR/proofs"; SCRATCH="$RR/scratch"
mkdir -p "$GPU_OUT" "$PROOFS" "$SCRATCH"
v020_log "S0 run dir: $RR  (dryrun=$V020_DRYRUN)"

cat > "$RR/runinfo.txt" <<EOF
probe=S0_instrument
benchmark=nested_all7_max_dom9 (the stability/skill gate)
input=$INPUT
measures=nsys_warm_trace + per_domain_CFL + launch/kernel-count
nsys_duration_s=$V020_DURATION
forecast_hours=$V020_HOURS
warm=persistent_xla_cache (JAX_COMPILATION_CACHE_DIR=$V020_JAX_CACHE)
EOF

NSYS_REP="$RR/nsys_all7"
NSYS_LOG="$RR/nsys.log"
STATS_PREFIX="$RR/nsys_stats"

# --- the GPU command: nsys-wrapped warm all-7 forecast ----------------------------
# Two-pass design for a clean WARM trace: the warm cache is reused, so the captured
# window is steady-state forecast, not compile. --duration bounds the GPU hold.
build_gpu_cmd() {
  # emits the argv (one token per line via NUL-safe printf in the caller); here we
  # just construct the array the caller passes to v020_run_gpu.
  GPU_CMD=(
    env
    PYTHONPATH="$V020_ROOT/src"
    JAX_ENABLE_X64=true
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_ENABLE_COMPILATION_CACHE=true
    JAX_COMPILATION_CACHE_DIR="$V020_JAX_CACHE"
    OMP_NUM_THREADS="$V020_OMP"
    GPUWRF_MYNN_BOULAC_ONZ=1
    GPUWRF_NESTED_SYNC_MODE=root
    XLA_PYTHON_CLIENT_ALLOCATOR=platform
    GPUWRF_SCRATCH="$SCRATCH"
    nsys profile
      --trace=cuda,nvtx,osrt
      --sample=none --cpuctxsw=none
      --cuda-memory-usage=false
      --gpu-metrics-devices=all
      --duration="$V020_DURATION"
      --force-overwrite=true
      --output="$NSYS_REP"
    taskset -c "$V020_TASKSET"
    python -m gpuwrf run
      --input-dir "$INPUT"
      --namelist "$INPUT/namelist.input"
      --output-dir "$GPU_OUT"
      --proof-dir "$PROOFS"
      --scratch-dir "$SCRATCH"
      --max-dom 9
      --hours "$V020_HOURS"
  )
}
build_gpu_cmd

START=$(date -u +%s)
v020_run_gpu "opus-s0-instrument" "$((V020_DURATION + 2400))" -- "${GPU_CMD[@]}" \
  > "$NSYS_LOG" 2>&1
GPU_RC=$?
END=$(date -u +%s)
WALL=$((END - START))
echo "$GPU_RC" > "$RR/s0_gpu.rc"
echo "$WALL"   > "$RR/s0_wall_s.txt"
v020_log "S0 GPU step rc=$GPU_RC wall=${WALL}s (0 in dry-run = skipped)"

# --- nsys stats export (CPU; only meaningful after a real capture) ----------------
# Skipped in dry-run unless an .nsys-rep already exists for the extractor to chew.
if [[ "$V020_DRYRUN" != "1" && -f "$NSYS_REP.nsys-rep" ]]; then
  v020_log "exporting nsys stats CSVs..."
  nsys stats --force-overwrite=true --force-export=true \
    --report cuda_api_sum --report cuda_gpu_kern_sum \
    --report cuda_gpu_sum --report cuda_gpu_mem_time_sum \
    --format csv --output "$STATS_PREFIX" "$NSYS_REP.nsys-rep" \
    > "$RR/nsys_stats.log" 2>&1 || v020_log "nsys stats export had warnings"
fi

# --- CPU extractor #1: launch/kernel-count -----------------------------------------
# In dry-run, point at an EXISTING nsys CSV set to prove the extractor works.
EXTRACT_PREFIX="$STATS_PREFIX"
if [[ "$V020_DRYRUN" == "1" ]]; then
  # find ANY existing kernel-sum CSV in the repo and derive its stats prefix, so the
  # extractor logic is genuinely exercised in the dry-run (not just no-op'd).
  any_csv="$(find "$V020_ROOT/proofs" -name '*cuda_gpu_kern_sum*.csv' 2>/dev/null | head -1)"
  if [[ -n "$any_csv" ]]; then
    EXTRACT_PREFIX="${any_csv%%cuda_gpu_kern_sum*}"
    EXTRACT_PREFIX="${EXTRACT_PREFIX%_}"
  fi
  v020_log "dry-run: nsys extractor will read existing CSVs at prefix '$EXTRACT_PREFIX'"
fi
WALL_ARG=()
[[ -f "$RR/s0_wall_s.txt" && "$V020_DRYRUN" != "1" ]] && WALL_ARG=(--wall-s "$WALL")
taskset -c "$V020_TASKSET" python "$HERE/v020_nsys_extract.py" \
  --stats-prefix "$EXTRACT_PREFIX" "${WALL_ARG[@]}" \
  --out "$RR/s0_launch_kernel_counts.json" \
  > "$RR/s0_extract.log" 2>&1 || v020_log "nsys extractor found no CSVs (expected if no capture yet)"

# --- CPU extractor #2: per-domain CFL probe ---------------------------------------
# Prefer this run's fresh wrfout; in dry-run fall back to any existing all-7 wrfout set.
CFL_RUN_DIR="$GPU_OUT"
if ! compgen -G "$CFL_RUN_DIR/wrfout_d01_*" > /dev/null 2>&1; then
  for cand in "$GPU_OUT" <DATA_ROOT>/wrf_downscale/canary_all7/cpu_run \
              <DATA_ROOT>/wrf_downscale/canary_all7/run_cadvariant ; do
    if compgen -G "$cand/wrfout_d01_*" > /dev/null 2>&1; then CFL_RUN_DIR="$cand"; break; fi
  done
  v020_log "CFL probe: using fallback wrfout dir $CFL_RUN_DIR (no fresh output yet)"
fi
taskset -c "$V020_TASKSET" python "$HERE/v020_cfl_probe.py" \
  --run-dir "$CFL_RUN_DIR" \
  --namelist "$INPUT/namelist.input" \
  --max-dom 9 --time LAST \
  --out "$RR/s0_per_domain_cfl.json" \
  > "$RR/s0_cfl.log" 2>&1 || v020_log "CFL probe had no wrfout to read (expected pre-capture)"

# --- consolidate ------------------------------------------------------------------
{
  echo "S0 instrument artifacts ($RR):"
  echo "  nsys report     : $NSYS_REP.nsys-rep"
  echo "  launch/kernel   : $RR/s0_launch_kernel_counts.json"
  echo "  per-domain CFL  : $RR/s0_per_domain_cfl.json"
  echo "  gpu rc / wall   : $(cat "$RR/s0_gpu.rc" 2>/dev/null) / $(cat "$RR/s0_wall_s.txt" 2>/dev/null)s"
} | tee "$RR/S0_SUMMARY.txt"
v020_log "S0 DONE -> $RR/S0_SUMMARY.txt"
exit 0
