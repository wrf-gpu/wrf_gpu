#!/usr/bin/env bash
# R0 compile-trilemma memory-fitting FLAG SWEEP (v0.22).
#
# For each candidate flag setting, on the FUSED cold compile (3-dom by default;
# MAXDOM=9 for the 9-nest stress when a long window + RAM are available), record:
#   - cold compile wall (s)
#   - peak host RSS / VmHWM (MiB, via ru_maxrss in perstep_timing_driver.py)
#   - serialized AOT-blob validity (a FRESH warm process must load source=aot_blob,
#     no jit_fused re-lower) AND bit-identical state digest (vs this setting's own
#     cold capture AND vs the no-flag baseline capture)
#   - runtime s/step (must stay == fused baseline; HARD-FAIL on >1-2% regression)
#
# PASS criterion (per setting) = materially-lower peak RAM AND zero runtime
# regression AND a valid+bit-identical AOT blob. Outcome is binary: a passing
# setting -> adopt; else a clean IMPOSSIBILITY proof + ship-the-blob remains.
#
# Each setting runs in its OWN fresh JAX cache dir so the cold compile is real
# (no warm hit from a prior setting). Capture phase = cold compile + serialize.
# Warm phase = fresh process loads the blob (proves blob valid + no re-lower).
#
# RUN UNDER THE GPU LOCK:
#   scripts/with_gpu_lock.sh --timeout 21600 --label r0-flag-sweep -- \
#     proofs/v022/r0_flag_sweep/run_r0_sweep.sh
#
# Env knobs:
#   R0_MAXDOM   (default 3)   3-dom fused, or 9 for the 9-nest stress
#   R0_SETTINGS (default "baseline mfe1 mfe05 mfl_O3 llvm_cheap mfe1_mfl_O3")
#   R0_KWARM/R0_FIXED_REPEATS for the s/step measurement
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

MAXDOM="${R0_MAXDOM:-3}"
KWARM="${R0_KWARM:-6}"
FIXED_REPEATS="${R0_FIXED_REPEATS:-3}"
SETTINGS="${R0_SETTINGS:-baseline mfe1 mfe05 mfl_O3 llvm_cheap mfe1_mfl_O3}"
OUT=proofs/v022/r0_flag_sweep
LOGS="$OUT/logs"
mkdir -p "$LOGS"
RUN_ID="${R0_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS="$OUT/results_maxdom${MAXDOM}_${RUN_ID}.jsonl"
: > "$RESULTS"

INPUT="${CANARY_INPUT_DIR:-<DATA_ROOT>/wrf_downscale/runs/20240901/cpu}"

# Map a setting name -> the env-var(s) that select its compile flags.
setting_env () {  # echoes "NAME=VAL" pairs (one per line); empty for baseline
  case "$1" in
    baseline)      : ;;
    mfe1)          echo "JAX_MEMORY_FITTING_EFFORT=1.0" ;;
    mfe05)         echo "JAX_MEMORY_FITTING_EFFORT=0.5" ;;
    mfl_O3)        echo "JAX_MEMORY_FITTING_LEVEL=O3" ;;
    mfe1_mfl_O3)   echo "JAX_MEMORY_FITTING_EFFORT=1.0"; echo "JAX_MEMORY_FITTING_LEVEL=O3" ;;
    llvm_cheap)    echo "XLA_FLAGS=--xla_llvm_disable_expensive_passes=true" ;;
    noremat)       echo "JAX_COMPILER_ENABLE_REMAT_PASS=false" ;;
    *) echo "UNKNOWN_SETTING_$1=1" ;;
  esac
}

# Baseline reference digest (filled after the baseline capture runs first).
BASE_REF=""

run_one () {  # $1 setting
  local setting="$1"
  local cache_dir="/tmp/gpuwrf_r0_${setting}_md${MAXDOM}_${RUN_ID}"
  local cap_log="$LOGS/cap_${setting}_md${MAXDOM}_${RUN_ID}.log"
  local warm_log="$LOGS/warm_${setting}_md${MAXDOM}_${RUN_ID}.log"
  rm -rf "$cache_dir"; mkdir -p "$cache_dir"
  rm -f "$cap_log" "$warm_log"

  # Per-setting flag env lines.
  mapfile -t FLAGENV < <(setting_env "$setting")

  COMMON_ENV=(
    OMP_NUM_THREADS=8
    PYTHONPATH=src
    JAX_ENABLE_X64=true
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_COMPILATION_CACHE_DIR="$cache_dir"
    GPUWRF_JAX_CACHE_DIR="$cache_dir"
    GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo
    CANARY_INPUT_DIR="$INPUT"
    CANARY_MAXDOM="$MAXDOM"
    MAXDOM="$MAXDOM"
    CANARY_KWARM="$KWARM"
    CANARY_SINGLE_SHAPE_TIMING=1
    CANARY_FIXED_ROOT_STEPS=1
    CANARY_FIXED_REPEATS="$FIXED_REPEATS"
    GPUWRF_NESTED_AOT=1
    GPUWRF_NESTED_PARALLEL_COMPILE=0
  )

  echo "================ R0 SETTING=$setting MAXDOM=$MAXDOM flags=[${FLAGENV[*]}] ================"

  # --- CAPTURE: cold compile + serialize blob (peak RSS here = cold-compile peak) ---
  local cap_proof="$OUT/ts_proof_${setting}_md${MAXDOM}"
  /usr/bin/time -v env "${COMMON_ENV[@]}" "${FLAGENV[@]}" \
    CANARY_TAG="r0_${setting}_cap" \
    python proofs/v021/canary_gate/perstep_timing_driver.py \
    > "$cap_log" 2>&1
  local cap_rc=$?

  # The fixed-mode driver writes its digest under ts_proof_<TAG>/.
  local cap_ref="proofs/v021/canary_gate/ts_proof_r0_${setting}_cap/warmK_state_digest.json"

  # --- WARM: fresh process loads the blob (proves blob valid + no re-lower) ---
  # Compare bit-identity vs (a) this setting's own cold digest and (b) baseline.
  local ref_for_warm="$cap_ref"
  if [[ -n "$BASE_REF" && -f "$BASE_REF" ]]; then ref_for_warm="$BASE_REF"; fi
  /usr/bin/time -v env "${COMMON_ENV[@]}" "${FLAGENV[@]}" \
    CANARY_TAG="r0_${setting}_warm" \
    CANARY_REF_DIGEST="$ref_for_warm" JAX_LOG_COMPILES=1 \
    python proofs/v021/canary_gate/perstep_timing_driver.py \
    > "$warm_log" 2>&1
  local warm_rc=$?

  # On the baseline setting, capture its digest as THE reference for all others.
  if [[ "$setting" == "baseline" && -f "$cap_ref" ]]; then
    BASE_REF="$(pwd)/$cap_ref"
    echo "MARKER:BASELINE_REF=$BASE_REF"
  fi

  # --- Extract metrics into the JSONL results ---
  python proofs/v022/r0_flag_sweep/extract_r0.py \
    --setting "$setting" --maxdom "$MAXDOM" \
    --flags "${FLAGENV[*]:-}" \
    --cap-log "$cap_log" --warm-log "$warm_log" \
    --cap-rc "$cap_rc" --warm-rc "$warm_rc" \
    --cache-dir "$cache_dir" \
    --base-ref "${BASE_REF:-}" \
    >> "$RESULTS"

  echo "---- setting $setting done (cap_rc=$cap_rc warm_rc=$warm_rc) ----"
}

# Baseline first (establishes the reference digest + reference RSS/s_per_step).
for s in $SETTINGS; do
  if [[ "$s" == "baseline" ]]; then run_one "$s"; fi
done
for s in $SETTINGS; do
  if [[ "$s" != "baseline" ]]; then run_one "$s"; fi
done

echo "================ R0 SWEEP COMPLETE -> $RESULTS ================"
python proofs/v022/r0_flag_sweep/extract_r0.py --summarize "$RESULTS"
