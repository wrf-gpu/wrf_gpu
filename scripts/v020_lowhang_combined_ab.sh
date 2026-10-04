#!/usr/bin/env bash
# v020_lowhang_combined_ab.sh -- UNIFIED warm A/B for the v0.20.0 low-hanging-fruit
# speed wave on the all-7 9-domain nested canary. Produces the headline numbers:
#   * v0.19 fp64 baseline (ALL levers OFF) warm s/fc-h
#   * v0.20 COMBINED (all bit-identical/numerics-free levers ON) warm s/fc-h
#   * each lever's marginal gain (toggle one lever off the baseline)
#   * root_sync_cadence sweep (K=1,2,3) on top of the combined config
#
# WHY a unified harness (vs the three per-lever scripts): every arm runs in the
# SAME process env (fp64 x64, GPUWRF_MYNN_BOULAC_ONZ=1, taskset 0-3, the SHARED
# persistent XLA cache) so cross-arm timings are apples-to-apples, and a single
# warm-up populates the cache once for ALL arms (the levers do not change the HLO,
# so every timed arm is warm). One GPU-lock holder for the whole session.
#
# All levers are bit-identical / numerics-free -> arms differ ONLY in wall clock
# and peak VRAM, never in the math. Lower s/fc-h wins; reject any arm whose peak
# VRAM regresses vs the v0.19 baseline.
#
# Env knobs (all optional):
#   INPUT_DIR    all-7 9-domain case (default: canary_all7/run)
#   MAX_DOM      domains (default 9)
#   HOURS        timed-arm forecast hours (default 2)
#   WARM_HOURS   warm-up forecast hours (default 1)
#   AB_TASKSET   host core pin (default 0-3; 4-31 is the CPU-WRF corpus)
#   JAX_CACHE    shared persistent compile cache (default <DATA_ROOT>/gpuwrf_jax_cache)
#   ARMS         space-separated arm list to run (default: all)
#   OUT_ROOT     artifact root (default proofs/v020/lowhang/combined_ab/<ts>)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-<DATA_ROOT>/wrf_downscale/canary_all7/run}"
MAX_DOM="${MAX_DOM:-9}"
HOURS="${HOURS:-2}"
WARM_HOURS="${WARM_HOURS:-1}"
AB_TASKSET="${AB_TASKSET:-0-3}"
JAX_CACHE="${JAX_CACHE:-<DATA_ROOT>/gpuwrf_jax_cache}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-proofs/v020/lowhang/combined_ab/${TS}}"
# Arms: name=ALLOCATOR:ASYNC:SYNCMODE  (the three speed levers).
#   v0.19 baseline = platform : 0 : root:1   (no levers)
#   combined       = cuda_async : 1 : root:1 (all levers, conservative K)
ALL_ARMS="v019_baseline alloc_only async_only combined combined_K2 combined_K3"
ARMS="${ARMS:-$ALL_ARMS}"

# Canonical fp64 warm-run env (matches the v0.18/v0.19 gate + allocator harness).
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export OMP_NUM_THREADS=4
export GPUWRF_MYNN_BOULAC_ONZ=1
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_ROOT" "$JAX_CACHE"

if [[ ! -f "$INPUT_DIR/namelist.input" ]]; then
  echo "ERROR: INPUT_DIR has no namelist.input: $INPUT_DIR" >&2; exit 2
fi

# arm_env NAME -> echoes the three per-arm GPUWRF assignments for that arm.
arm_env() {
  case "$1" in
    v019_baseline) echo "GPUWRF_ALLOCATOR=platform   GPUWRF_NESTED_ASYNC_OUTPUT=0 GPUWRF_NESTED_SYNC_MODE=root:1" ;;
    alloc_only)    echo "GPUWRF_ALLOCATOR=cuda_async GPUWRF_NESTED_ASYNC_OUTPUT=0 GPUWRF_NESTED_SYNC_MODE=root:1" ;;
    async_only)    echo "GPUWRF_ALLOCATOR=platform   GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:1" ;;
    combined)      echo "GPUWRF_ALLOCATOR=cuda_async GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:1" ;;
    combined_K2)   echo "GPUWRF_ALLOCATOR=cuda_async GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:2" ;;
    combined_K3)   echo "GPUWRF_ALLOCATOR=cuda_async GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:3" ;;
    warmup)        echo "GPUWRF_ALLOCATOR=cuda_async GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:1" ;;
    *) echo "" ;;
  esac
}

# run_arm NAME HOURS -> runs one forecast, samples 1 Hz peak VRAM, appends a row to results.tsv.
run_arm() {
  local name="$1" hrs="$2"
  local odir="$OUT_ROOT/$name"; rm -rf "$odir"; mkdir -p "$odir/out" "$odir/proof"
  local smi="$odir/nvidia_smi.csv"
  local envs; envs="$(arm_env "$name")"
  echo "--- ARM $name  ($envs  hours=$hrs  taskset=$AB_TASKSET) ---"

  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$smi" 2>/dev/null &
  local smi_pid=$!
  local start end wall rc
  start=$(date -u +%s)
  env $envs taskset -c "$AB_TASKSET" python -m gpuwrf run \
    --input-dir "$INPUT_DIR" --output-dir "$odir/out" --proof-dir "$odir/proof" \
    --max-dom "$MAX_DOM" --hours "$hrs" > "$odir/run.log" 2>&1
  rc=$?
  end=$(date -u +%s); wall=$((end - start))
  kill "$smi_pid" 2>/dev/null || true; wait "$smi_pid" 2>/dev/null || true

  local oom="no"
  grep -qiE "CUDA_ERROR_OUT_OF_MEMORY|RESOURCE_EXHAUSTED|Failed to allocate" "$odir/run.log" && oom="YES"
  echo "$rc" > "$odir/rc.txt"; echo "$wall" > "$odir/wall_s.txt"; echo "$oom" > "$odir/oom.txt"

  python - "$odir/proof/nested_pipeline_run.json" "$smi" "$hrs" "$name" "$rc" "$wall" "$oom" "$OUT_ROOT/results.tsv" <<'PY'
import json, sys, os
pj, smi, hours, name, rc, wall, oom, tsv = sys.argv[1:9]
hours = float(hours)
s_per_fc_h = fc_s = float("nan"); finite = present = "NA"
try:
    with open(pj) as fh: p = json.load(fh)
    fc_s = float(p["wall_clock_forecast_only_s"]); s_per_fc_h = fc_s / hours
    finite = p.get("all_domains_finite"); present = p.get("all_outputs_present")
except Exception as e:
    print(f"  (no/partial proof json: {e})")
peak = -1
try:
    with open(smi) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln: peak = max(peak, int(float(ln)))
except FileNotFoundError:
    pass
new = not os.path.exists(tsv)
with open(tsv, "a") as fh:
    if new: fh.write("arm\trc\ts_per_fc_h\tforecast_only_s\twall_s\tpeak_VRAM_MiB\tfinite\toutputs_present\toom\n")
    fh.write(f"{name}\t{rc}\t{s_per_fc_h:.1f}\t{fc_s:.1f}\t{wall}\t{peak}\t{finite}\t{present}\t{oom}\n")
print(f"  {name}: rc={rc} s/fc-h={s_per_fc_h:.1f} forecast_only_s={fc_s:.1f} "
      f"peakVRAM={peak}MiB finite={finite} present={present} oom={oom} wall={wall}s")
PY
}

echo "=== v0.20 low-hanging combined A/B  (input=$INPUT_DIR max_dom=$MAX_DOM) ==="
echo "    cache=$JAX_CACHE  out=$OUT_ROOT  arms='$ARMS'"
echo

# Warm-up: one throwaway run to ensure the persistent cache holds THIS branch's
# fp64 HLO so every timed arm below is warm (compile excluded from the timed window).
# SKIP_WARMUP=1 skips it when the shared cache is already hot (e.g. the H1 two-point
# anchor run that follows an H2 session).
if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
  echo "### warm-up (timing discarded) ###"
  run_arm warmup "$WARM_HOURS"
  echo
else
  echo "### warm-up SKIPPED (SKIP_WARMUP=1; assuming shared cache already hot) ###"
fi

for a in $ARMS; do run_arm "$a" "$HOURS"; done

echo
echo "=== RESULTS ($OUT_ROOT/results.tsv) ==="
cat "$OUT_ROOT/results.tsv"
