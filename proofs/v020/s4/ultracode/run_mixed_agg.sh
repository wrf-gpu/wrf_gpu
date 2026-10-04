#!/usr/bin/env bash
# run_mixed_agg.sh — bigswiss 1h for the mixed + aggressive fp32 arms (fp64 arm
# reuses the earlier probe). Holds the GPU lock for both. forecast-only ms/step
# is read from the pipeline proof JSON (excludes compile/init); peak VRAM from a
# 1 Hz nvidia-smi sampler (platform allocator => true peak incl RRTMG transient).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
INPUT="${V020_BIG_INPUT:-<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init}"
JAX_CACHE="${V020_JAX_CACHE:-<DATA_ROOT>/gpuwrf_jax_cache}"
RR="$HERE/mixed_agg_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$RR"
echo "[ma] rundir=$RR"

run_arm() {  # tag extra_env...
  local tag="$1"; shift
  local out="$RR/$tag/out"; local proofs="$RR/$tag/proofs"; local scratch="$RR/$tag/scratch"
  local log="$RR/$tag.log"; local vram="$RR/$tag.vram.csv"
  mkdir -p "$out" "$proofs" "$scratch"
  ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader 2>/dev/null >> "$vram"; sleep 1; done ) & local vpid=$!
  echo "[ma] $tag start $(date -u +%H:%M:%SZ)  env: $*"
  env PYTHONPATH="$ROOT/src" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" \
      OMP_NUM_THREADS=4 GPUWRF_MYNN_BOULAC_ONZ=1 "$@" \
      taskset -c 0-3 timeout 1500 python -m gpuwrf run \
        --input-dir "$INPUT" --namelist "$INPUT/namelist.input" \
        --output-dir "$out" --proof-dir "$proofs" --scratch-dir "$scratch" \
        --max-dom 1 --domain d01 --hours 1 > "$log" 2>&1
  local rc=$?
  kill "$vpid" 2>/dev/null || true
  local peak; peak=$(grep -oE "[0-9]+ MiB" "$vram" | grep -oE "[0-9]+" | sort -n | tail -1)
  local oom="false"; grep -qiE "RESOURCE_EXHAUSTED|out of memory|CUDA_ERROR_OUT_OF_MEMORY" "$log" && oom="true"
  local fo; fo=$(python3 -c "import json,glob,sys
fs=glob.glob('$proofs/pipeline_run*.json')
print(json.load(open(fs[0]))['wall_clock_forecast_only_s'] if fs else 'null')" 2>/dev/null)
  echo "[ma] $tag rc=$rc forecast_only_s=$fo peak=${peak:-?}MiB oom=$oom"
  printf '{"tag":"%s","rc":%s,"forecast_only_s":%s,"peak_vram_mib":%s,"oom":%s,"wrfout":"%s"}\n' \
    "$tag" "$rc" "${fo:-null}" "${peak:-null}" "$oom" "$(ls $out/wrfout_* 2>/dev/null | head -1)" > "$RR/$tag.metric.json"
}

run_arm mixed      GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020
run_arm aggressive GPUWRF_FORCE_FP64=0
echo "[ma] DONE -> $RR"
ls "$RR"/*.metric.json
