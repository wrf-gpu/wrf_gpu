#!/usr/bin/env bash
# Reproducible v0.20 S4 1 km full-physics demo case runner.
#
# Usage:
#   ART_DIR=proofs/v020/s4/1km_demo_YYYYMMDDTHHMMSSZ \
#   CASE_TIMEOUT_S=7200 \
#   proofs/v020/s4/run_1km_demo_case.sh TAG REGIME NX NY NZ STEPS RAD_CADENCE SEGMENT_STEPS TILE_COLS LW_CLOUD_FP32
#
# REGIME is "fp64" or "aggressive".  This script holds the repository GPU lock,
# pins CPU work to cores 0-3, records stdout/stderr, samples nvidia-smi once per
# second, and emits TAG.metric.json next to the run JSON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

if [[ "${GPUWRF_1KM_RUNNER_INNER:-0}" != "1" ]]; then
  label="${GPU_LOCK_LABEL:-codex-v020-s4-1km-${1:-case}}"
  lock_timeout="${GPU_LOCK_TIMEOUT_S:-7200}"
  exec "$ROOT/scripts/with_gpu_lock.sh" --timeout "$lock_timeout" --label "$label" -- \
    env GPUWRF_1KM_RUNNER_INNER=1 "$0" "$@"
fi

if [[ $# -ne 10 ]]; then
  echo "usage: $0 TAG REGIME NX NY NZ STEPS RAD_CADENCE SEGMENT_STEPS TILE_COLS LW_CLOUD_FP32" >&2
  exit 2
fi

tag="$1"
regime="$2"
nx="$3"
ny="$4"
nz="$5"
steps="$6"
rad_cadence="$7"
segment_steps="$8"
tile_cols="$9"
lw_cloud_fp32="${10}"

case "$regime" in
  fp64|aggressive) ;;
  *) echo "REGIME must be fp64 or aggressive, got '$regime'" >&2; exit 2;;
esac

case "$lw_cloud_fp32" in
  0|1) ;;
  *) echo "LW_CLOUD_FP32 must be 0 or 1, got '$lw_cloud_fp32'" >&2; exit 2;;
esac

art_dir="${ART_DIR:-$HERE/1km_demo_$(date -u +%Y%m%dT%H%M%SZ)}"
jax_cache="${V020_JAX_CACHE:-<DATA_ROOT>/gpuwrf_jax_cache}"
case_timeout="${CASE_TIMEOUT_S:-7200}"
dt_s="${DT_S:-6.0}"
acoustic_substeps="${ACOUSTIC_SUBSTEPS:-10}"

mkdir -p "$art_dir"
json="$art_dir/$tag.json"
log="$art_dir/$tag.log"
vram="$art_dir/$tag.vram.csv"
metric="$art_dir/$tag.metric.json"
rm -f "$json" "$log" "$vram" "$metric"

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[1km] tag=$tag regime=$regime grid=${nx}x${ny}x${nz} dt_s=$dt_s acoustic_substeps=$acoustic_substeps steps=$steps cadence=$rad_cadence segment=$segment_steps tile=$tile_cols lw_cloud_fp32=$lw_cloud_fp32"
echo "[1km] art_dir=$art_dir"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader 2>/dev/null >> "$vram" || true
    sleep 1
  done
) &
sampler_pid=$!

cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT

env_args=(
  -u GPUWRF_RRTMG_LW_CLOUD_OPTICS_FP32
  PYTHONPATH="$ROOT/src"
  OMP_NUM_THREADS=4
  JAX_ENABLE_X64=true
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_ALLOCATOR=platform
  JAX_ENABLE_COMPILATION_CACHE=true
  JAX_COMPILATION_CACHE_DIR="$jax_cache"
  GPUWRF_RRTMG_LW_COLUMN_TILE_COLS="$tile_cols"
  GPUWRF_RRTMG_SW_COLUMN_TILE_COLS="$tile_cols"
)

if [[ "$regime" == "aggressive" ]]; then
  env_args+=(
    GPUWRF_FORCE_FP64=0
    GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020
  )
else
  env_args+=(
    GPUWRF_FORCE_FP64=1
    GPUWRF_ACOUSTIC_PRECISION_MODE=fp64_default
  )
fi

if [[ "$lw_cloud_fp32" == "1" ]]; then
  env_args+=(GPUWRF_RRTMG_LW_CLOUD_OPTICS_FP32=1)
fi

set +e
taskset -c 0-3 timeout "$case_timeout" env "${env_args[@]}" \
  python "$HERE/s4_1km_full_physics_demo.py" \
    --regime "$regime" \
    --nx "$nx" \
    --ny "$ny" \
    --nz "$nz" \
    --dt-s "$dt_s" \
    --acoustic-substeps "$acoustic_substeps" \
    --steps "$steps" \
    --radiation-cadence-steps "$rad_cadence" \
    --segment-steps "$segment_steps" \
    --out "$json" > "$log" 2>&1
rc=$?
set -e

cleanup
trap - EXIT

peak="$(grep -oE '[0-9]+ MiB' "$vram" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1 || true)"
oom="false"
if grep -qiE 'RESOURCE_EXHAUSTED|out of memory|CUDA_ERROR_OUT_OF_MEMORY|OutOfMemory|out-of-memory' "$log" 2>/dev/null; then
  oom="true"
fi

python - "$metric" "$tag" "$rc" "$oom" "${peak:-}" "$started_utc" "$json" "$log" "$vram" <<'PY'
import json
import sys

metric, tag, rc, oom, peak, started, json_path, log_path, vram_path = sys.argv[1:]
payload = {
    "tag": tag,
    "rc": int(rc),
    "oom_log_detected": oom == "true",
    "peak_vram_mib": int(peak) if peak else None,
    "started_utc": started,
    "json": json_path,
    "log": log_path,
    "vram_csv": vram_path,
}
with open(metric, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

tail -n 80 "$log" || true
echo "[1km] metric=$metric rc=$rc oom=$oom peak=${peak:-null}MiB"
exit "$rc"
