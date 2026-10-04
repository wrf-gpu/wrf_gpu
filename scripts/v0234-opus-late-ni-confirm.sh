#!/usr/bin/env bash
# Bounded, manager-authorized GPU confirmation of the v0234 late-Ni fix.
# One ordinary production d01 dispatch from the retained step-1147 carry, output
# off, health check after. Under the shared GPU lock. See the paired Python arm.
set -euo pipefail

nonce=$1
run_dir=$2
repo=<USER_HOME>/src/wrf_gpu2_wt/v0234-opus-late-ni-fix
lineage=<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/corrected_ni_rca_max_22c2bd7a
runtime_authority=$lineage/v0234_v10_runtime_authority_complete1_5a6298fb
autotune_pin=$lineage/v0234_gpt_v10_replay_c17fca1e201ae106_reference/autotune-results.pb

test -f "$autotune_pin"
test ! -e "$run_dir"
test -z "$(git -C "$repo" status --porcelain)"   # clean, committed tree only
model_commit=$(git -C "$repo" rev-parse HEAD)

if [[ ! "$nonce" =~ ^[0-9a-f]{32,128}$ ]]; then
  echo "nonce is not 32-128 lowercase hex" >&2; exit 2
fi
log=/tmp/v0234-opus-late-ni-confirm-${nonce:0:16}.log
test ! -e "$log"

set +e
"$repo/scripts/with_gpu_lock.sh" \
  --timeout 300 \
  --label v0234-opus-late-ni-fix \
  -- /usr/bin/timeout --signal=TERM --kill-after=60s 1800s \
  /usr/bin/taskset -c 13,14,15,29,30,31 \
  /usr/bin/nice -n 15 /usr/bin/ionice -c 3 \
  /usr/bin/env \
    OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH="$repo:$repo/src" PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=false \
    XLA_FLAGS="--xla_gpu_load_autotune_results_from=$autotune_pin" \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GPUWRF_ALLOCATOR=cuda_async GPUWRF_FINITE_CHECK=1 GPUWRF_BITWISE=1 \
    GPUWRF_BATCH_ENSEMBLE=1 GPUWRF_ADVANCE_CHUNK_LOOP=fori \
    GPUWRF_NESTED_FUSE=0 GPUWRF_NESTED_DEFUSE_COMPILE=0 \
    GPUWRF_NESTED_PARALLEL_COMPILE=0 GPUWRF_NESTED_AOT=0 \
    GPUWRF_AOT_VERIFY=0 GPUWRF_NESTED_ASYNC_OUTPUT=0 \
    GPUWRF_NEST_OUTPUT_PIPELINE=0 GPUWRF_FULL_WRFOUT=0 \
    GPUWRF_TRAINING_OUTPUT_SUBSET=0 GPUWRF_NESTED_SYNC_MODE=root \
    GPUWRF_JAX_CACHE=0 GPUWRF_JAX_CACHE_LOCK=0 \
    GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE=1 \
    GPUWRF_WRF_ROOT="$runtime_authority" \
    GPUWRF_OPUS_LATE_NI_NONCE="$nonce" \
    GPUWRF_OPUS_MODEL_COMMIT="$model_commit" \
    <USER_HOME>/miniconda3/bin/python \
    -m scripts.v0234_opus_late_ni_gpu_confirm \
    --nonce "$nonce" --run-dir "$run_dir" \
    2>&1 | /usr/bin/tee "$log"
rc=${PIPESTATUS[0]}
set -e

if [[ -d "$run_dir" ]]; then /usr/bin/cp -- "$log" "$run_dir/launcher.log"; fi
exit "$rc"
