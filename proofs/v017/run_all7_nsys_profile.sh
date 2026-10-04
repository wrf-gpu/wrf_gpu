#!/usr/bin/env bash
# v0.17 all-7 GPU-COMPUTE roofline profile: where does the ~637 s/forecast-hour go,
# and is it occupancy-bound (tiny nests underfill) or memory-bandwidth-bound (fp32
# would help) or compute-bound?  Eager root mode (clear per-domain/per-kernel
# separation), warm cache.  Bounded capture (~duration) so the GPU hold is short.
# Wrap in scripts/with_gpu_lock.sh --label opus-hostgap.
set -uo pipefail
ROOT=<USER_HOME>/src/wrf_gpu2/.wt-opus-hostgap
INPUT=<DATA_ROOT>/wrf_downscale/canary_all7/run
RR="${1:?need RR}"
mkdir -p "$RR/gpu_output" "$RR/proofs" "$RR/scratch"
cd "$ROOT"
# --duration bounds the whole collection: ~7 min init + ~3 min steady forecast.
# --gpu-metrics-devices=all -> SM occupancy + DRAM read/write throughput (roofline).
env PYTHONPATH="$ROOT/src" JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache \
    OMP_NUM_THREADS=4 GPUWRF_MYNN_BOULAC_ONZ=1 GPUWRF_SCRATCH="$RR/scratch" \
    GPUWRF_NESTED_SYNC_MODE=root XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none --cpuctxsw=none \
      --cuda-memory-usage=false \
      --duration=660 \
      --force-overwrite=true \
      --output="$RR/nsys_all7" \
    taskset -c 0-3 python -m gpuwrf run \
      --input-dir "$INPUT" --namelist "$INPUT/namelist.input" \
      --output-dir "$RR/gpu_output" --proof-dir "$RR/proofs" --scratch-dir "$RR/scratch" \
      --max-dom 9 --hours 1 > "$RR/nsys.log" 2>&1
echo "nsys rc=$? ; report=$RR/nsys_all7.nsys-rep"
# Per-kernel GPU time + GPU metrics summary.
nsys stats --force-overwrite=true --force-export=true \
  --report cuda_gpu_kern_sum --report cuda_gpu_sum --report cuda_gpu_mem_time_sum \
  --format csv --output "$RR/nsys_stats" "$RR/nsys_all7.nsys-rep" > "$RR/nsys_stats.log" 2>&1 || true
echo "=== top GPU kernels by total time ==="
ls "$RR"/nsys_stats*cuda_gpu_kern_sum* 2>/dev/null | head -1 | xargs -r head -25
