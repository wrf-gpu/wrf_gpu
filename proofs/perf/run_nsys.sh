#!/usr/bin/env bash
# nsys kernel-level profile of the warmed per-step coupled forecast.
# Produces proofs/perf/nsys_warmed_step.{nsys-rep,sqlite} + a kernel-summary txt.
# GPU is shared with the wind agent: keep MEM_FRACTION low.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=proofs/perf/nsys_warmed_step
export PYTHONPATH=src OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
       XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async

# Profile only the steady region; -c nvtx would need the range named, but we
# capture the whole run and post-filter the WARMED_STEP_REGION NVTX range.
taskset -c 0-3 nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --output="$OUT" \
  python proofs/perf/nsys_step_driver.py --steps 36 --warm-hours 0.05

# Export stats: GPU kernel summary + CUDA API summary (launch overhead) + NVTX.
nsys stats --force-overwrite=true --force-export=true \
  --report cuda_gpu_kern_sum --report cuda_gpu_sum --report cuda_api_sum --report nvtx_sum \
  --format csv --output "${OUT}_stats" "${OUT}.nsys-rep" 2>&1 | tail -5 || true
echo "nsys profile + stats written to ${OUT}*"
