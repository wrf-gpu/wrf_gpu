#!/usr/bin/env bash
# v0.15 SHIP-GATE 1c — tiered identity on the MERGED code (riming + MP tiling +
# niter=16 default).  Produces two 3-hour dumps from THIS HEAD and gates the
# merged-default-vs-v0.14-equivalent delta against the FROZEN v0.14 manifest.
#
#   BASE = v0.14-equivalent default graph: niter=50, MP tiling OFF
#   CAND = v0.15 merged default graph:     niter=16 (default), MP tiling ON
#
# Run UNDER the GPU lock (this script does NOT take the lock itself):
#   scripts/with_gpu_lock.sh --label v015-shipgates -- timeout 3600 \
#     bash proofs/perf/v015/run_1c_tiered_merged.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

COMMON="taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
  GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo"

echo "=== 1c BASE dump (v0.14-equivalent: niter=50, tiling OFF) ==="
$COMMON GPUWRF_MYNN_COND_NITER=50 GPUWRF_MP_COLUMN_TILING=0 \
  python proofs/perf/v015/probe_ab_identity.py --tag merged_base_v014eq --hours 3 --dump-state

echo "=== 1c CAND dump (v0.15 merged default: niter=16, tiling ON) ==="
$COMMON \
  python proofs/perf/v015/probe_ab_identity.py --tag merged_cand_default --hours 3 --dump-state

echo "=== 1c tiered field gate vs FROZEN v0.14 manifest ==="
taskset -c 0-3 env PYTHONPATH=src python proofs/perf/v015/compare_tiered_identity.py \
  proofs/perf/v015/ab_merged_base_v014eq_state.npz \
  proofs/perf/v015/ab_merged_cand_default_state.npz \
  --hours 3 --out proofs/perf/v015/tiered_gate_merged.json
echo "=== 1c DONE ==="
