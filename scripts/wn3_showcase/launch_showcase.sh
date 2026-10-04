#!/usr/bin/env bash
# Usage: launch_showcase.sh BASE   (after FINAL freeze; 4 physical GPU-arm cores, alisios L4 grant to 2026-10-04 13:10Z; whole lock <= 7800 s)
set -euo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W7; b=${1:?}
[[ -d $W/source ]] || { echo "run setup.sh FINAL_COMMIT first" >&2; exit 2; }
exec env GPUWRF_GPU_ARM_CPUS=10,11,12,13,26,27,28,29 <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label wn3 --timeout 28800 -- \
  timeout --signal=TERM --kill-after=120 7800 bash $W/showcase.sh "$b"
