#!/bin/bash
# usage: run_arm.sh ROOT ACTION [extra args]   (legacy CPU arm, core 8, nice 19)
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
root=$1; action=$2; shift 2
mkdir -p "$root"
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/o1-restart/src \
  GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
  CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false
export GPUWRF_FAST_DEFAULTS=${GPUWRF_FAST_DEFAULTS:-0}
timeout 2400 taskset -c 8 nice -n 19 python <USER_HOME>/wrf_gpu2_lanes/o1-restart/tools/cpu_swiss_restart.py \
  --action "$action" --root "$root" "$@" > "$root/$action.log" 2>&1
echo "rc=$? action=$action" >> "$root/$action.log"
