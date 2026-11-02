#!/usr/bin/env bash
# v46-topo TS01: deviceless E41 prescreen (E156) of current main on the warmstart mini24 crop. CPU only.
set -u
L=<USER_HOME>/wrf_gpu2_lanes/v46-topo; W=<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/E41_MINI_ATTRIB
snap=$1; case=$2; dom=$3; tag=$4
out=$L/DL_$tag
ulimit -c 0
while [ -e /tmp/wrf_gpu2_quiet ]; do echo "QUIET wait $(date -u +%T)"; sleep 60; done
echo "start $tag $(date -u +%T)"
env -u XLA_FLAGS JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 \
  GPUWRF_NESTED_PARALLEL_COMPILE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c 12,28 nice -n 19 systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 -- \
  bash -c 'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"' v46-dl \
  timeout 3600 <USER_HOME>/miniconda3/bin/python3 "$L/dl_prod_d01.py" --source "$snap" --case "$case" \
  --target "$L/target_config.pbtxt" --domain "$dom" --out "$out" > "$out.log" 2>&1
rc=$?
echo "end $tag rc=$rc $(date -u +%T)"
