#!/usr/bin/env bash
# MPS A/B (+ MPS-vmm arm + launcher smoke), yielding to fid-q2/fid-cloud: wait outside the lock while their wrapper exists,
# re-check inside (rc 75 = yielded -> re-queue). Pin 12,13,28,29; arm env keeps GPUWRF_GPU_ARM_CPUS=10,11,26,27 (key identity).
set -uo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W6; A=$W/mps_ab
source $W/fid_waiting.sh
mkdir -p $A; echo "gated loop start $(date -u +%FT%TZ)" >> $A/queued.txt
while :; do
  while fid_waiting; do sleep 30; done
  env GPUWRF_GPU_ARM_CPUS=12,13,28,29 <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label wn3 --timeout 28800 -- \
    timeout --signal=TERM --kill-after=90 3000 env ROOT=$W GPUWRF_CENSUS=0 GPUWRF_GPU_ARM_CPUS=10,11,26,27 bash $W/mps_gate.sh
  rc=$?
  [[ $rc == 75 ]] || { echo "lock session rc=$rc $(date -u +%T)" >> $A/queued.txt; exit $rc; }
  sleep 30
done
