#!/bin/bash
for lane in sas nssl; do
  if [ $lane = sas ]; then sch=cu4_sas; else sch=mp18_nssl; fi
  D=<USER_HOME>/wrf_gpu2_lanes/o1-gpusmoke/snap_$lane; cd $D
  PYTHONPATH=$D/src python scripts/gpu_smoke_v034.py --out <USER_HOME>/wrf_gpu2_lanes/o1-gpusmoke/run_$lane --deadline 18:57 --steps 3 --per-scheme-s 600 --min-slot-s 330 --schemes $sch --assume-available
  echo "$(date -u +%T) $lane done rc=$?"
done
