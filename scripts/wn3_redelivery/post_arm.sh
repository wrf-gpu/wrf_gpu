#!/usr/bin/env bash
# After a W8 24 h arm (CPU, lane cores, QUIET-respecting): lossless deflate BEFORE any manifest (frames not yet delivered),
# then per case D6 + output_integrity (wn3_score) -> twin manifest (harness path) -> frozen R32-01 formal pre-run (lane copy).
# Usage: post_arm.sh ARM_DIR TAG(e.g. LW12)
set -uo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W8; arm=${1:?}; tag=${2:?}; repo=<USER_HOME>/src/wrf_gpu2_wt/wn3
case "$arm" in <USER_HOME>/wrf_gpu2_lanes/wn3/W8/?*) ;; *) echo "refuse: $arm outside W8" >&2; exit 2;; esac
[[ -f $arm/end.json ]] || { echo "arm not finished" >&2; exit 2; }
P="nice -n 19 taskset -c 8,9,12,13,24,25,28,29 env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3"
until [[ ! -e /tmp/wrf_gpu2_quiet ]]; do sleep 30; done
touch $arm/.cases_done; $P $repo/scripts/compress_wrfout.py $arm --workers 4 --poll 5 | tail -1
for d in $arm/2026*_a1; do
  [[ -d $d/wrfout ]] || continue
  c=$(basename $d); issue=${c%_a1}
  until [[ ! -e /tmp/wrf_gpu2_quiet ]]; do sleep 30; done
  echo "== $c $(date -u +%T)"
  $P $repo/scripts/wn3_score.py --cpu-dir <DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_$c/run/run --gpu-dir $d/wrfout --out $d/d6_24h --hours 24 > $d/d6_24h.log 2>&1
  echo "D6+integrity rc=$?"; tail -1 $d/d6_24h.log | cut -c1-400
  $P $repo/scripts/wn3_twin_manifest.py $issue $d 24 $W/source $d/twin_manifest_${issue}_24h.json --version-tag "wrf_gpu2 $tag" | tail -1
  bash <USER_HOME>/wrf_gpu2_lanes/wn3/W6/r32_prerun.sh $issue $d/twin_manifest_${issue}_24h.json PRERUN_${tag}_${issue}_formal.json formal > $d/r32_prerun_formal.json 2>&1
  tail -1 $d/r32_prerun_formal.json | cut -c1-300
done
echo "post done $(date -u +%T)"
