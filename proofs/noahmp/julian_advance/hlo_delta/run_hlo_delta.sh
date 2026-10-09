#!/bin/bash
# Julian ON/OFF lowered-HLO delta, PROD + WN3 ordinary step (virtual CUDA lower on CPU; same export path for all arms).
J=<USER_HOME>/wrf_gpu2_lanes/b-thompson/JUL; R=<USER_HOME>/src/wrf_gpu2_wt/b-thompson; X=$J/src_alias
MAIN=$(git -C $R rev-parse main); BR=$(git -C $R rev-parse candidate/b-noahmp-julian)
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
arm() {  # tag rev flag
  rm -rf $J/src_alias; mkdir -p $X && git -C $R archive $2 src data/fixtures data/manifests scripts | tar -x -C $X && ln -s <USER_HOME>/src/wrf_pristine $X/data/wrf_pristine
  for c in prod:<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725:d01,d02: wn3:<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1:d01,d02,d03:--fused; do
    IFS=: read cn cp doms fused <<< "$c"
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
    if [ "$3" = on ]; then export GPUWRF_NOAHMP_JULIAN_ADVANCE=1; else unset GPUWRF_NOAHMP_JULIAN_ADVANCE; fi
    t=$(date +%s); taskset -c 24,25,28,29 nice -n 19 timeout 3000 python3 $J/hlo_delta.py --source $X --case $cp --domains $doms --out $J/hlo/$1_$cn --rev $2 $fused > $J/hlo/$1_$cn.log 2>&1
    echo "$1 $cn rc=$? $(( $(date +%s) - t )) s"
  done
}
mkdir -p $J/hlo
arm main_off $MAIN off
arm br_off $BR off
arm br_on $BR on
rm -rf $J/src_alias
echo ALL_DONE
