#!/usr/bin/env bash
# All-frames integrity (frozen REGISTRY_ALLFRAMES.md 2a11c0cb) for ONE 0227 arm: holds a D6 core slot (flock), runs the integrity COPY with
# --all-frames per domain in parallel (one core each), then classify_all.sh (r4_eval on every frame + integ_classes), then the ANNEX
# post-pass (annex_postpass.py 3b03e7cb). Posts ANNEX verdict + registry line.
# Usage: integ_arm.sh WRFOUT_DIR LABEL [CASE=20260227_18z_a1]  -> V33/gate/arms0227/LABEL/integrity_all/
set -uo pipefail
W=${1:?}; LAB=${2:?}; CASE=${3:-20260227_18z_a1}; G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; A=$G/integrity_all; O=$G/arms0227/$LAB/integrity_all
MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_${CASE%%_icp*}/run/run; mkdir -p $O
[[ $(sha256sum $A/REGISTRY_ALLFRAMES.md | cut -c1-8) == 2a11c0cb ]] || { echo "registry sha drift"; exit 3; }
source "$G/cpu_slot.sh"
wn3_slot
IFS=, read -ra cs <<< "$CPUS"; k=0
for d in d01 d02 d03; do
  wn3_ready
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 7200 taskset -c ${cs[$((k % ${#cs[@]}))]} nice -n 19 env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 $A/scripts/wn3_integrity.py --cpu-dir $CPU --gpu-dir $W \
    --out $O/$d.json --domains $d --all-frames > $O/$d.log 2>&1 & k=$((k + 1))
done; wait
wn3_ready
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 7200 taskset -c $CPUS bash $A/classify_all.sh $CASE $W $O > $O/classify.log 2>&1
# ANNEX post-pass (mass-opus reference annex_postpass.py 3b03e7cb, GATES_FROZEN_FINAL33V; reproduced 4/4 item-level 2026-10-07 01:3xZ)
[[ $(sha256sum $A/annex_postpass.py | cut -c1-8) == 3b03e7cb ]] || { echo "annex sha drift"; exit 3; }
wn3_ready
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 7200 taskset -c $CPUS nice -n 19 env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 $A/annex_postpass.py $O/classes_all_72h.json $CPU $W > $O/annex_postpass.txt 2>&1; exec 8>&-
an=$(grep -E "^ANNEX (PASS|FAIL)" $O/annex_postpass.txt | tail -1); [[ -n $an ]] || an="ANNEX ERROR (see $O/annex_postpass.txt)"
v="$an | registry: $(tail -1 $O/classify.log | cut -c1-1500)"
echo "$(date -u +%FT%TZ) wn3 -> integrate, mass-opus, manager: ARM $LAB [$CASE] INTEGRITY all-frames [frozen REGISTRY_ALLFRAMES 2a11c0cb, copy wn3_integrity 413299de, all 73 frames/domain]: $v | $O/classes_all_72h.json" >> $MB
