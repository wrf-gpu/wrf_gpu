#!/usr/bin/env bash
# Binding D6 72 h (frozen wn3_score via the D6 core slots, verdict file with 'complete') for ONE 0227 arm frame dir (mass-opus 20:50:15Z
# causal arm, disclosed item 4: D6 on ON vs OFF). Usage: d6_arm.sh WRFOUT_DIR LABEL [CASE=20260227_18z_a1]  -> V33/gate/arms0227/LABEL/
set -uo pipefail
W=${1:?}; LAB=${2:?}; CASE=${3:-20260227_18z_a1}; G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; O=$G/arms0227/$LAB; mkdir -p $O
CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_$CASE/run/run
log() { echo "$(date -u +%FT%TZ) $LAB $*" | tee -a $O/d6.log; }
n=$(ls $W | grep -c '^wrfout_d0'); ls $W/*.tmp >/dev/null 2>&1 && { log "tmp files present in $W - not final"; exit 2; }
(( n == 219 )) || { log "expected 219 frames, found $n"; exit 2; }
[[ -e $O/d6_72h_verdict.json ]] && { log "verdict exists, not rescoring"; exit 0; }
source "$G/cpu_slot.sh"
wn3_slot
[[ -d $O/d6_72h ]] && mv $O/d6_72h $O/d6_72h.incomplete.$(date +%s)
log "D6 start cores $D6CPUS frames $W"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 7200 nice -n 19 taskset -c $D6CPUS env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 <USER_HOME>/miniconda3/bin/python3 $G/wn3_score_nolimit.py \
  --cpu-dir $CPU --gpu-dir $W --out $O/d6_72h --hours 72 > $O/d6_72h.log 2>&1; r=$?; exec 8>&-
wn3_ready
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 8,9 nice -n 19 env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 $G/d6_verdict.py $O/d6_72h 72 $O/d6_72h_verdict.json > $O/d6_verdict.log 2>&1
log "D6 rc=$r $(tail -1 $O/d6_verdict.log | cut -c1-300)"

wn3_ready
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 120 taskset -c 8,9 nice -n 19 <USER_HOME>/miniconda3/bin/python3 "$G/strict24.py" "$O/d6_72h" "$O/d6_24h_strict_verdict.json" >> "$O/d6_verdict.log" 2>&1
