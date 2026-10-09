#!/usr/bin/env bash
# Score ONE 0227 72 h GPU arm (manager 2026-10-06T10:48:34Z late-cell bisect; integrate's leave-one-group-out arms) with the FROZEN tools,
# CPU only (SCHED_IDLE, lane cores):
#   1. twin manifest  — frozen wn3_twin_manifest.py (tools_src 3e194296c): attests the arm's receipt/source snapshot/inputs (== server 0227 inputs)
#   2. R32            — alisios' frozen twin scorer (R32-31 wrapper -> R32-01, freeze 89f4b37c), formal mode, NO change
#   3. table          — statistic/limit of the 9 OPEN/EXPLAINED 0227 cells per tau window vs RC / P0227 / v0.3.2 (r32_cells_arms.py)
#   4. D6 (optional, D6=1) — frozen wn3_score via the D6 core slot (d6_slots.conf), verdict file with 'complete'
# ARM_CASE = launcher case dir (receipt.json + wrfout/ + wrfout/proofs/nested_pipeline_run.json); SNAP = the arm's immutable source
# snapshot (git HEAD == receipt source); LABEL = short arm name (output names). Out: V33/gate/arms0227/LABEL/.
# Usage: score_arm_0227.sh ARM_CASE SNAP LABEL
set -uo pipefail
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0
while [[ -e /tmp/wrf_gpu2_quiet ]]; do sleep 10; done
ARM=${1:?}; SNAP=${2:?}; LAB=${3:?}; G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; T=$G/tools_src/scripts; R32=<USER_HOME>/wrf_gpu2_lanes/wn3/r32_copy
O=$G/arms0227/$LAB; mkdir -p $O; CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run
P="chrt -i 0 nice -n 19 taskset -c 8,9 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 <USER_HOME>/miniconda3/bin/python3"
log() { echo "$(date -u +%FT%TZ) $LAB $*" | tee -a $O/score.log; }
[[ -f $ARM/receipt.json && -d $ARM/wrfout ]] || { log "no launcher case dir $ARM"; exit 2; }
log "start arm=$ARM snap=$SNAP"
# launcher --compress replaces frames in place: wait (<= 45 min) until no *.tmp and the compress receipt covers every frame
for i in $(seq 1 90); do
  nf=$(ls $ARM/wrfout | grep -c '^wrfout_d0'); nr=$(wc -l < $ARM/compress_receipt.jsonl 2>/dev/null || echo -1)
  ls $ARM/wrfout/*.tmp >/dev/null 2>&1 || { (( nr < 0 || nr >= nf )) && break; }
  (( i == 1 )) && log "waiting for in-place compression ($nr/$nf receipts)"; sleep 30
done
log "frames $(ls $ARM/wrfout | grep -c '^wrfout_d0'), compress receipts $(wc -l < $ARM/compress_receipt.jsonl 2>/dev/null || echo none)"
$P $T/wn3_twin_manifest.py 20260227_18z $ARM 72 $SNAP $O/twin_manifest_20260227_18z_72h.json --version-tag "wrf_gpu2 arm $LAB" > $O/manifest.log 2>&1; r1=$?
log "twin manifest rc=$r1 $(tail -1 $O/manifest.log | cut -c1-200)"; (( r1 == 0 )) || exit 3
name=GATE_arm_${LAB}_20260227_18z_72h_formal.json
( cd $R32/R32-31 && WN3_PRERUN_MANIFEST=$O/twin_manifest_20260227_18z_72h.json WN3_PRERUN_CPUS=8,9 chrt -i 0 nice -n 19 taskset -c 8,9 \
    env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 r3231_score_twin.py 20260227_18z $name formal ) > $O/r32.log 2>&1; r2=$?
[[ -f $R32/R32-31/$name ]] && cp $R32/R32-31/$name $R32/R32-31/$name.sha256 $O/
log "R32 rc=$r2 $(tail -1 $O/r32.log | cut -c1-200)"
$P $G/r32_cells_arms.py $O/$name > $O/cells.txt 2>&1; log "cells: $(head -1 $O/cells.txt | cut -c1-200)"
if [[ ${D6:-0} == 1 ]]; then
  exec 8>/dev/null
  until while read -r s sc; do [[ $s =~ ^[0-9]+$ && -n $sc ]] || continue; exec 8>$G/.d6slot.$s; flock -n 8 && { D6CPUS=$sc; break 2; }; done < $G/d6_slots.conf; false; do sleep 20; done
  log "D6 start cores $D6CPUS"
  nice -n 19 taskset -c $D6CPUS env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 <USER_HOME>/miniconda3/bin/python3 $G/wn3_score_nolimit.py \
    --cpu-dir $CPU --gpu-dir $ARM/wrfout --out $O/d6_72h --hours 72 > $O/d6_72h.log 2>&1; r3=$?; exec 8>&-
  $P $G/d6_verdict.py $O/d6_72h 72 $O/d6_72h_verdict.json > $O/d6_verdict.log 2>&1; log "D6 rc=$r3 $(tail -1 $O/d6_verdict.log | cut -c1-200)"
fi
log "done"
