#!/usr/bin/env bash
# MON33-class explicit scoring for ANY Monica 2-domain arm (candidate reruns, e.g. FINAL33W2 MON33 on 471efc475). Same tools/steps as the
# 03:00Z FINAL MON33 scoring, combined into one parameterized script: binding D6 72 h via d6_domains.py (frozen wn3_score + d6_verdict, single
# asserted domain-line substitution, domains d01,d02) + all-frames integrity (copy wn3_integrity 413299de --all-frames --domains, R4 data via
# r4_eval_domains.py = frozen r4_eval 73deaf41 with one asserted substitution [the 03:13Z bug: plain r4_eval crashed on absent d03], frozen
# integ_classes, annex_postpass 3b03e7cb). D6 core slots via flock. Posts both results.
# Usage: mon33_score.sh LABEL GPU_WRFOUT_DIR [CPU_DIR=RF02 restart48 cpu_combined]   -> V33/mon33/LABEL/
set -uo pipefail
LAB=${1:?}; GW=${2:?}; CW=${3:-<USER_HOME>/wrf_gpu2_lanes/release-docs/V033_G3/reference/RF02/restart48/cpu_combined}
G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; A=$G/integrity_all; T=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/mon33; M=$T/$LAB; MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
mkdir -p $M; PY="nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 <USER_HOME>/miniconda3/bin/python3"
log() { echo "$(date -u +%FT%TZ) $LAB $*" | tee -a $M/run.log; }
source "$G/cpu_slot.sh"
slot() { wn3_slot; }
[[ $(sha256sum $A/annex_postpass.py | cut -c1-8) == 3b03e7cb && $(sha256sum $A/scripts/wn3_integrity.py | cut -c1-8) == 413299de \
   && $(sha256sum $G/r4_eval.py | cut -c1-8) == 73deaf41 ]] || { log "tool sha drift"; exit 3; }
n1=$(ls $GW | grep -c '^wrfout_d01'); n2=$(ls $GW | grep -c '^wrfout_d02'); nt=$(ls $GW | grep -c '\.tmp$')
(( n1 == 73 && n2 == 73 && nt == 0 )) || { log "frames d01 $n1 d02 $n2 tmp $nt — need 73/73/0"; exit 2; }
[[ -e $M/d6_72h_verdict.json ]] && { log "verdict exists, not rescoring"; exit 0; }
slot; log "D6 start cores $CPUS gpu $GW cpu $CW"
wn3_ready
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 7200 taskset -c $CPUS $PY $T/d6_domains.py --domains d01,d02 --cpu-dir $CW --gpu-dir $GW --out $M/d6_72h --hours 72 > $M/d6_72h.log 2>&1; r=$?
d6=$(tail -1 $M/d6_72h.log | cut -c1-900); log "D6 rc=$r $d6"
echo "$(date -u +%FT%TZ) wn3 -> manager, integrate, release-docs (cc mass-opus, review-s2small): $LAB D6 72 h binding [frozen wn3_score + d6_verdict, ONLY change = domain list d01,d02 (d6_domains.py, single asserted substitution); CPU $CW]: $d6 | $M/d6_72h_verdict.json" >> $MB
O=$M/integrity_all; mkdir -p $O; IFS=, read -ra cs <<< "$CPUS"; k=0
for d in d01 d02; do wn3_ready; if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 7200 taskset -c ${cs[$k]} $PY $A/scripts/wn3_integrity.py --cpu-dir $CW --gpu-dir $GW --out $O/$d.json --domains $d --all-frames > $O/$d.log 2>&1 & k=$((k+1)); done; wait
$PY - $O <<'Y'
import json, sys
from pathlib import Path
O = Path(sys.argv[1]); doms = {d: json.load(open(O / f"{d}.json"))["domains"][d] for d in ("d01", "d02")}
f = json.load(open(O / "d01.json")); f["domains"] = doms; f["pass"] = all(v["pass"] for v in doms.values())
(O / "integrity_all_72h.json").write_text(json.dumps(f, indent=1) + "\n")
Y
taskset -c $CPUS $PY $T/r4_eval_domains.py d01,d02 --cpu-dir $CW --gpu-dir $GW --out $O/r4_watch_all_72h.json --integrity $O/integrity_all_72h.json > $O/r4_all.log 2>&1 || log "r4 failed (see $O/r4_all.log)"
$PY - $O <<'Y' > $O/classify.log 2>&1
import json, sys
sys.path.insert(0, "<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate")
from integ_classes import classify, compact
O = sys.argv[1]; v = classify(json.load(open(f"{O}/integrity_all_72h.json")), json.load(open(f"{O}/r4_watch_all_72h.json")))
json.dump(v, open(f"{O}/classes_all_72h.json", "w"), indent=1, default=str); print(compact(v))
Y
taskset -c $CPUS $PY $A/annex_postpass.py $O/classes_all_72h.json $CW $GW > $O/annex_postpass.txt 2>&1; exec 8>&-
an=$(grep -E "^ANNEX (PASS|FAIL)" $O/annex_postpass.txt | tail -1); [[ -n $an ]] || an="ANNEX ERROR (see $O/annex_postpass.txt)"
log "INTEGRITY $an"
echo "$(date -u +%FT%TZ) wn3 -> manager, integrate, release-docs (cc mass-opus, review-s2small): $LAB INTEGRITY all-frames [copy 413299de --all-frames, frozen registry integ_classes + r4_eval_domains every frame, annex 3b03e7cb; d01+d02 × 73 frames]: $an | registry: $(tail -1 $O/classify.log | cut -c1-1200) | $O/classes_all_72h.json" >> $MB

wn3_ready
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 120 taskset -c 8,9 $PY "$G/strict24.py" "$M/d6_72h" "$M/d6_24h_strict_verdict.json" --domains d01,d02 >> "$M/run.log" 2>&1
