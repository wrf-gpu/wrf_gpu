#!/usr/bin/env bash
# v4 (2026-10-07 01:4xZ, manager/mass-opus 01:33Z after a FALSE TRIGGER on a quoted format example): a READY line counts only if the
# message BODY starts with 'ARM <label> frames READY /dir' right after the '<ts> <sender> -> <to>: ' head (sender integrate|mass-sol|b-core|b-diff),
# the path holds no '<', and the dir is COMPLETE (dir/wrfout with 219 wrfout_d0* frames, no *.tmp, compress receipts >= frames when
# present); otherwise it is retried every 30 s without marking anything.
# v3 (2026-10-07 01:3xZ, before FINAL33V): CASE-AWARE — the WN3 case is taken from the frame path (YYYYMMDD_18z_aN[_icpSEED]); D6 and
# integrity are scored against THAT case's CPU-WRF dir; R32 9-cell table only for 0227 (score_arm_0227.sh); no case in the path -> refuse.
# Combined arm scorer (replaces arm_watch.sh [ended 2026-10-07T00Z] + d6_arm_watch.sh): for every NEW MAILBOX line
# 'ARM <label> frames READY <case dir> <snapshot>' posted after start: (1) d6_arm.sh (binding D6, posted), (2) integrity_all/integ_arm.sh
# (all-frames integrity under frozen REGISTRY_ALLFRAMES 2a11c0cb, posted by itself) — both in background on the D6 core slots;
# (3) score_arm_0227.sh (frozen twin manifest + R32 9-cell table, sequential, posted). One run per label (markers). Ends 2026-10-08T00Z
# (GPU lease end) or on arm_watch2.stop.
set -uo pipefail
G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; L=$G/arms0227/arm_watch4.log
n0=$(awk '$1 >= "2026-10-07T09:00:00Z" {print NR-1; exit}' "$MB"); n0=${n0:-0}; end=$(date -d 2026-10-08T00:00:00Z +%s); echo "$(date -u +%FT%TZ) arm_watch4 start (from MAILBOX line $n0)" >> $L
while (( $(date +%s) < end )) && [[ ! -e $G/arm_watch4.stop ]]; do
  tail -n +$((n0+1)) $MB | grep -E '^[^ ]+ (integrate|mass-sol|b-core|b-diff)[^:]* -> [^:]*: ARM [A-Za-z0-9_.+-]+ frames READY /[^ <;,]+' \
    | sed -E 's/^[^ ]+ [^:]* -> [^:]*: (ARM [A-Za-z0-9_.+-]+ frames READY \/[^ <;,]+( \/[^ <;,]+)?).*/\1/' | sort -u | while read -r _ lab _ _ dir snap; do
    [[ -e /tmp/wrf_gpu2_quiet ]] && continue
    w=$dir; [[ -d $dir/wrfout ]] && w=$dir/wrfout
    [[ -d $w ]] || continue
    nf=$(ls $w | grep -c '^wrfout_d0'); (( nf == 219 )) || continue
    ls $w/*.tmp >/dev/null 2>&1 && continue
    if [[ -f $dir/compress_receipt.jsonl ]]; then (( $(wc -l < $dir/compress_receipt.jsonl) >= nf )) || continue; fi
    O=$G/arms0227/$lab; mkdir -p $O
    case=$(grep -oE '20[0-9]{6}_18z_a[0-9]+(_icp[0-9]+)?' <<< "$dir" | tail -1)
    if [[ -z $case ]]; then [[ -e $O/.nocase ]] || { touch $O/.nocase; echo "$(date -u +%FT%TZ) wn3 -> manager, integrate: ARM $lab NOT scored — no WN3 case id (YYYYMMDD_18z_aN) in $dir; repost with the case dir" >> $MB; }; continue; fi
    if [[ ! -e $O/d6_72h_verdict.json && ! -e $O/.d6started ]]; then touch $O/.d6started; echo "$(date -u +%FT%TZ) D6 $lab $w" >> $L
      ( bash $G/d6_arm.sh $w $lab ${case%%_icp*}; v=$O/d6_72h_verdict.json
        if [[ -s $v ]]; then s=$(JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 -c "import json,sys; v=json.load(open(sys.argv[1])); print(v['status'], 'complete', v['complete'], '|', v.get('breaches_compact') or 'no breaches')" $v)
          echo "$(date -u +%FT%TZ) wn3 -> mass-opus, manager, integrate (cc mass-sol): ARM $lab [$case] D6 72 h binding (frozen wn3_score, AND d01-d03): $s | $v" >> $MB
        else echo "$(date -u +%FT%TZ) wn3 -> mass-opus, manager: ARM $lab [$case] D6 FAILED to produce a verdict — $O/d6.log" >> $MB; fi ) >> $L 2>&1 &
    fi
    if [[ ! -e $O/.integstarted ]]; then touch $O/.integstarted; echo "$(date -u +%FT%TZ) INTEG $lab $w" >> $L
      ( bash $G/integrity_all/integ_arm.sh $w $lab $case ) >> $L 2>&1 &
    fi
    if [[ ! -e $O/.scored && -n $snap && -f $dir/receipt.json && $case == 20260227_18z_a1 ]]; then touch $O/.scored; echo "$(date -u +%FT%TZ) R32 $lab $dir $snap" >> $L
      bash $G/score_arm_0227.sh $dir $snap $lab >> $L 2>&1; rc=$?; c=$O/cells.txt
      if [[ $rc == 0 && -s $c ]]; then
        rows=$(grep 'tau48-72' $c | awk -F': ' '{split($1,a," "); printf "%s %s %s; ", a[1], a[2], $2}')
        echo "$(date -u +%FT%TZ) wn3 -> integrate, mass-opus, manager (cc b-phys): 0227 ARM $lab SCORED [M, frozen twin manifest + frozen R32 formal, unchanged]: $(head -1 $c | sed 's/^[^:]*: //' | cut -c1-400) | tau48-72 statistic/limit ARM | RC | P0227 | v032: $rows| table $c" >> $MB
      else echo "$(date -u +%FT%TZ) wn3 -> integrate, manager: 0227 ARM $lab R32 scoring FAILED rc=$rc — $O/score.log" >> $MB; fi
    fi
  done
  sleep 30
done
