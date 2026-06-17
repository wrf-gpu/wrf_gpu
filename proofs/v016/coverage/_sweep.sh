#!/usr/bin/env bash
# v0.16 coverage sweep: remaining L2 targets, sequential, GPU-lock-serialized, idempotent.
# Detached (setsid) so it survives a manager crash and shares the GPU fairly with the fp32 lane.
cd /home/user/src/wrf_gpu2/.wt-v016-coverage || exit 1
LOG=proofs/v016/coverage/_sweep.log
FOOT="Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
# cu6 is already running independently -> excluded. 5 done (mp10,pbl1,lw1,sw1,cu3) -> auto-skipped if present.
TARGETS=(mp:1 mp:2 mp:3 mp:4 mp:6 mp:14 mp:16 pbl:2 pbl:7 pbl:8 pbl:99 sfclay:1 sfclay:2 sfclay:3 sfclay:7 sfclay:91 cu:2 lsm:2 sw:2)
echo "$(date -u +%FT%TZ) SWEEP START (${#TARGETS[@]} targets)" >>"$LOG"
for t in "${TARGETS[@]}"; do
  fam=${t%%:*}; opt=${t##*:}
  vf="proofs/v016/coverage/${fam}${opt}_gate.json"
  if [ -f "$vf" ]; then echo "$(date -u +%H:%M:%S) skip ${fam}${opt} (exists)" >>"$LOG"; continue; fi
  echo "$(date -u +%H:%M:%S) START ${fam}${opt}" >>"$LOG"
  bash scripts/with_gpu_lock.sh --label v016-sweep --timeout 28800 -- \
    env PYTHONPATH=src python proofs/v016/coupled_coverage_gate.py --family "$fam" --option "$opt" --hours 1 >>"$LOG" 2>&1
  rc=$?
  echo "$(date -u +%H:%M:%S) END ${fam}${opt} rc=$rc" >>"$LOG"
  git add -A proofs/v016/coverage >/dev/null 2>&1
  git commit -q -m "v016 coverage verdict: ${fam}${opt} (rc=$rc)

$FOOT" >/dev/null 2>&1
done
echo "$(date -u +%FT%TZ) SWEEP COMPLETE" >>"$LOG"
