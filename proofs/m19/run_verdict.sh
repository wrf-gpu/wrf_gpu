#!/usr/bin/env bash
# M19 verdict launcher -- execution mechanics only (no scoring touched).
# Runs the 3-case fail-fast verdict with a GPU-memory-fraction retry ladder so a
# 72h segment OOM retries at a lower fraction before giving up (principal spec).
# All progress is line-buffered to the log so a >600s-silent watchdog kill loses
# nothing already written.
set -u
cd "$(dirname "$0")/../.." || exit 2

OUT=proofs/m19/verdict_result.json
LOG=proofs/m19/verdict_run.log
LEADS="24 48 72"
COMMON_ENV="PYTHONPATH=src OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1"

: > "$LOG"
echo "=== M19 verdict launch $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "branch: $(git branch --show-current)  head: $(git log --oneline -1)" | tee -a "$LOG"

rc=99
for MF in 0.9 0.8 0.7; do
  echo "--- attempt mem_fraction=$MF $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  env $COMMON_ENV XLA_PYTHON_CLIENT_MEM_FRACTION=$MF \
    taskset -c 0-3 \
    python -u proofs/m19/verdict_3case.py --execute --leads $LEADS --out "$OUT" \
    >> "$LOG" 2>&1
  rc=$?
  echo "--- attempt mem_fraction=$MF exit=$rc $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  # rc 0 = PASS verdict written; rc 1 = FAIL verdict written (also a complete run,
  # NOT an OOM) -> stop. Only a hard crash (rc>1, e.g. OOM/segfault) retries lower.
  if [ "$rc" -le 1 ] && [ -f "$OUT" ]; then
    echo "=== completed run (exit=$rc), verdict written -> $OUT ===" | tee -a "$LOG"
    break
  fi
  echo "=== attempt mem_fraction=$MF failed hard (exit=$rc); retrying lower ===" | tee -a "$LOG"
done

echo "=== M19 verdict launcher done exit=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
exit $rc
