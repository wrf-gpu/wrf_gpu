#!/usr/bin/env bash
# R6 night sequence: runs AFTER the G1 mono probe exits. Sequential GPU probes,
# each individually wrapped in with_gpu_lock.sh. Stops on any probe failure so
# the worker can review before further GPU spend. Every artifact under
# <DATA_ROOT>/wrf_gpu2/v025/r6/<ts>/.
set -u
PY=<USER_HOME>/miniconda3/bin/python3
ROOT=<USER_HOME>/src/wrf_gpu2_wt/r6-compile-recovery
OUT=<DATA_ROOT>/wrf_gpu2/v025/r6/20260917T2350
G1_PARENT_PID="$1"
SEGS="${2:-34 17 12 8 4}"
cd "$ROOT"

echo "[seq] waiting for G1 parent pid $G1_PARENT_PID ..."
while [ -n "$G1_PARENT_PID" ] && kill -0 "$G1_PARENT_PID" 2>/dev/null; do sleep 20; done
echo "[seq] G1 done at $(date -Is); G1 rc Marker: $(grep -o 'R6_PROBE_SUMMARY.*' "$OUT/g1-mono-cold.launch.log" | tail -1)"

run() { # label runs the probe; abort sequence on failure; skip already-DONE runs (resume)
  local run_id="$1"; shift
  if grep -qs '"status": "DONE"' "$OUT/$run_id/probe.json" 2>/dev/null; then
    echo "[seq] SKIP $run_id (already DONE)"; return 0
  fi
  if [ -e "$OUT/$run_id" ]; then
    echo "[seq] SKIP $run_id (dir exists without DONE - inspect)"; return 0
  fi
  echo "[seq] START $run_id at $(date -Is)"
  scripts/with_gpu_lock.sh --label r6-compile-recovery --timeout 7200 -- \
    "$PY" scripts/v025/r6_compile_probe.py run "$@" --run-id "$run_id" --out-root "$OUT" \
    > "$OUT/${run_id}.launch.log" 2>&1
  local rc=$?
  echo "[seq] DONE  $run_id rc=$rc at $(date -Is)"
  if [ $rc -ne 0 ]; then
    echo "[seq] ABORT: $run_id failed rc=$rc; remaining stages skipped" >&2
    exit $rc
  fi
}

# --- G2 sweep: cold compile + 3 warm executions per segment size (fresh process+cache each)
for SEG in $SEGS; do
  run "g2-seg${SEG}-cold" --entry segmented --segment-steps "$SEG" --invoke 4 --save-final --hlo-text --wall-cap 3600
done

# --- informational: single-scan entry (compile-bounded, bitwise-vs-seg candidate)
run "info-single-scan-cold" --entry single --invoke 4 --save-final --hlo-text --wall-cap 3600

# --- informational: mono cached-load + 3 warm from G1's populated cache
G1_CACHE="$OUT/g1-mono-cold/cache"
run "info-mono-cachedload" --entry mono --invoke 4 --reuse-cache "$G1_CACHE" --wall-cap 1800

# --- G3: cached load per swept size, fresh process, cache just populated by G2
for SEG in $SEGS; do
  run "g3-seg${SEG}-cachedload" --entry segmented --segment-steps "$SEG" --invoke 1 \
    --reuse-cache "$OUT/g2-seg${SEG}-cold/cache" --wall-cap 900
done

echo "[seq] ALL STAGES COMPLETE at $(date -Is)"
