#!/usr/bin/env bash
# with_gpu_lock.sh — serialize GPU access across parallel wrf_gpu2 workers.
#
# The workstation has ONE GPU. When several agents explore hypotheses in
# parallel, each must run its GPU gates (stage compare, short forecast, etc.)
# one at a time or they collide / OOM. Wrap every GPU command with this script:
# it blocks until the shared GPU lock is free, runs the command holding the
# lock, then releases it (released automatically even if the command crashes,
# because the lock lives on an open fd that closes on exit).
#
# Light CPU-side work (reading dumps, building comparators, writing code) does
# NOT need the lock and should run in parallel. Pin CPU-heavy analysis to a few
# cores (e.g. taskset -c 0-3, OMP_NUM_THREADS=4) so parallel workers do not
# saturate the box.
#
# Usage:
#   scripts/with_gpu_lock.sh [--timeout SECONDS] [--label NAME] -- <command> [args...]
#
# Defaults: timeout 7200s (2h), label "<user>:<pid>".
# Exit codes: the wrapped command's rc; 2 for usage error; 124 on lock timeout.

set -euo pipefail

LOCK_FILE="/tmp/wrf_gpu2_gpu.lock"
HOLDER_FILE="${LOCK_FILE}.holder"
TIMEOUT=7200
LABEL="${USER:-worker}:$$"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2;;
    --label)   LABEL="$2"; shift 2;;
    --)        shift; break;;
    *) echo "with_gpu_lock: unknown arg '$1' (did you forget '--' before the command?)" >&2; exit 2;;
  esac
done
if [[ $# -eq 0 ]]; then
  echo "with_gpu_lock: no command given after '--'" >&2
  exit 2
fi

# Shared, world-writable lock file so any worker (any worktree) can flock it.
if [[ ! -e "$LOCK_FILE" ]]; then
  touch "$LOCK_FILE" 2>/dev/null || true
  chmod 666 "$LOCK_FILE" 2>/dev/null || true
fi

# Append-open (never truncates) the fd we flock; flock releases when fd 9 closes.
exec 9>>"$LOCK_FILE"

if [[ -s "$HOLDER_FILE" ]]; then
  echo "[with_gpu_lock] $LABEL waiting for GPU lock; current holder: $(cat "$HOLDER_FILE" 2>/dev/null)" >&2
else
  echo "[with_gpu_lock] $LABEL waiting for GPU lock ($LOCK_FILE, timeout ${TIMEOUT}s)..." >&2
fi

if ! flock -w "$TIMEOUT" 9; then
  echo "[with_gpu_lock] $LABEL TIMED OUT after ${TIMEOUT}s; holder: $(cat "$HOLDER_FILE" 2>/dev/null)" >&2
  exit 124
fi

printf 'holder=%s pid=%s since=%s cmd=%s\n' "$LABEL" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$HOLDER_FILE" 2>/dev/null || true
echo "[with_gpu_lock] $LABEL ACQUIRED GPU lock -> running: $*" >&2

set +e
"$@"
rc=$?
set -e

: > "$HOLDER_FILE" 2>/dev/null || true
echo "[with_gpu_lock] $LABEL released GPU lock (rc=$rc)" >&2
exit "$rc"
