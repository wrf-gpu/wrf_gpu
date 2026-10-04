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
# 126: missing manager run approval (mechanical launch gate, 2026-09-21).
#
# MANAGER RUN APPROVAL GATE (mechanical, ruling 2026-09-21): every GPU run
# needs a manager-written approval file BEFORE the lock is acquired. The
# manager writes `.agent/run_approvals/<LABEL>-<YYYYMMDD>.approval` containing
# `tag:`, `purpose:`, `approved-by: manager` and the UTC timestamp (valid 48h).
# Agents must never write approval files themselves; the principal may bypass
# interactively with GPUWRF_APPROVAL_BYPASS=1.

set -euo pipefail

# SELF-VERIFY against the main-branch pinned digest (ruling 2026-09-21, second
# bypass): a stale worktree copy of this script without the gate must refuse,
# because the gate only protects launches when every launching tree runs THE
# gated script. Fail-closed: if the pin cannot be read, refuse.
SELF_PATH="$(readlink -f "$0")"
PINNED="$(git -C "$(dirname "$SELF_PATH")" show main:scripts/with_gpu_lock.sha256 2>/dev/null | awk '{print $1}')"
if [[ -z "$PINNED" ]]; then
  echo "[with_gpu_lock] REFUSED (fail-closed): cannot read scripts/with_gpu_lock.sha256 from main." >&2
  exit 126
fi
MY_DIGEST="$(sha256sum "$SELF_PATH" | awk '{print $1}')"
if [[ "$MY_DIGEST" != "$PINNED" ]]; then
  echo "[with_gpu_lock] REFUSED: script digest mismatch vs main pin — this is a stale worktree copy without the launch gate. Merge main into this worktree before any GPU launch." >&2
  exit 126
fi

# Canonical machine-level approval dir (mirrors the /tmp lock: shared across
# all worktrees and users on this box).
APPROVAL_DIR="/tmp/wrf_gpu2_run_approvals"
mkdir -p "$APPROVAL_DIR" 2>/dev/null || true
chmod 777 "$APPROVAL_DIR" 2>/dev/null || true
TIMEOUT=7200
LABEL="${USER:-worker}:$$"

require_approval() {
  # Nested inheritance: if an enclosing with_gpu_lock call in this process tree
  # already holds the lock, the authorization was checked THERE. Requiring a
  # second file for the inner label would make wrapped drivers (e.g. run_b4.sh)
  # structurally unlaunchable.
  if [[ "${GPUWRF_GPU_LOCK_HELD:-0}" == "1" ]]; then
    echo "[with_gpu_lock] $LABEL approval inherited from enclosing lock holder (GPUWRF_GPU_LOCK_HELD=1)" >&2
    return 0
  fi
  if [[ "${GPUWRF_APPROVAL_BYPASS:-0}" == "1" ]]; then
    echo "[with_gpu_lock] $LABEL approval BYPASSED via GPUWRF_APPROVAL_BYPASS=1 (principal/emergency only)" >&2
    return 0
  fi
  local base="${LABEL}"
  # Kernel v2 (2026-10-01): multi-use lane lease written by the manager,
  # ${APPROVAL_DIR}/<LABEL>.lease with a line "expires-utc: <ISO8601>". Not consumed.
  local lease="${APPROVAL_DIR}/${base}.lease"
  if [[ -s "$lease" ]]; then
    local exp; exp=$(grep -m1 '^expires-utc:' "$lease" 2>/dev/null | cut -d' ' -f2 || true)
    if [[ -n "$exp" ]] && (( $(date -u -d "$exp" +%s 2>/dev/null || echo 0) > $(date -u +%s) )); then
      echo "[with_gpu_lock] $LABEL lane lease OK until $exp: $lease" >&2
      return 0
    fi
    echo "[with_gpu_lock] $LABEL lane lease expired or malformed: $lease" >&2
  fi
  local today _d
  for _d in "$(date -u +%Y%m%d)" "$(date -u -d yesterday +%Y%m%d 2>/dev/null || true)"; do
    local f="${APPROVAL_DIR}/${base}-${_d}.approval"
    if [[ -s "$f" ]]; then
      local ts now age
      ts=$(grep -m1 '^timestamp-utc:' "$f" 2>/dev/null | cut -d' ' -f2 || true)
      if [[ -n "$ts" ]]; then
        now=$(date -u +%s); age=$(( now - $(date -u -d "$ts" +%s 2>/dev/null || echo 0) ))
        if (( age > 172800 )); then
          echo "[with_gpu_lock] $LABEL approval file exists but is stale (${age}s old, max 48h): $f" >&2
          exit 126
        fi
      fi
      echo "[with_gpu_lock] $LABEL manager approval OK: $f" >&2
      echo "[with_gpu_lock]   $(grep -m1 '^purpose:' "$f" 2>/dev/null || echo '(no purpose line)')" >&2
      # Single-use: consume the file so every run needs a fresh approval
      # (ruling: one approval file PER RUN, not per label per day).
      mkdir -p "$APPROVAL_DIR/.used" 2>/dev/null || true
      mv "$f" "$APPROVAL_DIR/.used/$(basename "$f").$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
      return 0
    fi
  done
  cat >&2 <<EOF
[with_gpu_lock] $LABEL REFUSED: no manager approval file for today.
Required before any GPU run (ruling 2026-09-21, chain violations #3/#4):
  ${APPROVAL_DIR}/${LABEL}-$(date -u +%Y%m%d).approval
with lines: tag: <run-tag> / purpose: <why> / approved-by: manager /
  timestamp-utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Workers must REQUEST the file from the manager, never write it themselves.
Principal emergency bypass: GPUWRF_APPROVAL_BYPASS=1.
EOF
  exit 126
}

LOCK_FILE="/tmp/wrf_gpu2_gpu.lock"
HOLDER_FILE="${LOCK_FILE}.holder"
TIMEOUT=7200
LABEL="${USER:-worker}:$$"
TOKEN="gpuwrf-lock-$$-${RANDOM:-0}-$(date -u +%s%N)"

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

require_approval

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

# Manager priority (2026-10-03): flock is not FIFO. If /tmp/wrf_gpu2_gpu.next names another label that is
# currently waiting (file younger than 1 h), yield the lock to it; the named label removes the file on acquire.
NEXT_FILE="/tmp/wrf_gpu2_gpu.next"
_deadline=$(( SECONDS + TIMEOUT ))
while :; do
  _rem=$(( _deadline - SECONDS ))
  if (( _rem <= 0 )) || ! flock -w "$_rem" 9; then
    echo "[with_gpu_lock] $LABEL TIMED OUT after ${TIMEOUT}s; holder: $(cat "$HOLDER_FILE" 2>/dev/null)" >&2
    exit 124
  fi
  _next="$(awk 'NR==1{print $1}' "$NEXT_FILE" 2>/dev/null || true)"
  if [[ -n "$_next" && "$_next" != "$LABEL" ]] \
     && (( $(date +%s) - $(stat -c %Y "$NEXT_FILE" 2>/dev/null || echo 0) < 3600 )) \
     && pgrep -f -- "with_gpu_lock.sh.* --label ${_next}( |$)" >/dev/null 2>&1; then
    flock -u 9; sleep 5; continue
  fi
  [[ "$_next" == "$LABEL" ]] && rm -f "$NEXT_FILE" 2>/dev/null
  break
done

# Kernel v2 (2026-10-02): a previous holder's CUDA context can outlive its lock release.
# Wait (bounded) until the GPU is drained before starting, and name the leftover processes.
MIN_FREE="${GPUWRF_LOCK_MIN_FREE_MIB:-24000}"
if command -v nvidia-smi >/dev/null 2>&1; then
  for _i in $(seq 1 60); do
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [[ -z "$_free" || "$_free" -ge "$MIN_FREE" ]] && break
    (( _i == 1 )) && echo "[with_gpu_lock] $LABEL waiting for GPU drain: free ${_free} MiB < ${MIN_FREE}; apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')" >&2
    sleep 2
  done
fi
printf 'holder=%s pid=%s since=%s token=%s cmd=%s\n' "$LABEL" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TOKEN" "$*" > "$HOLDER_FILE" 2>/dev/null || true
echo "[with_gpu_lock] $LABEL ACQUIRED GPU lock -> running: $*" >&2
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[with_gpu_lock] $LABEL GPU memory at acquire: $(nvidia-smi --query-gpu=index,name,memory.free,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" >&2 || true
fi

export GPUWRF_GPU_LOCK_HELD=1
export GPUWRF_GPU_LOCK_FD=9
export GPUWRF_GPU_LOCK_FILE="$LOCK_FILE"
export GPUWRF_GPU_LOCK_HOLDER_FILE="$HOLDER_FILE"
export GPUWRF_GPU_LOCK_LABEL="$LABEL"
export GPUWRF_GPU_LOCK_TOKEN="$TOKEN"

# Lock holder runs on the exclusive GPU-arm cores (manager 2026-10-02 13:17Z: +4 cores; one GPU arm at a time).
# Override with GPUWRF_GPU_ARM_CPUS="" to keep the caller's affinity.
GPU_ARM_CPUS="${GPUWRF_GPU_ARM_CPUS-10,11,26,27}"
set +e
# The lock must outlive the GPU child (2026-10-03, dispatch D04): run the child in the background, forward
# TERM/INT/HUP to it and only release the lock after it has really exited (a child that closed fd 9 or is
# inside a compile would otherwise keep the GPU while the next holder starts).
if [[ -n "$GPU_ARM_CPUS" ]] && command -v taskset >/dev/null 2>&1; then
  echo "[with_gpu_lock] $LABEL pinned to GPU-arm cores $GPU_ARM_CPUS" >&2
  taskset -c "$GPU_ARM_CPUS" "$@" &
else
  "$@" &
fi
child=$!
trap 'echo "[with_gpu_lock] $LABEL forwarding signal to child $child; waiting for its exit" >&2; kill -TERM "$child" 2>/dev/null' TERM INT HUP
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
trap - TERM INT HUP
set -e

: > "$HOLDER_FILE" 2>/dev/null || true
echo "[with_gpu_lock] $LABEL released GPU lock (rc=$rc)" >&2
exit "$rc"
