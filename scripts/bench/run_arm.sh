#!/usr/bin/env bash
# run_arm.sh — execute ONE benchmark arm while holding the GPU lock (invoked per arm by bench_prod.sh).
# The caller (with_gpu_lock) has already pinned CPUs (taskset) and set GPUWRF_GPU_LOCK_HELD=1.
# * optionally sets $BENCH_QUIET_FILE ("<quiet-label> <UTC>") once the lock is held and removes it on exit (trap);
# * samples the measured process for the whole arm into $BENCH_ARM_DIR: nvidia-smi dmon (pucm) and
#   pidstat -u -t (or a /proc/<pid>/task fallback); both background and killed by the EXIT trap;
# * before returning (=> before the lock is released) drains our GPU memory and reaps children.
# Usage: run_arm.sh <envfile> <src> <tools> <xla-flags> <quiet-label> <arm-timeout-s> -- <command...>
#   <arm-timeout-s> applies `timeout` to the MEASURED command only, INSIDE the lock (excludes queue wait; E20).
set -uo pipefail
ENVFILE=$1; SRC=$2; TOOLS=$3; XLA_FLAGS_REQ=$4; QUIET_LABEL=${5:-}; ARM_TMO=${6:-}; shift 6
[[ "${1:-}" == "--" ]] && shift
source "$ENVFILE"
export PYTHONPATH="$SRC/src" T="$TOOLS" BENCH_SRC_REPO="$SRC"
# s0_env.sh unsets XLA_FLAGS; re-export the requested flags for the measured run (recorded in the receipt env)
[[ -n "$XLA_FLAGS_REQ" ]] && export XLA_FLAGS="$XLA_FLAGS_REQ"
# Candidate flags are frozen with the source snapshot and loaded before Python.
# Load after s0_env.sh, which clears XLA_FLAGS.
if [[ -n "${GPUWRF_BENCH_FLAGS_ENV:-}" ]]; then
  ARM_FLAGS_ENV="$GPUWRF_BENCH_FLAGS_ENV"
  [[ "$ARM_FLAGS_ENV" == /* ]] || ARM_FLAGS_ENV="$SRC/$ARM_FLAGS_ENV"
  [[ -f "$ARM_FLAGS_ENV" ]] || { echo "run_arm: missing flags env: $ARM_FLAGS_ENV" >&2; exit 2; }
  source "$ARM_FLAGS_ENV" || exit 2
fi
s0_guard bench
QUIET_FILE="${BENCH_QUIET_FILE:-/tmp/wrf_gpu2_quiet}"
GUARD="${BENCH_COMPILE_GUARD:-<USER_HOME>/wrf_gpu2_lanes/compile/phase2/compile_only_guard.py}"
GUARD_PY="${BENCH_COMPILE_GUARD_PY:-<USER_HOME>/miniconda3/bin/python}"
GUARD_WAIT="${BENCH_GUARD_WAIT:-60}"
ARM_DIR="${BENCH_ARM_DIR:-}"
DMON_PID=""; SAMP_PID=""; CMD_PID=""
cleanup() {
  # a TERM to this script interrupts `wait`: never leave the measured command running on the GPU without the lock
  [[ -n "$CMD_PID" ]] && kill -0 "$CMD_PID" 2>/dev/null && { kill -TERM "$CMD_PID" 2>/dev/null; sleep 5; kill -KILL "$CMD_PID" 2>/dev/null; } || true
  [[ -n "$QUIET_LABEL" ]] && rm -f "$QUIET_FILE" 2>/dev/null || true
  [[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null || true
  [[ -n "$SAMP_PID" ]] && kill "$SAMP_PID" 2>/dev/null || true
  # hand the compile lane back its launches (best effort; only if we paused it)
  if [[ -n "${QUIET_LABEL:-}" && -f "${GUARD:-}" ]]; then "$GUARD_PY" "$GUARD" resume >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT TERM INT
if [[ -n "$QUIET_LABEL" ]]; then
  if [[ -f "$GUARD" ]]; then
    _t0=$SECONDS; _ack=""
    while (( SECONDS - _t0 < GUARD_WAIT )); do
      _ack=$(timeout 15 "$GUARD_PY" "$GUARD" pause 2>&1) || true
      [[ "$_ack" == *'"pause_ack": true'* ]] && break
      sleep 2
    done
    if [[ "$_ack" != *'"pause_ack": true'* ]]; then
      echo "[bench] REFUSE ($QUIET_LABEL): compile_only_guard pause not acked within ${GUARD_WAIT}s; last output: ${_ack:0:300}" >&2
      exit 7
    fi
    echo "[bench] compile_only_guard pause ack (${QUIET_LABEL}): ${_ack:0:200}" >&2
  else
    echo "[bench] note: compile_only_guard missing at $GUARD; continuing without pause (${QUIET_LABEL})" >&2
  fi
  printf '%s %s\n' "$QUIET_LABEL" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
fi
# run the measured command in the background (timing-neutral) so we can sample its pid; the arm timeout
# applies ONLY here, after the lock is held (queue wait already spent) -- E20.
if [[ -n "$ARM_TMO" && "$ARM_TMO" -gt 0 ]]; then
  timeout --signal=TERM --kill-after=60 "$ARM_TMO" "$@" & CMD_PID=$!
else
  "$@" & CMD_PID=$!
fi
MEAS_PID=""
if [[ "${BENCH_SAMPLERS:-1}" == 1 && -n "$ARM_DIR" ]]; then
  mkdir -p "$ARM_DIR"
  # resolve the measured python pid (the command may be wrapped, e.g. by nsys); bounded search
  for _ in $(seq 1 20); do
    if [[ "$(cat /proc/$CMD_PID/comm 2>/dev/null)" == python* ]]; then MEAS_PID=$CMD_PID; break; fi
    c=$(pgrep -P "$CMD_PID" 2>/dev/null | while read -r x; do
          [[ "$(cat /proc/$x/comm 2>/dev/null)" == python* ]] && { echo "$x"; break; }; done)
    [[ -n "$c" ]] && { MEAS_PID=$c; break; }
    kill -0 "$CMD_PID" 2>/dev/null || break
    sleep 0.5
  done
  printf 'cmd_pid=%s measured_pid=%s pidstat=%s at %s\n' "$CMD_PID" "${MEAS_PID:-none}" \
    "$(command -v pidstat >/dev/null 2>&1 && echo yes || echo no)" "$(date -u +%FT%TZ)" > "$ARM_DIR/samplers.info"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -d 1 > "$ARM_DIR/nvidia-smi.dmon.log" 2>&1 & DMON_PID=$!
  fi
  if [[ -n "$MEAS_PID" ]]; then
    if command -v pidstat >/dev/null 2>&1; then
      pidstat -u -t -p "$MEAS_PID" 5 > "$ARM_DIR/pidstat.log" 2>&1 & SAMP_PID=$!
    else
      ( while kill -0 "$MEAS_PID" 2>/dev/null; do
          printf '%s ' "$(date -u +%FT%TZ)"
          awk '{print "utime_ticks="$14" stime_ticks="$15" threads="$20}' "/proc/$MEAS_PID/stat" 2>/dev/null
          sleep 5
        done ) > "$ARM_DIR/proc_cpu.log" 2>&1 & SAMP_PID=$!
    fi
  fi
fi
wait "$CMD_PID"; rc=$?
# stop samplers (only for the measured process) before draining/reaping, so they are not "children still alive"
[[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null; DMON_PID=""
[[ -n "$SAMP_PID" ]] && kill "$SAMP_PID" 2>/dev/null; SAMP_PID=""
wait 2>/dev/null || true
# bounded drain: do not release the lock while our CUDA context / children linger
MIN_FREE="${GPUWRF_LOCK_MIN_FREE_MIB:-24000}"
if command -v nvidia-smi >/dev/null 2>&1; then
  for _i in $(seq 1 60); do
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [[ -z "$_free" || "$_free" -ge "$MIN_FREE" ]] && break
    (( _i == 1 )) && echo "[bench] arm draining: free ${_free} MiB < ${MIN_FREE}; apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')" >&2
    sleep 2
  done
fi
# reap any remaining direct children (bounded: never hang the lock on a stuck child)
for _i in $(seq 1 15); do pgrep -P $$ >/dev/null 2>&1 || break; sleep 2; done
if pgrep -P $$ >/dev/null 2>&1; then
  echo "[bench] warning: children still alive at arm end: $(pgrep -aP $$ 2>/dev/null | tr '\n' ';')" >&2
else
  wait 2>/dev/null || true
fi
exit "$rc"
