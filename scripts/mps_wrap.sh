#!/usr/bin/env bash
# SI41: run <command> under a PRIVATE CUDA MPS daemon (N-parallel cases share SMs instead of time-slicing).
#
# Must run INSIDE scripts/with_gpu_lock.sh: the daemon is started and stopped within the lock. Pipe/log dirs are
# private (<mps_dir>/pipe, <mps_dir>/log): only processes inheriting this CUDA_MPS_PIPE_DIRECTORY connect, so every
# other CUDA process on the box (other lanes, desktop) keeps its normal context. Compute mode stays Default, no root.
# The daemon is stopped on EVERY exit path (normal, command failure, TERM/INT/HUP from timeout or the user).
#
# Usage: mps_wrap.sh <mps_dir> -- <command> [args...]      exit code = the command's exit code
#        (3 = refused outside the GPU lock, 4 = daemon start failed)
set -uo pipefail
D=${1:?usage: mps_wrap.sh <mps_dir> -- <command> [args...]}; shift; [[ ${1:-} == -- ]] && shift
(( $# > 0 )) || { echo "mps_wrap: no command" >&2; exit 2; }
[[ ${GPUWRF_GPU_LOCK_HELD:-} == 1 ]] || { echo "mps_wrap: refuse, not inside with_gpu_lock" >&2; exit 3; }
mkdir -p "$D/pipe" "$D/log"
export CUDA_MPS_PIPE_DIRECTORY=$D/pipe CUDA_MPS_LOG_DIRECTORY=$D/log
LOG=$D/mps_wrap.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

nvidia-cuda-mps-control -d || { log "daemon start FAILED"; echo "mps_wrap: daemon start failed" >&2; exit 4; }
log "daemon up pipe=$D/pipe"

child="" sampler=""
# Our daemon/server only: same uid AND this exact pipe dir in the process environment.
mps_pids() {
  local p
  for p in $(pgrep -u "$(id -u)" -f 'nvidia-cuda-mps-(control|server)'); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qxF "CUDA_MPS_PIPE_DIRECTORY=$D/pipe" && echo "$p"
  done
}
cleanup() {
  [[ -n $sampler ]] && { pkill -P "$sampler" 2>/dev/null; kill "$sampler" 2>/dev/null; }
  { echo get_server_list | nvidia-cuda-mps-control; } >> "$LOG" 2>&1
  { echo quit | nvidia-cuda-mps-control; } >> "$LOG" 2>&1
  local left=""
  for _ in $(seq 1 "${MPS_WRAP_QUIT_WAIT_S:-60}"); do left=$(mps_pids); [[ -z $left ]] && break; sleep 1; done
  if [[ -n $left ]]; then log "daemon still up after quit: TERM $left"; kill $left 2>/dev/null; fi
  log "daemon down"
}
forward() { log "signal $1 -> child $child"; [[ -n $child ]] && kill "-$1" "$child" 2>/dev/null; }
trap cleanup EXIT
trap 'forward TERM' TERM
trap 'forward INT' INT
trap 'forward HUP' HUP

# One snapshot of the MPS clients once the cases are stepping (evidence that they really share the server).
( sleep 120; { log "clients:"; echo get_server_list | nvidia-cuda-mps-control;
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader; } >> "$LOG" 2>&1 ) &
sampler=$!

"$@" &
child=$!
rc=0
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break   # a trapped signal interrupts wait; keep waiting for the child itself
done
log "command exit rc=$rc"
exit "$rc"
