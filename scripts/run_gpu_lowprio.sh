#!/usr/bin/env bash
#
# run_gpu_lowprio.sh -- versioned GPU mutex + low-priority launcher.
#
# Use this instead of any /tmp/wrf_gpu_run_lowprio.sh helper. It serializes GPU
# jobs with flock, sets the standard JAX fp64/no-prealloc environment, pins CPU
# helper work, and keeps the command line visible in ps/logs.
#
# Usage:
#   scripts/run_gpu_lowprio.sh [--cores 0-23] [--lock /tmp/wrf_gpu_validation_gpu.lock] -- <command> [args...]
#
# Exit 75 means another GPU job already owns the lock.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORES="${GPUWRF_GPU_CORES:-0-23}"
LOCK_PATH="${GPUWRF_GPU_LOCK:-/tmp/wrf_gpu_validation_gpu.lock}"
NICE_LEVEL="${GPUWRF_GPU_NICE:-10}"
IONICE_CLASS="${GPUWRF_GPU_IONICE_CLASS:-2}"
IONICE_LEVEL="${GPUWRF_GPU_IONICE_LEVEL:-7}"
USE_FLOCK=1

usage() {
  printf '%s\n' \
    "Usage: scripts/run_gpu_lowprio.sh [--cores 0-23] [--lock PATH] [--nice N] -- <command> [args...]" \
    "" \
    "Sets PYTHONPATH=src, JAX_ENABLE_X64=true, XLA_PYTHON_CLIENT_PREALLOCATE=false." \
    "Holds the GPU flock by default; exit 75 means the GPU lock is busy." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cores)
      [[ $# -ge 2 ]] || { echo "missing value for --cores" >&2; exit 2; }
      CORES="$2"
      shift 2
      ;;
    --lock)
      [[ $# -ge 2 ]] || { echo "missing value for --lock" >&2; exit 2; }
      LOCK_PATH="$2"
      shift 2
      ;;
    --nice)
      [[ $# -ge 2 ]] || { echo "missing value for --nice" >&2; exit 2; }
      NICE_LEVEL="$2"
      shift 2
      ;;
    --ionice-class)
      [[ $# -ge 2 ]] || { echo "missing value for --ionice-class" >&2; exit 2; }
      IONICE_CLASS="$2"
      shift 2
      ;;
    --ionice-level)
      [[ $# -ge 2 ]] || { echo "missing value for --ionice-level" >&2; exit 2; }
      IONICE_LEVEL="$2"
      shift 2
      ;;
    --no-flock)
      USE_FLOCK=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

cd "$ROOT"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

if [[ "$USE_FLOCK" -eq 1 ]]; then
  mkdir -p "$(dirname "$LOCK_PATH")"
  exec 9>"$LOCK_PATH"
  if ! flock -n 9; then
    echo "GPU lock busy: $LOCK_PATH" >&2
    exit 75
  fi
fi

cmd=("$@")

if command -v taskset >/dev/null 2>&1; then
  cmd=(taskset -c "$CORES" "${cmd[@]}")
fi
if command -v ionice >/dev/null 2>&1; then
  cmd=(ionice -c "$IONICE_CLASS" -n "$IONICE_LEVEL" "${cmd[@]}")
fi
if command -v nice >/dev/null 2>&1; then
  cmd=(nice -n "$NICE_LEVEL" "${cmd[@]}")
fi

echo "[gpu-lowprio] root=$ROOT lock=${LOCK_PATH:-none} cores=$CORES"
echo "[gpu-lowprio] JAX_ENABLE_X64=$JAX_ENABLE_X64 XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "[gpu-lowprio] exec: ${cmd[*]}"
exec "${cmd[@]}"
