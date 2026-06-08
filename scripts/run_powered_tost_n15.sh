#!/usr/bin/env bash
#
# Robust launcher for the powered n=15 TOST campaign.
#
# Foreground:
#   scripts/run_powered_tost_n15.sh --resume
#
# Detached, durable log/rc/runinfo:
#   PYTHON=/home/enric/miniconda3/bin/python scripts/run_powered_tost_n15.sh --detach --resume

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT"

DETACH=0
OUT_DIR="${GPUWRF_TOST_RUN_DIR:-/mnt/data/wrf_gpu_validation/v0130_marathon}"
LOG_NAME="${GPUWRF_TOST_LOG_NAME:-n15_current}"
CORES="${GPUWRF_GPU_CORES:-0-23}"
PY="${PYTHON:-python3}"
FORWARD_ARGS=()
NEEDS_GPU=1

usage() {
  printf '%s\n' \
    "Usage: scripts/run_powered_tost_n15.sh [--detach] [--out-dir DIR] [--log-name NAME] [--cores 0-23] [TOST args...]" \
    "" \
    "Common:" \
    "  scripts/run_powered_tost_n15.sh --resume" \
    "  PYTHON=/home/enric/miniconda3/bin/python scripts/run_powered_tost_n15.sh --detach --resume" \
    "  scripts/run_powered_tost_n15.sh --dry-run --resume" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=1
      shift
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || { echo "missing value for --out-dir" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --log-name)
      [[ $# -ge 2 ]] || { echo "missing value for --log-name" >&2; exit 2; }
      LOG_NAME="$2"
      shift 2
      ;;
    --cores)
      [[ $# -ge 2 ]] || { echo "missing value for --cores" >&2; exit 2; }
      CORES="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --dry-run|--skip-gpu)
      NEEDS_GPU=0
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$OUT_DIR"
  log="$OUT_DIR/${LOG_NAME}.log"
  rc_file="$OUT_DIR/${LOG_NAME}.rc"
  runinfo="$OUT_DIR/${LOG_NAME}.runinfo"
  rm -f "$rc_file"
  {
    printf 'head=%s\n' "$(git rev-parse HEAD)"
    printf 'started=%s\n' "$(date -Is)"
    printf 'launcher=scripts/run_powered_tost_n15.sh --detach\n'
    printf 'log=%s\n' "$log"
    printf 'rc_file=%s\n' "$rc_file"
    printf 'cores=%s\n' "$CORES"
    printf 'args='
    printf '%q ' "${FORWARD_ARGS[@]}"
    printf '\n'
  } > "$runinfo"

  GPUWRF_TOST_RC_FILE="$rc_file" \
  GPUWRF_TOST_RUNINFO="$runinfo" \
  setsid "$SELF" --cores "$CORES" "${FORWARD_ARGS[@]}" > "$log" 2>&1 &
  pid=$!
  printf 'pid=%s\n' "$pid" >> "$runinfo"
  echo "TOST n=15 detached: pid=$pid log=$log rc=$rc_file runinfo=$runinfo"
  exit 0
fi

runner_cmd=(
  env
  GPUWRF_GPU_LOCK_WRAPPER=
  GPUWRF_AEMET_ROOT="${GPUWRF_AEMET_ROOT:-/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations}"
  "$PY"
  proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py
  "${FORWARD_ARGS[@]}"
)

if [[ "$NEEDS_GPU" -eq 1 ]]; then
  cmd=(
    "$ROOT/scripts/run_gpu_lowprio.sh"
    --cores "$CORES"
    --
    "${runner_cmd[@]}"
  )
else
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  cmd=("${runner_cmd[@]}")
fi

if [[ -n "${GPUWRF_TOST_RC_FILE:-}" ]]; then
  set +e
  "${cmd[@]}"
  rc=$?
  printf '%s\n' "$rc" > "$GPUWRF_TOST_RC_FILE"
  if [[ -n "${GPUWRF_TOST_RUNINFO:-}" ]]; then
    {
      printf 'ended=%s\n' "$(date -Is)"
      printf 'rc=%s\n' "$rc"
    } >> "$GPUWRF_TOST_RUNINFO"
  fi
  exit "$rc"
fi

exec "${cmd[@]}"
