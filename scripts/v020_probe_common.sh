#!/usr/bin/env bash
# v020_probe_common.sh — shared helpers for the v0.20.0 GPU probe scripts
# (S0 instrument, G0 fp32-vs-fp64 decision, P4 CFL ladder, skill ladder).
#
# SOURCE this from a probe script; it does NOT run anything itself. It centralises
# the repo root, the canonical inputs, the warm-cache env block, the GPU-lock wrapper
# path, and a couple of tiny portable helpers. Pure shell, no GPU, no side effects on
# source other than setting variables + defining functions.
#
# Collision-avoidance contract (v0.20.0 SPRINT C, worker/opus/v020-probes):
#   * This file + every scripts/v020_* + proofs/v020/instrument/* are NEW FILES.
#   * NOTHING here edits src/gpuwrf or any other worker's area.
#   * Every GPU command in a probe script is wrapped in scripts/with_gpu_lock.sh and
#     is GATED so a CPU dry-run (V020_DRYRUN=1) never touches the GPU.

set -uo pipefail

# --- repo root (this file lives in <root>/scripts) --------------------------------
_V020_THIS="${BASH_SOURCE[0]}"
V020_SCRIPTS_DIR="$(cd "$(dirname "$_V020_THIS")" && pwd)"
V020_ROOT="$(cd "$V020_SCRIPTS_DIR/.." && pwd)"
export V020_ROOT V020_SCRIPTS_DIR

# --- the GPU lock wrapper (MANDATORY for every GPU command) ------------------------
V020_GPU_LOCK="$V020_SCRIPTS_DIR/with_gpu_lock.sh"
export V020_GPU_LOCK

# --- canonical cases --------------------------------------------------------------
# The nested all-7 max_dom=9 benchmark (the stability/skill gate for every change).
: "${V020_ALL7_INPUT:=<DATA_ROOT>/wrf_downscale/canary_all7/run}"
# The DRAM-bound single large grid that fp64 fits (the place where fp32 speed can
# physically exist; ~211k cols, working set >> 96 MiB L2). The plan text says
# "~130-147k cols"; this v0.17 bigswiss grid (460x460x44) is the available large
# fp64-fitting case and is firmly past L2-residency, which is the property G0 needs.
: "${V020_BIG_INPUT:=<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init}"
export V020_ALL7_INPUT V020_BIG_INPUT

# Output root for all probe artifacts (timestamped run dir under it per-probe).
: "${V020_OUT_ROOT:=$V020_ROOT/proofs/v020/instrument}"
export V020_OUT_ROOT

# --- compilation cache (warm runs reuse the persistent XLA cache) -----------------
: "${V020_JAX_CACHE:=<DATA_ROOT>/gpuwrf_jax_cache}"
export V020_JAX_CACHE

# --- core-budget pin (Claude on 0-3; leave 4-31 for CPU WRF) ----------------------
: "${V020_TASKSET:=0-3}"
: "${V020_OMP:=4}"
export V020_TASKSET V020_OMP

# --- dry-run flag: when 1, NO GPU command runs; only non-GPU logic is exercised ----
: "${V020_DRYRUN:=0}"
export V020_DRYRUN

# v020_log MSG... -> timestamped stderr line.
v020_log() { printf '[v020][%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

# v020_mk_rundir PREFIX -> echoes a fresh timestamped run dir under V020_OUT_ROOT.
v020_mk_rundir() {
  local prefix="${1:?need prefix}"
  local rr="$V020_OUT_ROOT/${prefix}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$rr"
  echo "$rr"
}

# v020_env_block -> prints the canonical warm-run env assignments (one per line),
# suitable for `env $(v020_env_block) <cmd>` OR for echoing in a dry-run.
# fp64 (x64=true) is the DEFAULT; callers override JAX_ENABLE_X64 for the fp32 arm.
v020_env_block() {
  cat <<EOF
PYTHONPATH=$V020_ROOT/src
JAX_ENABLE_X64=${JAX_ENABLE_X64:-true}
XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_ENABLE_COMPILATION_CACHE=true
JAX_COMPILATION_CACHE_DIR=$V020_JAX_CACHE
OMP_NUM_THREADS=$V020_OMP
GPUWRF_MYNN_BOULAC_ONZ=1
XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
GPUWRF_NESTED_SYNC_MODE=${GPUWRF_NESTED_SYNC_MODE:-root}
EOF
}

# v020_run_gpu LABEL TIMEOUT -- CMD...
#   In normal mode: runs CMD wrapped in with_gpu_lock.sh (serialised GPU access).
#   In dry-run mode (V020_DRYRUN=1): prints the exact command it WOULD run and
#   returns 0 WITHOUT touching the GPU. This is the single choke-point that makes
#   every probe script CPU-safe-by-construction.
v020_run_gpu() {
  local label="${1:?need label}"; shift
  local timeout="${1:?need timeout}"; shift
  if [[ "$1" != "--" ]]; then
    echo "v020_run_gpu: expected '--' before the command" >&2; return 2
  fi
  shift
  if [[ "$V020_DRYRUN" == "1" ]]; then
    v020_log "DRY-RUN (no GPU): would run under with_gpu_lock --label $label --timeout $timeout --"
    printf '    GPU-CMD>'; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  "$V020_GPU_LOCK" --label "$label" --timeout "$timeout" -- "$@"
}

# v020_assert_case DIR -- fail loudly if a required input case is missing (checked
# even in dry-run so a missing case is caught before the GPU is ever held).
v020_assert_case() {
  local dir="${1:?need dir}"
  if [[ ! -d "$dir" ]]; then
    v020_log "MISSING input case dir: $dir"
    return 1
  fi
  if [[ ! -f "$dir/namelist.input" ]]; then
    v020_log "WARNING: $dir has no namelist.input"
  fi
  return 0
}
