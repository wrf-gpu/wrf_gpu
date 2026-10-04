#!/usr/bin/env bash
# Flag A/B driver — run under one GPU lock; one fresh process per arm.
set -uo pipefail
PY=<USER_HOME>/miniconda3/bin/python3
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=<USER_HOME>/src/wrf_gpu2/.agent/sprints/2026-09-18-v0250-host-forensics/artifacts
mkdir -p "$OUT"
export GPUWRF_WRF_ROOT=<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_per_fusion_autotune_cache_dir=<USER_HOME>/.cache/gpuwrf/autotune --xla_gpu_force_compilation_parallelism=8"

run_arm() {
  local name="$1"; shift
  local xla_append="$1"; shift
  echo "=== ARM $name (flags: '${xla_append}') ==="
  if [[ -n "$xla_append" ]]; then
    XLA_FLAGS="$XLA_FLAGS $xla_append" $PY "$HERE/hf_flagab.py" --arm "$name" --flag="$xla_append" --out "$OUT/flagab_${name}.json"
  else
    $PY "$HERE/hf_flagab.py" --arm "$name" --flag= --out "$OUT/flagab_${name}.json"
  fi
}

run_arm base ""
run_arm cb_while "--xla_gpu_enable_command_buffer=while"
run_arm cb_cond "--xla_gpu_enable_command_buffer=conditional"
run_arm cb_while_cond "--xla_gpu_enable_command_buffer=while,conditional"
run_arm cb_full "--xla_gpu_enable_command_buffer=while,conditional,cublas,cudnn,custom_call"
run_arm dblbuf "--xla_gpu_enable_while_loop_double_buffering=true"
run_arm cbwc_unroll "--xla_gpu_enable_command_buffer=while,conditional --xla_gpu_command_buffer_unroll_loops=true"
echo "ALL_ARMS_DONE"
