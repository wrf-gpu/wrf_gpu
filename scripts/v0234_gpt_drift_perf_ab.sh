#!/usr/bin/env bash
# One manager-authorized lock-v2 payload for a matched current/pre-chain timing A/B.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <64-hex-nonce>" >&2
  exit 64
fi
if [[ ! $1 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <64-hex-nonce>" >&2
  exit 64
fi

nonce=$1
short_nonce=${nonce:0:16}
current_repo=<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-drift-perf-bisect
prechain_repo=/tmp/v0234-gpt-perf-prechain-6045ff02
fixture=<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z
wrf_root=<USER_HOME>/src/wrf_pristine/WRF
sprint_dir=${current_repo}/.agent/sprints/2026-07-20-v0234-gpt-drift-perf-bisect
probe=${current_repo}/scripts/v0234_gpt_drift_perf_probe.py
current_namespace=<DATA_ROOT>/wrf_gpu2/v0234_gpt_drift_perf_current_replacement_${short_nonce}
prechain_namespace=<DATA_ROOT>/wrf_gpu2/v0234_gpt_drift_perf_prechain_6045_replacement_${short_nonce}
current_json=${sprint_dir}/CURRENT_MATCHED_PROBE_REPLACEMENT.json
prechain_json=${sprint_dir}/PRECHAIN_6045_MATCHED_PROBE_REPLACEMENT.json

if [[ -e $current_namespace || -e $prechain_namespace ]]; then
  echo "refusing non-fresh A/B namespace" >&2
  exit 65
fi
if [[ -e $current_json || -e $prechain_json ]]; then
  echo "refusing to overwrite an existing sprint proof" >&2
  exit 65
fi
if [[ $(git -C "$prechain_repo" rev-parse HEAD) != 6045ff02f263f52614f9f21de19550615b847af0 ]]; then
  echo "pre-chain detached worktree is not at 6045ff02" >&2
  exit 66
fi
if [[ -n $(git -C "$prechain_repo" status --short -- src/gpuwrf) ]]; then
  echo "pre-chain src/gpuwrf is dirty" >&2
  exit 66
fi
for table in \
  run/MPTABLE.TBL \
  run/SOILPARM.TBL \
  run/GENPARM.TBL \
  run/RRTMG_LW_DATA \
  run/RRTMG_SW_DATA \
  run/CAMtr_volume_mixing_ratio
do
  if [[ ! -f $wrf_root/$table ]]; then
    echo "missing production WRF runtime table: $wrf_root/$table" >&2
    exit 67
  fi
done

# Match the production CLI's allocator re-exec environment while preserving
# the release defaults (fused, AOT-on, verify-off, root-sync) as UNSET values.
unset GPUWRF_ALLOCATOR GPUWRF_NESTED_AOT GPUWRF_AOT_VERIFY
unset GPUWRF_NESTED_FUSE GPUWRF_NESTED_DEFUSE_COMPILE
unset GPUWRF_NESTED_PARALLEL_COMPILE GPUWRF_NESTED_SYNC_MODE
unset GPUWRF_BITWISE GPUWRF_GWD_NESTED XLA_FLAGS
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export GPUWRF_WRF_ROOT=$wrf_root
export PYTHONUNBUFFERED=1

echo "AB_PHASE current source=$(git -C "$current_repo" rev-parse HEAD) namespace=$current_namespace"
timeout --foreground --signal=INT --kill-after=30s 900s \
  env PYTHONPATH=${current_repo}/src \
  python "$probe" \
    --input-dir "$fixture" \
    --namespace "$current_namespace" \
    --output-json "$current_json" \
    --source-repo "$current_repo" \
    --max-dom 2 \
    --k-steps 4

echo "AB_PHASE prechain source=6045ff02f263f52614f9f21de19550615b847af0 namespace=$prechain_namespace"
timeout --foreground --signal=INT --kill-after=30s 2400s \
  env PYTHONPATH=${prechain_repo}/src \
  python "$probe" \
    --input-dir "$fixture" \
    --namespace "$prechain_namespace" \
    --output-json "$prechain_json" \
    --source-repo "$prechain_repo" \
    --expected-source-sha 6045ff02f263f52614f9f21de19550615b847af0 \
    --max-dom 2 \
    --k-steps 4

echo "AB_COMPLETE current=$current_json prechain=$prechain_json"
