#!/usr/bin/env bash
# Short, output-off B1 mechanism sanity check on the exact drift d01/d02 fixture.
# Must run as one canonical lock-v2 production-preemptible payload.

set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <64-hex-nonce>" >&2
  exit 64
fi

nonce=$1
short_nonce=${nonce:0:16}
repo=<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-drift-perf-bisect
fixture=<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z
wrf_root=<USER_HOME>/src/wrf_pristine/WRF
sprint_dir=${repo}/.agent/sprints/2026-07-20-v0234-gpt-drift-perf-bisect
probe=${repo}/scripts/v0234_gpt_drift_perf_probe.py
run_root=<DATA_ROOT>/wrf_gpu2/v0234_b1_short_ab_${short_nonce}
prewarm_json=${sprint_dir}/B1_SHORT_PREWARM.json
arm_a_json=${sprint_dir}/B1_SHORT_RELEASED.json
arm_b_json=${sprint_dir}/B1_SHORT_PREPARED.json
summary_json=${sprint_dir}/B1_SHORT_AB_SUMMARY.json
source_sha=$(git -C "$repo" rev-parse HEAD)

if [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]; then
  echo "refusing: canonical GPU lock is not held" >&2
  exit 65
fi
if [[ -e $run_root ]]; then
  echo "refusing non-fresh short-A/B namespace: $run_root" >&2
  exit 65
fi
for path in "$prewarm_json" "$arm_a_json" "$arm_b_json" "$summary_json"; do
  if [[ -e $path ]]; then
    echo "refusing to overwrite proof: $path" >&2
    exit 65
  fi
done
if [[ -n $(git -C "$repo" status --short -- src/gpuwrf) ]]; then
  echo "refusing dirty model tree" >&2
  exit 66
fi
if [[ ! -f $fixture/namelist.input || ! -f $fixture/wrfinput_d01 || ! -f $fixture/wrfinput_d02 ]]; then
  echo "exact d01/d02 fixture is incomplete" >&2
  exit 67
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

mkdir "$run_root"

# Match the reviewed B1 path: de-fused B=1, AOT on, sequential construction,
# root synchronization, finite guard, x64, and output disabled by the probe.
unset GPUWRF_ALLOCATOR GPUWRF_AOT_VERIFY GPUWRF_BITWISE
unset GPUWRF_NESTED_DEFUSE_COMPILE GPUWRF_GWD_NESTED
unset GPUWRF_PREPARED_RUNTIME_REUSE XLA_FLAGS
export GPUWRF_NESTED_FUSE=0
export GPUWRF_NESTED_AOT=1
export GPUWRF_NESTED_PARALLEL_COMPILE=0
export GPUWRF_NESTED_SYNC_MODE=root
export GPUWRF_FINITE_CHECK=1
export GPUWRF_NEST_OUTPUT_PIPELINE=0
export GPUWRF_WRF_ROOT=$wrf_root
export GPUWRF_JAX_CACHE_DIR=$run_root/cache
export JAX_COMPILATION_CACHE_DIR=$run_root/cache
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1

run_phase() {
  local phase=$1
  local timeout_s=$2
  local namespace=$3
  local output_json=$4
  local runtime_lifetime=$5
  local segment_count=$6
  local expected_loads=${7:-}
  local resource_csv=$run_root/${phase}_nvidia_smi.csv
  local wall_json=$run_root/${phase}_wall.json
  local sampler_pid=0
  local start_ns end_ns rc
  local -a expected_arg=()

  if [[ -n $expected_loads ]]; then
    expected_arg=(--expected-load-count "$expected_loads")
  fi
  nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits -l 1 >"$resource_csv" &
  sampler_pid=$!
  start_ns=$(date +%s%N)
  set +e
  timeout --foreground --signal=INT --kill-after=30s "${timeout_s}s" \
    env PYTHONPATH=${repo}/src \
    <USER_HOME>/miniconda3/bin/python "$probe" \
      --input-dir "$fixture" \
      --namespace "$namespace" \
      --output-json "$output_json" \
      --source-repo "$repo" \
      --expected-source-sha "$source_sha" \
      --max-dom 2 \
      --protocol segmented \
      --runtime-lifetime "$runtime_lifetime" \
      --segment-count "$segment_count" \
      --segment-root-steps 1 \
      "${expected_arg[@]}"
  rc=$?
  set -e
  end_ns=$(date +%s%N)
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  jq -n \
    --arg phase "$phase" \
    --argjson rc "$rc" \
    --argjson start_ns "$start_ns" \
    --argjson end_ns "$end_ns" \
    --arg resource_csv "$resource_csv" \
    '{phase:$phase,returncode:$rc,start_epoch_ns:$start_ns,end_epoch_ns:$end_ns,
      command_wall_s:(($end_ns-$start_ns)/1000000000),resource_csv:$resource_csv}' \
    >"$wall_json"
  if [[ $rc -ne 0 ]]; then
    echo "phase $phase failed rc=$rc" >&2
    return "$rc"
  fi
}

# A one-segment current-source call creates any missing exact de-fused AOT blobs.
# It is not part of the timing comparison. The two timed arms then start in
# separate fresh processes against the same warmed cache and identical source.
run_phase prewarm 1200 "$run_root/prewarm" "$prewarm_json" prepared 1
run_phase arm_a_released 600 "$run_root/arm_a_released" "$arm_a_json" released 3 6
run_phase arm_b_prepared 600 "$run_root/arm_b_prepared" "$arm_b_json" prepared 3 2

jq -e '.status == "PASS" and .result.expected_load_count_ok == true' "$arm_a_json" >/dev/null
jq -e '.status == "PASS" and .result.expected_load_count_ok == true' "$arm_b_json" >/dev/null

a_state=$(jq -er '.result.terminal_states.sha256' "$arm_a_json")
b_state=$(jq -er '.result.terminal_states.sha256' "$arm_b_json")
a_carry=$(jq -er '.result.terminal_carries.sha256' "$arm_a_json")
b_carry=$(jq -er '.result.terminal_carries.sha256' "$arm_b_json")
if [[ $a_state != "$b_state" || $a_carry != "$b_carry" ]]; then
  echo "released/prepared terminal state or carry digest differs" >&2
  exit 68
fi

jq -n \
  --arg schema wrfgpu2.v0234.prepared-runtime-short-ab.v1 \
  --arg source_sha "$source_sha" \
  --arg fixture "$fixture" \
  --arg namespace "$run_root" \
  --arg prewarm_json "$prewarm_json" \
  --arg arm_a_json "$arm_a_json" \
  --arg arm_b_json "$arm_b_json" \
  --arg state_sha256 "$a_state" \
  --arg carry_sha256 "$a_carry" \
  --slurpfile a "$arm_a_json" \
  --slurpfile b "$arm_b_json" \
  '{schema:$schema,status:"PASS_DIRECTIONAL_SANITY",source_sha:$source_sha,
    fixture:$fixture,namespace:$namespace,proofs:{prewarm:$prewarm_json,
    released:$arm_a_json,prepared:$arm_b_json},mechanism:{released_loads:$a[0].result.observed_load_count,
    prepared_loads:$b[0].result.observed_load_count,expected_transition:"6_to_2"},
    exact_terminal:{states_sha256:$state_sha256,carries_sha256:$carry_sha256},
    timing:{released_segment_wall_s:$a[0].result.total_segment_wall_s,
    prepared_segment_wall_s:$b[0].result.total_segment_wall_s,
    command_wall_gain_fraction:(($a[0].result.total_segment_wall_s-$b[0].result.total_segment_wall_s)/$a[0].result.total_segment_wall_s),
    released_post_first_median_s:$a[0].result.post_first_segment_median_wall_s,
    prepared_post_first_median_s:$b[0].result.post_first_segment_median_wall_s,
    post_first_gain_fraction:(($a[0].result.post_first_segment_median_wall_s-$b[0].result.post_first_segment_median_wall_s)/$a[0].result.post_first_segment_median_wall_s)},
    scope:"output-off, max_dom=2, three one-root-step segments; directional sanity only, not B1 admission"}' \
  >"$summary_json"

echo "B1_SHORT_AB_COMPLETE summary=$summary_json namespace=$run_root"
