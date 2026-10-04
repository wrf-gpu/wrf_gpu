#!/usr/bin/env bash
# Current-head candidate-on frozen-WRF boundary bundle performance discriminator.
# Must run as exactly one manager-authorized canonical lock-v2
# production-preemptible payload. No output/profiler/WRF/MPI/model edit occurs.

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
seed_cache=<DATA_ROOT>/wrf_gpu2/v0234_b1_short_ab_63ae3d49b4aa67ca/cache
seed_inventory_expected=3ed61354c53f8bdb21ee7fe51906566fb1a400dedf3b8a2e6f51968527c16bd4
off_proof=${sprint_dir}/FLAT2_FUSION_DISCRIMINATOR_TIMEOUT.json
run_root=<DATA_ROOT>/wrf_gpu2/v0234_frozen_bundle_perf_discriminator_${short_nonce}
cache=${run_root}/cache
namespace=${run_root}/candidate_on
output_json=${sprint_dir}/FROZEN_BUNDLE_ON_CURRENT_PROBE.json
summary_json=${sprint_dir}/FROZEN_BUNDLE_DISCRIMINATOR.json
source_sha=$(git -C "$repo" rev-parse HEAD)
resource_csv=${run_root}/candidate_on_nvidia_smi.csv
wall_json=${run_root}/candidate_on_wall.json
time_txt=${run_root}/candidate_on_time_v.txt

inventory_sha256() {
  local root=$1
  (
    cd "$root"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

if [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]; then
  echo "refusing: canonical GPU lock is not held" >&2
  exit 65
fi
if [[ -e $run_root ]]; then
  echo "refusing non-fresh discriminator namespace: $run_root" >&2
  exit 65
fi
for path in "$output_json" "$summary_json"; do
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
if [[ ! -d $seed_cache ]]; then
  echo "missing immutable partial compile seed: $seed_cache" >&2
  exit 68
fi
seed_inventory=$(inventory_sha256 "$seed_cache")
if [[ $seed_inventory != "$seed_inventory_expected" ]]; then
  echo "partial compile seed inventory changed: $seed_inventory" >&2
  exit 68
fi
jq -e '.status | startswith("TIMEOUT_FAIL_CLOSED")' "$off_proof" >/dev/null

mkdir "$run_root"
mkdir "$cache"
cp --reflink=auto -a "$seed_cache/." "$cache/"

jq -n \
  --arg seed_cache "$seed_cache" \
  --arg seed_inventory "$seed_inventory" \
  --arg source_sha "$source_sha" \
  --arg nonce "$nonce" \
  --arg candidate_off_proof "$off_proof" \
  '{seed_cache:$seed_cache,seed_inventory_sha256:$seed_inventory,
    source_sha:$source_sha,nonce:$nonce,candidate_off_proof:$candidate_off_proof,
    isolation:"Same current-source de-fused/AOT/prepared K=3 protocol and exact fixture as the timed-out flag-off arm; the only model configuration flip is GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE=1."}' \
  >"$run_root/SEED_ATTESTATION.json"

unset GPUWRF_ALLOCATOR GPUWRF_AOT_VERIFY GPUWRF_BITWISE GPUWRF_GWD_NESTED
unset XLA_FLAGS
export GPUWRF_NESTED_AOT=1
export GPUWRF_NESTED_FUSE=0
export GPUWRF_NESTED_DEFUSE_COMPILE=0
export GPUWRF_NESTED_PARALLEL_COMPILE=0
export GPUWRF_NESTED_SYNC_MODE=root
export GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE=1
export GPUWRF_FINITE_CHECK=1
export GPUWRF_NEST_OUTPUT_PIPELINE=0
export GPUWRF_PREPARED_RUNTIME_REUSE=1
export GPUWRF_WRF_ROOT=$wrf_root
export GPUWRF_JAX_CACHE_DIR=$cache
export JAX_COMPILATION_CACHE_DIR=$cache
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -l 1 >"$resource_csv" &
sampler_pid=$!
start_ns=$(date +%s%N)
set +e
timeout --foreground --signal=INT --kill-after=30s 3000s \
  /usr/bin/time -v -o "$time_txt" \
  env PYTHONPATH=${repo}/src \
  <USER_HOME>/miniconda3/bin/python "$probe" \
    --input-dir "$fixture" \
    --namespace "$namespace" \
    --output-json "$output_json" \
    --source-repo "$repo" \
    --expected-source-sha "$source_sha" \
    --max-dom 2 \
    --k-steps 3 \
    --runtime-lifetime prepared
rc=$?
set -e
end_ns=$(date +%s%N)
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true

jq -n \
  --argjson rc "$rc" \
  --argjson start_ns "$start_ns" \
  --argjson end_ns "$end_ns" \
  --arg resource_csv "$resource_csv" \
  --arg time_v "$time_txt" \
  '{phase:"candidate_on",returncode:$rc,start_epoch_ns:$start_ns,
    end_epoch_ns:$end_ns,command_wall_s:(($end_ns-$start_ns)/1000000000),
    resource_csv:$resource_csv,time_v:$time_v}' \
  >"$wall_json"
if [[ $rc -ne 0 ]]; then
  echo "candidate-on phase failed rc=$rc" >&2
  exit "$rc"
fi

jq -e '.status == "PASS" and .result.all_theta_finite == true and
       .result.defuse_last.defused == true and
       .environment.GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE == "1" and
       .result.terminal_states.ok == true and .result.terminal_carries.ok == true' \
  "$output_json" >/dev/null

peak_vram=$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$resource_csv")
peak_rss=$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/,"",$2); print ($2+0)/1024}' "$time_txt")
state_sha=$(jq -er '.result.terminal_states.sha256' "$output_json")
carry_sha=$(jq -er '.result.terminal_carries.sha256' "$output_json")

jq -n \
  --arg schema wrfgpu2.v0234.frozen-boundary-bundle-perf-discriminator.v1 \
  --arg source_sha "$source_sha" \
  --arg fixture "$fixture" \
  --arg namespace "$run_root" \
  --arg probe "$output_json" \
  --arg state_sha "$state_sha" \
  --arg carry_sha "$carry_sha" \
  --argjson peak_vram "$peak_vram" \
  --argjson peak_rss "$peak_rss" \
  --slurpfile on "$output_json" \
  --slurpfile off "$off_proof" \
  '{schema:$schema,status:"PASS",source_sha:$source_sha,fixture:$fixture,
    namespace:$namespace,proof:$probe,
    configuration:{max_dom:2,root_dt_s:18,k_steps:3,output:false,
      runtime_lifetime:"prepared",aot:true,fused:false,root_sync:true,
      frozen_wrf_boundary_bundle:true},
    exact_terminal:{states_sha256:$state_sha,carries_sha256:$carry_sha},
    timing:{candidate_on_s_per_root_step:$on[0].result.median_s_per_root_step,
      candidate_on_s_per_fc_hour:$on[0].result.projected_s_per_fc_hour,
      candidate_off_unfinished_initial_1_lower_bound_s:$off[0].defused_arm.post_aot_initial_1_lower_bound.lower_bound_seconds,
      candidate_on_speedup_lower_bound_over_candidate_off:
        ($off[0].defused_arm.post_aot_initial_1_lower_bound.lower_bound_seconds /
         $on[0].result.median_s_per_root_step),
      retained_cpu_s_per_fc_hour:113.8815,
      gpu_speedup_over_retained_cpu:(113.8815/$on[0].result.projected_s_per_fc_hour)},
    resources:{card_peak_used_vram_mib:$peak_vram,process_peak_rss_mib:$peak_rss},
    causal_question:"Does the accepted frozen-WRF live-child boundary bundle eliminate the current legacy-path native-cascade pathology on the exact drift fixture?"}' \
  >"$summary_json"

echo "FROZEN_BUNDLE_DISCRIMINATOR_COMPLETE summary=$summary_json namespace=$run_root"
