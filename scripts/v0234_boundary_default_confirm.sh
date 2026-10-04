#!/usr/bin/env bash
# Short current-head confirmation that a CLEAN/UNSET production child selects
# the accepted frozen-WRF boundary bundle and retains its recovered cadence.
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
fix_commit=8192d1b6fda7f247f4dc34ee056b39e0efb30ad9
accepted_proof=${sprint_dir}/FROZEN_BUNDLE_DISCRIMINATOR.json
seed_cache=<DATA_ROOT>/wrf_gpu2/v0234_frozen_bundle_perf_discriminator_11dd4a8bdd3e26b5/cache
seed_inventory_expected=a0d3d415a9b7bfaf70e7b86dc8c00d875ce4006ce145e49f157baae260968b93
run_root=<DATA_ROOT>/wrf_gpu2/v0234_boundary_default_confirm_${short_nonce}
cache=${run_root}/cache
namespace=${run_root}/default_unset
output_json=${sprint_dir}/BOUNDARY_DEFAULT_CONFIRM_PROBE.json
summary_json=${sprint_dir}/BOUNDARY_DEFAULT_CONFIRM.json
source_sha=$(git -C "$repo" rev-parse HEAD)
resource_csv=${run_root}/default_unset_nvidia_smi.csv
wall_json=${run_root}/default_unset_wall.json
time_txt=${run_root}/default_unset_time_v.txt

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
  echo "refusing non-fresh confirmation namespace: $run_root" >&2
  exit 65
fi
for path in "$output_json" "$summary_json"; do
  if [[ -e $path ]]; then
    echo "refusing to overwrite proof: $path" >&2
    exit 65
  fi
done
if ! git -C "$repo" merge-base --is-ancestor "$fix_commit" "$source_sha"; then
  echo "default-path fix is not an ancestor of current source" >&2
  exit 66
fi
if ! git -C "$repo" diff --quiet "$fix_commit" "$source_sha" -- src/gpuwrf; then
  echo "model source changed after the default-path fix" >&2
  exit 66
fi
if [[ -n $(git -C "$repo" status --short -- src/gpuwrf tests) ]]; then
  echo "refusing dirty model/test tree" >&2
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
  echo "missing immutable accepted-path compile seed: $seed_cache" >&2
  exit 68
fi
seed_inventory=$(inventory_sha256 "$seed_cache")
if [[ $seed_inventory != "$seed_inventory_expected" ]]; then
  echo "accepted-path compile seed inventory changed: $seed_inventory" >&2
  exit 68
fi
jq -e '.status == "PASS" and .configuration.frozen_wrf_boundary_bundle == true' \
  "$accepted_proof" >/dev/null

mkdir "$run_root"
mkdir "$cache"
cp --reflink=auto -a "$seed_cache/." "$cache/"

jq -n \
  --arg seed_cache "$seed_cache" \
  --arg seed_inventory "$seed_inventory" \
  --arg source_sha "$source_sha" \
  --arg fix_commit "$fix_commit" \
  --arg nonce "$nonce" \
  --arg accepted_proof "$accepted_proof" \
  '{seed_cache:$seed_cache,seed_inventory_sha256:$seed_inventory,
    source_sha:$source_sha,fix_commit:$fix_commit,nonce:$nonce,
    accepted_proof:$accepted_proof,
    isolation:"Exact accepted-path d01/d02 paired K=3 timing protocol, but GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE is deliberately UNSET so the repaired shared production default is the sole selector."}' \
  >"$run_root/SEED_ATTESTATION.json"

unset GPUWRF_ALLOCATOR GPUWRF_AOT_VERIFY GPUWRF_BITWISE GPUWRF_GWD_NESTED
unset GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE
unset XLA_FLAGS
export GPUWRF_NESTED_AOT=1
export GPUWRF_NESTED_FUSE=0
export GPUWRF_NESTED_DEFUSE_COMPILE=0
export GPUWRF_NESTED_PARALLEL_COMPILE=0
export GPUWRF_NESTED_SYNC_MODE=root
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
timeout --foreground --signal=INT --kill-after=30s 900s \
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
  '{phase:"default_unset",returncode:$rc,start_epoch_ns:$start_ns,
    end_epoch_ns:$end_ns,command_wall_s:(($end_ns-$start_ns)/1000000000),
    resource_csv:$resource_csv,time_v:$time_v}' \
  >"$wall_json"
if [[ $rc -ne 0 ]]; then
  echo "default-unset confirmation failed rc=$rc" >&2
  exit "$rc"
fi

accepted_state=$(jq -er '.exact_terminal.states_sha256' "$accepted_proof")
accepted_carry=$(jq -er '.exact_terminal.carries_sha256' "$accepted_proof")
jq -e \
  --arg state "$accepted_state" \
  --arg carry "$accepted_carry" \
  '.status == "PASS" and
   .environment.GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE == null and
   .setup.nested_frozen_wrf_boundary_bundle.d01 == false and
   .setup.nested_frozen_wrf_boundary_bundle.d02 == true and
   .result.all_theta_finite == true and
   .result.defuse_last.defused == true and
   .result.median_s_per_root_step <= 1.2109410627745092 and
   .result.terminal_states.ok == true and
   .result.terminal_carries.ok == true and
   .result.terminal_states.sha256 == $state and
   .result.terminal_carries.sha256 == $carry' \
  "$output_json" >/dev/null

peak_vram=$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$resource_csv")
peak_rss=$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/,"",$2); print ($2+0)/1024}' "$time_txt")

jq -n \
  --arg schema wrfgpu2.v0234.boundary-default-confirm.v1 \
  --arg source_sha "$source_sha" \
  --arg fix_commit "$fix_commit" \
  --arg fixture "$fixture" \
  --arg namespace "$run_root" \
  --arg proof "$output_json" \
  --arg accepted_proof "$accepted_proof" \
  --argjson peak_vram "$peak_vram" \
  --argjson peak_rss "$peak_rss" \
  --slurpfile measured "$output_json" \
  --slurpfile accepted "$accepted_proof" \
  '{schema:$schema,status:"PASS",source_sha:$source_sha,fix_commit:$fix_commit,
    fixture:$fixture,namespace:$namespace,proof:$proof,
    accepted_explicit_proof:$accepted_proof,
    selection:{environment_value:null,d01:false,d02:true,
      fresh_default_matches_accepted_configuration:true},
    exact_terminal:{states_sha256:$measured[0].result.terminal_states.sha256,
      carries_sha256:$measured[0].result.terminal_carries.sha256,
      matches_accepted_explicit_arm:
        ($measured[0].result.terminal_states.sha256 == $accepted[0].exact_terminal.states_sha256 and
         $measured[0].result.terminal_carries.sha256 == $accepted[0].exact_terminal.carries_sha256)},
    timing:{s_per_root_step:$measured[0].result.median_s_per_root_step,
      s_per_fc_hour:$measured[0].result.projected_s_per_fc_hour,
      paired_s_per_root_step:$measured[0].result.paired_s_per_root_step,
      accepted_explicit_s_per_root_step:$accepted[0].timing.candidate_on_s_per_root_step,
      acceptance_ceiling_s_per_root_step:1.2109410627745092},
    correctness:{all_theta_finite:$measured[0].result.all_theta_finite},
    resources:{card_peak_used_vram_mib:$peak_vram,process_peak_rss_mib:$peak_rss},
    verdict:"FRESH_DEFAULT_PATH_CONFIRMED_FAST_AND_EXACT"}' \
  >"$summary_json"

echo "BOUNDARY_DEFAULT_CONFIRM_COMPLETE summary=$summary_json namespace=$run_root"
