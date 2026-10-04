#!/usr/bin/env bash
# Current-head flat-maxdom2 fused-vs-de-fused steady-state discriminator.
# Must run as exactly one manager-authorized canonical lock-v2
# production-preemptible payload. No model/output/profiler/WRF/MPI work occurs.

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
global_cache=<USER_HOME>/.cache/gpuwrf/jit/0.23.3-jax0.10.0-jaxlib0.10.0-cuda_sm120
run_root=<DATA_ROOT>/wrf_gpu2/v0234_flat2_fusion_discriminator_${short_nonce}
defused_json=${sprint_dir}/FLAT2_DEFUSED_CURRENT_PROBE.json
fused_json=${sprint_dir}/FLAT2_FUSED_CURRENT_CONTROL.json
summary_json=${sprint_dir}/FLAT2_FUSION_DISCRIMINATOR.json
source_sha=$(git -C "$repo" rev-parse HEAD)

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
for path in "$defused_json" "$fused_json" "$summary_json"; do
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

# These are the three retained exact-version persistent fused executables. Only
# the matching HLO can hit; copying all three avoids guessing which current
# carry variant JAX will request. SHA-256 pins the immutable seed bytes.
declare -A fused_seed_sha=(
  [jit_fused_jit-6b63f1c1440d3b0c9c1585b008bac0014af87774905d3b6a5e434fc3d98b35b3-cache]=d72f2b1394f089d5f29694be80baed0897d21d73b8267311b72122dca25ff005
  [jit_fused_jit-90918c83766fb3b5f206225a879f8a1904d340862ff8c7a912e7cc1875dc3683-cache]=51515bdf9c47de457b2e6968ae7acbc13ff95f6d927eb830b029af9028e77eba
  [jit_fused_jit-e0908a66905d19c68e9106803f7b4e4393e8071ee86585992397d64101923164-cache]=0e9adcc45d23f545279912b733a72a6fd2b4ab626f5e346cdb3c603793e9109c
)
for file in "${!fused_seed_sha[@]}"; do
  got=$(sha256sum "$global_cache/$file" | awk '{print $1}')
  if [[ $got != "${fused_seed_sha[$file]}" ]]; then
    echo "fused persistent-cache seed changed: $file $got" >&2
    exit 68
  fi
  if [[ ! -f $global_cache/${file%-cache}-atime ]]; then
    echo "missing fused persistent-cache atime companion: $file" >&2
    exit 68
  fi
done

mkdir "$run_root"
mkdir "$run_root/cache_defused" "$run_root/cache_fused"
cp --reflink=auto -a "$seed_cache/." "$run_root/cache_defused/"
cp --reflink=auto -a "$seed_cache/." "$run_root/cache_fused/"
for file in "${!fused_seed_sha[@]}"; do
  cp --reflink=auto -a "$global_cache/$file" "$run_root/cache_fused/$file"
  cp --reflink=auto -a "$global_cache/${file%-cache}-atime" "$run_root/cache_fused/${file%-cache}-atime"
done

jq -n \
  --arg seed_cache "$seed_cache" \
  --arg seed_inventory "$seed_inventory" \
  --arg source_sha "$source_sha" \
  --arg nonce "$nonce" \
  '{seed_cache:$seed_cache,seed_inventory_sha256:$seed_inventory,
    source_sha:$source_sha,nonce:$nonce,
    note:"Fresh arm caches are byte-seeded from the invalid B1 partial compiler namespace; timing begins only after executable construction/loading."}' \
  >"$run_root/SEED_ATTESTATION.json"

unset GPUWRF_ALLOCATOR GPUWRF_AOT_VERIFY GPUWRF_BITWISE GPUWRF_GWD_NESTED
unset GPUWRF_NESTED_DEFUSE_COMPILE XLA_FLAGS
export GPUWRF_NESTED_AOT=1
export GPUWRF_NESTED_PARALLEL_COMPILE=0
export GPUWRF_NESTED_SYNC_MODE=root
export GPUWRF_FINITE_CHECK=1
export GPUWRF_NEST_OUTPUT_PIPELINE=0
export GPUWRF_PREPARED_RUNTIME_REUSE=1
export GPUWRF_WRF_ROOT=$wrf_root
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1

run_phase() {
  local phase=$1
  local timeout_s=$2
  local fuse=$3
  local cache=$4
  local namespace=$5
  local output_json=$6
  local resource_csv=$run_root/${phase}_nvidia_smi.csv
  local wall_json=$run_root/${phase}_wall.json
  local sampler_pid=0
  local start_ns end_ns rc

  export GPUWRF_NESTED_FUSE=$fuse
  export GPUWRF_JAX_CACHE_DIR=$cache
  export JAX_COMPILATION_CACHE_DIR=$cache

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
      --k-steps 3 \
      --runtime-lifetime prepared
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

# Run the unknown arm first. If its resumed d02 compilation cannot finish in
# 40 minutes, fail closed without spending GPU time on a redundant fused control.
run_phase defused 2400 0 "$run_root/cache_defused" "$run_root/defused" "$defused_json"
run_phase fused 750 1 "$run_root/cache_fused" "$run_root/fused" "$fused_json"

jq -e '.status == "PASS" and .result.all_theta_finite == true and
       .result.defuse_last.defused == true and
       .result.terminal_states.ok == true and .result.terminal_carries.ok == true' \
  "$defused_json" >/dev/null
jq -e '.status == "PASS" and .result.all_theta_finite == true and
       .result.defuse_last.fused == true and
       .result.terminal_states.ok == true and .result.terminal_carries.ok == true' \
  "$fused_json" >/dev/null

defused_state=$(jq -er '.result.terminal_states.sha256' "$defused_json")
fused_state=$(jq -er '.result.terminal_states.sha256' "$fused_json")
defused_carry=$(jq -er '.result.terminal_carries.sha256' "$defused_json")
fused_carry=$(jq -er '.result.terminal_carries.sha256' "$fused_json")
if [[ $defused_state != "$fused_state" || $defused_carry != "$fused_carry" ]]; then
  echo "fused/de-fused terminal state or carry digest differs" >&2
  exit 69
fi

defused_vram=$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$run_root/defused_nvidia_smi.csv")
fused_vram=$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$run_root/fused_nvidia_smi.csv")
defused_rss=$(jq '[.calls[].rss_peak_mib] | max' "$defused_json")
fused_rss=$(jq '[.calls[].rss_peak_mib] | max' "$fused_json")

jq -n \
  --arg schema wrfgpu2.v0234.flat2-fusion-discriminator.v1 \
  --arg source_sha "$source_sha" \
  --arg fixture "$fixture" \
  --arg namespace "$run_root" \
  --arg defused_json "$defused_json" \
  --arg fused_json "$fused_json" \
  --arg state_sha256 "$defused_state" \
  --arg carry_sha256 "$defused_carry" \
  --argjson defused_peak_vram_mib "$defused_vram" \
  --argjson fused_peak_vram_mib "$fused_vram" \
  --argjson defused_peak_rss_mib "$defused_rss" \
  --argjson fused_peak_rss_mib "$fused_rss" \
  --slurpfile d "$defused_json" \
  --slurpfile f "$fused_json" \
  '{schema:$schema,status:"PASS",source_sha:$source_sha,fixture:$fixture,
    namespace:$namespace,proofs:{defused:$defused_json,fused:$fused_json},
    configuration:{max_dom:2,root_dt_s:18,k_steps:3,output:false,
      runtime_lifetime:"prepared",aot:true,root_sync:true},
    exact_terminal:{states_sha256:$state_sha256,carries_sha256:$carry_sha256},
    timing:{
      defused_s_per_root_step:$d[0].result.median_s_per_root_step,
      fused_s_per_root_step:$f[0].result.median_s_per_root_step,
      defused_s_per_fc_hour:$d[0].result.projected_s_per_fc_hour,
      fused_s_per_fc_hour:$f[0].result.projected_s_per_fc_hour,
      defused_speedup_over_fused:($f[0].result.median_s_per_root_step/$d[0].result.median_s_per_root_step),
      retained_cpu_s_per_fc_hour:113.8815,
      defused_gpu_speedup_over_retained_cpu:(113.8815/$d[0].result.projected_s_per_fc_hour)},
    resources:{defused_peak_vram_mib:$defused_peak_vram_mib,
      fused_peak_vram_mib:$fused_peak_vram_mib,
      defused_peak_rss_mib:$defused_peak_rss_mib,
      fused_peak_rss_mib:$fused_peak_rss_mib},
    causal_candidate:{commit:"fe4949d31f8570c09777208b13c6666b8802fe92",
      mechanism:"default-on whole-root fusion for the previously unfused flat max-dom2 cascade",
      historical_gpu_perf_gate:"deferred in P3 report and retained execution gate was never run"}}' \
  >"$summary_json"

echo "FLAT2_FUSION_DISCRIMINATOR_COMPLETE summary=$summary_json namespace=$run_root"
