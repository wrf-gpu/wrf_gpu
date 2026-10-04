#!/usr/bin/env bash
# Exactly one canonical-lock-v2 24 h native d01/d02 drift trajectory.  This
# payload runs GPUWRF only; the all-cell CPU-truth score is deliberately run
# after lock release so it does not consume the reserved GPU window.

set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <64-hex-nonce>" >&2
  exit 64
fi

nonce=$1
short_nonce=${nonce:0:16}
repo=<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-drift-perf-bisect
input_dir=<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z
truth_dir=<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z
wrf_root=<USER_HOME>/src/wrf_pristine/WRF
sprint_dir=${repo}/.agent/sprints/2026-07-20-v0234-gpt-drift-perf-bisect
validator=${repo}/scripts/v0234_gpt_drift_validation.py
harness_anchor=c86f5328f037e385c724d4748276d2ae0916a730
namespace=<DATA_ROOT>/wrf_gpu2/v0234_drift24_${short_nonce}
cache_dir=<DATA_ROOT>/wrf_gpu2/v0234_drift24_cache_${short_nonce}
monitor_dir=<DATA_ROOT>/wrf_gpu2/v0234_drift24_monitor_${short_nonce}
preflight=${sprint_dir}/DRIFT24_STATIC_PREFLIGHT_${short_nonce}.json
resource_csv=${monitor_dir}/nvidia_smi.csv
time_v=${monitor_dir}/time_v.txt
wall_json=${monitor_dir}/wall.json
summary_json=${sprint_dir}/DRIFT24_ARM_SUMMARY_${short_nonce}.json

if [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]; then
  echo "refusing: canonical GPU lock is not held" >&2
  exit 65
fi
if ! git -C "$repo" merge-base --is-ancestor "$harness_anchor" HEAD; then
  echo "refusing: reviewed 24 h harness anchor is not an ancestor" >&2
  exit 66
fi
if ! git -C "$repo" diff --quiet "$harness_anchor" HEAD -- \
  src/gpuwrf scripts/v0234_gpt_drift_validation.py \
  tests/test_v0234_gpt_drift_validation.py
then
  echo "refusing: model or reviewed validation harness changed after anchor" >&2
  exit 66
fi
if [[ -n $(git -C "$repo" status --short -- \
  src/gpuwrf tests scripts/v0234_gpt_drift_validation.py scripts/v0234_gpt_drift_24h.sh) ]]
then
  echo "refusing: model/test/validator tree is dirty" >&2
  exit 66
fi
for path in "$namespace" "$cache_dir" "$monitor_dir" "$summary_json"; do
  if [[ -e $path ]]; then
    echo "refusing non-fresh 24 h artifact path: $path" >&2
    exit 67
  fi
done
if [[ ! -f $preflight ]]; then
  echo "missing CPU-only static preflight: $preflight" >&2
  exit 68
fi
jq -e \
  --arg namespace "$namespace" \
  --arg cache_dir "$cache_dir" \
  --arg nonce "$nonce" \
  '.verdict == "STATIC_PREFLIGHT_PASS" and
   .forecast_hours == 24 and
   .namespace == $namespace and
   .cache_destination == $cache_dir and
   .nonce == $nonce and
   .radiation_cadence_minutes == 30 and
   .output_cadence_minutes == 60 and
   .candidate_uses_cpu_wrf_history == false' \
  "$preflight" >/dev/null

mkdir "$monitor_dir"
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -l 1 >"$resource_csv" &
sampler_pid=$!
stop_sampler() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap stop_sampler EXIT

start_ns=$(date +%s%N)
set +e
timeout --foreground --signal=INT --kill-after=60s 7200s \
  /usr/bin/time -v -o "$time_v" \
  <USER_HOME>/miniconda3/bin/python "$validator" arm \
    --repo "$repo" \
    --input-dir "$input_dir" \
    --truth-dir "$truth_dir" \
    --wrf-root "$wrf_root" \
    --namespace "$namespace" \
    --cache-dir "$cache_dir" \
    --nonce "$nonce" \
    --hours 24
rc=$?
set -e
end_ns=$(date +%s%N)
stop_sampler
trap - EXIT

jq -n \
  --argjson rc "$rc" \
  --argjson start_ns "$start_ns" \
  --argjson end_ns "$end_ns" \
  --arg resource_csv "$resource_csv" \
  --arg time_v "$time_v" \
  '{phase:"drift24_gpu_trajectory",returncode:$rc,start_epoch_ns:$start_ns,
    end_epoch_ns:$end_ns,command_wall_s:(($end_ns-$start_ns)/1000000000),
    phase_bound_seconds:7200,kill_grace_seconds:60,
    resource_csv:$resource_csv,time_v:$time_v}' >"$wall_json"
if [[ $rc -ne 0 ]]; then
  echo "24 h drift trajectory failed or timed out rc=$rc" >&2
  exit "$rc"
fi

pipeline_proof=${namespace}/pipeline_proofs/nested_pipeline_run.json
arm_result=${namespace}/ARM_RESULT.json
jq -e \
  '.verdict == "PIPELINE_GREEN" and
   .hours == 24 and
   .max_dom == 2 and
   .domains == ["d01", "d02"] and
   .all_domains_finite == true and
   .all_outputs_present == true and
   .per_domain.d01.history_interval_min == 60 and
   .per_domain.d02.history_interval_min == 60 and
   .per_domain.d01.wrfout_count == 24 and
   .per_domain.d02.wrfout_count == 24 and
   .per_domain.d01.expected_wrfout_count == 24 and
   .per_domain.d02.expected_wrfout_count == 24 and
   .metadata.domains.d01.namelist.radiation_cadence_steps == 100 and
   .metadata.domains.d02.namelist.radiation_cadence_steps == 300 and
   .metadata.domains.d01.namelist.nested_frozen_wrf_boundary_bundle == false and
   .metadata.domains.d02.namelist.nested_frozen_wrf_boundary_bundle == true and
   .metadata.nested_runtime.prepared_runtime_reuse == true' \
  "$pipeline_proof" >/dev/null
jq -e \
  '.returncode == 0 and .forecast_hours == 24 and
   .lock_intent == "production-preemptible" and
   .lock_arm_count == 1 and .science_trajectory_count == 1 and
   .retry_count == 0' \
  "$arm_result" >/dev/null

peak_vram=$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$resource_csv")
peak_rss=$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/,"",$2); print ($2+0)/1024}' "$time_v")
jq -n \
  --arg namespace "$namespace" \
  --arg pipeline_proof "$pipeline_proof" \
  --arg arm_result "$arm_result" \
  --arg wall_json "$wall_json" \
  --arg resource_csv "$resource_csv" \
  --arg time_v "$time_v" \
  --argjson peak_vram "$peak_vram" \
  --argjson peak_rss "$peak_rss" \
  --arg pipeline_sha "$(sha256sum "$pipeline_proof" | awk '{print $1}')" \
  --arg arm_sha "$(sha256sum "$arm_result" | awk '{print $1}')" \
  --slurpfile pipeline "$pipeline_proof" \
  '{schema:"wrfgpu2.v0234.drift24-arm-summary.v1",status:"GPU_TRAJECTORY_PASS",
    namespace:$namespace,forecast_hours:24,max_dom:2,output:true,
    output_cadence_minutes:60,radiation_cadence_minutes:30,
    forecast_wall_seconds:$pipeline[0].wall_clock_forecast_only_s,
    total_model_wall_seconds:$pipeline[0].wall_clock_total_s,
    all_domains_finite:$pipeline[0].all_domains_finite,
    all_outputs_present:$pipeline[0].all_outputs_present,
    resources:{card_peak_used_vram_mib:$peak_vram,process_peak_rss_mib:$peak_rss},
    artifacts:{pipeline_proof:$pipeline_proof,pipeline_proof_sha256:$pipeline_sha,
      arm_result:$arm_result,arm_result_sha256:$arm_sha,wall:$wall_json,
      nvidia_smi:$resource_csv,time_v:$time_v},
    cpu_truth_score_pending_after_lock_release:true}' >"$summary_json"

echo "DRIFT24_GPU_TRAJECTORY_COMPLETE summary=$summary_json"
echo "SCORE_AFTER_LOCK: taskset -c 13,14,15,29,30,31 nice -n 15 ionice -c 3 env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= <USER_HOME>/miniconda3/bin/python $validator score --truth-dir $truth_dir --candidate-dir $namespace/gpu_output/canary_l2_20260501_18z_l2_72h_20260519T173026Z --hours 24 --out $sprint_dir/DRIFT24_SCORE_${short_nonce}.json"
