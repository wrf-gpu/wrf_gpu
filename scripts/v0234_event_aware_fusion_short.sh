#!/usr/bin/env bash
# One short, warm, output-free B2 event-aware K=1/K=3 scheduler screen.
# Execution requires a separately manager-authorized canonical lock-v2 payload.

set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <64-hex-nonce>" >&2
  exit 64
fi

nonce=$1
short_nonce=${nonce:0:16}
# repo is derived from THIS wrapper's location so the arm always uses the probe in
# the same worktree/branch as the wrapper (the Opus B2 discriminator branch carries
# the corrected probe with the MANAGER-ACCEPTED bounded terminal gate).
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
wrf_root=<USER_HOME>/src/wrf_pristine/WRF
sprint_dir=$repo/.agent/sprints/2026-07-20-v0234-opus-b2-discriminator
probe=$repo/scripts/v0234_event_aware_fusion_probe.py
run_root=<DATA_ROOT>/wrf_gpu2/v0234_b2_event_aware_short_${short_nonce}
probe_namespace=$run_root/probe
proof_json=$sprint_dir/B2_EVENT_AWARE_SHORT_SCREEN_${short_nonce}.json
summary_json=$sprint_dir/B2_EVENT_AWARE_SHORT_RESULT_${short_nonce}.json
source_sha=$(git -C "$repo" rev-parse HEAD)
source_diff_base=93a2a7ee4b9a483146a1104a970adcceeb2292a0
resource_csv=$run_root/nvidia_smi.csv
process_tsv=$run_root/process_resources.tsv
time_v=$run_root/time_v.txt
wall_json=$run_root/wall.json
cache_dir=$run_root/cache

if [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]; then
  echo "refusing: canonical GPU lock is not held" >&2
  exit 65
fi
if [[ -e $run_root ]]; then
  echo "refusing non-fresh namespace: $run_root" >&2
  exit 65
fi
for path in "$proof_json" "$summary_json"; do
  if [[ -e $path ]]; then
    echo "refusing to overwrite proof: $path" >&2
    exit 65
  fi
done
if [[ -n $(git -C "$repo" status --short -- src/gpuwrf) ]]; then
  echo "refusing dirty production model tree" >&2
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

mkdir "$run_root" "$cache_dir"

unset GPUWRF_AOT_VERIFY GPUWRF_BITWISE GPUWRF_GWD_NESTED
unset GPUWRF_NESTED_DEFUSE_COMPILE GPUWRF_NESTED_EVENT_AWARE_FUSION_K
unset GPUWRF_NESTED_PARALLEL_COMPILE XLA_FLAGS
export GPUWRF_NESTED_AOT=0
export GPUWRF_NESTED_FUSE=1
export GPUWRF_NESTED_PARALLEL_COMPILE=0
export GPUWRF_NESTED_SYNC_MODE=root
export GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE=0
export GPUWRF_FINITE_CHECK=1
export GPUWRF_PREPARED_RUNTIME_REUSE=1
export GPUWRF_WRF_ROOT=$wrf_root
export GPUWRF_JAX_CACHE_DIR=$cache_dir
export JAX_COMPILATION_CACHE_DIR=$cache_dir
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -l 1 >"$resource_csv" &
sampler_pid=$!

start_ns=$(date +%s%N)
set +e
/usr/bin/time -v -o "$time_v" \
  timeout --foreground --signal=INT --kill-after=30s 1200s \
  env PYTHONPATH=$repo/src \
  <USER_HOME>/miniconda3/bin/python "$probe" \
    --namespace "$probe_namespace" \
    --output-json "$proof_json" \
    --source-repo "$repo" \
    --expected-source-sha "$source_sha" \
    --source-diff-base "$source_diff_base" \
    --repeats 5 &
command_pid=$!
set -e

(
  printf 'epoch_s\tpid\tvmrss_kib\tvmswap_kib\tmemavailable_kib\n'
  while kill -0 "$command_pid" 2>/dev/null; do
    timeout_pid=$(pgrep -P "$command_pid" | head -n 1 || true)
    python_pid=""
    if [[ -n $timeout_pid ]]; then
      python_pid=$(pgrep -P "$timeout_pid" | head -n 1 || true)
    fi
    target_pid=${python_pid:-${timeout_pid:-$command_pid}}
    rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$target_pid/status" 2>/dev/null || true)
    swap=$(awk '/^VmSwap:/ {print $2}' "/proc/$target_pid/status" 2>/dev/null || true)
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%s)" "$target_pid" "${rss:-0}" "${swap:-0}" "${available:-0}"
    sleep 1
  done
) >"$process_tsv" &
process_sampler_pid=$!

set +e
wait "$command_pid"
rc=$?
set -e
end_ns=$(date +%s%N)
kill "$process_sampler_pid" "$sampler_pid" 2>/dev/null || true
wait "$process_sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true

jq -n \
  --argjson returncode "$rc" \
  --argjson start_ns "$start_ns" \
  --argjson end_ns "$end_ns" \
  --arg resource_csv "$resource_csv" \
  --arg process_tsv "$process_tsv" \
  --arg time_v "$time_v" \
  '{returncode:$returncode,start_epoch_ns:$start_ns,end_epoch_ns:$end_ns,
    command_wall_s:(($end_ns-$start_ns)/1000000000),resource_csv:$resource_csv,
    process_resources:$process_tsv,time_v:$time_v}' >"$wall_json"

if [[ $rc -ne 0 ]]; then
  echo "B2 short screen failed rc=$rc; retained $run_root and $proof_json" >&2
  exit "$rc"
fi

# MANAGER-ACCEPTED bounded terminal criterion (0:1, 2026-07-20): the redundant
# post-check gates on ordering-exact + finite + the BOUNDED terminal verdict
# (fused-vs-eager within the in-run eager reassociation floor), matching the probe's
# own gate. Strict bit identity is NOT required (it is reassociation-scale); the probe
# retains it only as .exactness.terminal_bit_identical_informational.
jq -e '
  .status == "PASS" and
  .transfer_audit.ok == true and
  .exactness.event_state_output_order_exact == true and
  .exactness.bounded_terminal_criterion.passed == true and
  .exactness.all_numeric_leaves_finite == true and
  .exactness.candidate_k3_cascade_counts == {
    "event_fallback:d02": 1, "fused:d02": 8
  }
' "$proof_json" >/dev/null

peak_vram=$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$resource_csv")
min_free_vram=$(awk -F, '{gsub(/ /,"",$3); v=$3+0; if (!seen || v<m) m=v; seen=1} END {print m+0}' "$resource_csv")
mean_gpu_util=$(awk -F, '{gsub(/ /,"",$4); sum+=$4+0; n++} END {if (n) print sum/n; else print 0}' "$resource_csv")
sampled_peak_rss_kib=$(awk 'NR>1 {v=$3+0; if (v>m) m=v} END {print m+0}' "$process_tsv")
sampled_peak_swap_kib=$(awk 'NR>1 {v=$4+0; if (v>m) m=v} END {print m+0}' "$process_tsv")
time_peak_rss_kib=$(awk -F: '/Maximum resident set size/ {gsub(/ /,"",$2); print $2+0}' "$time_v")
gain=$(jq -r '.timing.candidate_k3_gain_fraction' "$proof_json")
classification=$(awk -v gain="$gain" 'BEGIN {
  if (gain >= 0.08) print "PASS_DIRECTIONAL_SCREEN__REQUEST_REAL_SHAPE_CONFIRM";
  else if (gain < 0.05) print "STOP_B2_BELOW_5_PERCENT";
  else print "GRAY_5_TO_8_PERCENT__MANAGER_DECISION";
}')

jq -n \
  --arg schema wrfgpu2.v0234.event-aware-fusion-short-result.v1 \
  --arg status "$classification" \
  --arg source_sha "$source_sha" \
  --arg nonce "$nonce" \
  --arg namespace "$run_root" \
  --arg proof_json "$proof_json" \
  --arg wall_json "$wall_json" \
  --argjson peak_vram_mib "$peak_vram" \
  --argjson minimum_free_vram_mib "$min_free_vram" \
  --argjson mean_gpu_util_percent "$mean_gpu_util" \
  --argjson sampled_peak_rss_kib "$sampled_peak_rss_kib" \
  --argjson sampled_peak_swap_kib "$sampled_peak_swap_kib" \
  --argjson time_peak_rss_kib "${time_peak_rss_kib:-0}" \
  --slurpfile proof "$proof_json" \
  --slurpfile wall "$wall_json" \
  '{schema:$schema,status:$status,source_sha:$source_sha,nonce:$nonce,
    namespace:$namespace,proofs:{screen:$proof_json,wall:$wall_json},
    scope:{fixture:"real operational all-seven minigrid",output:false,
      marker_callback:true,physics:false,aot:false,warm_in_process:true,
      repeats:5,k_values:[1,3],profiler:false,wrf_mpi:false},
    exactness:$proof[0].exactness,
    timing:$proof[0].timing,
    compile_and_cache:$proof[0].setup,
    resources:{peak_vram_mib:$peak_vram_mib,
      minimum_free_vram_mib:$minimum_free_vram_mib,
      mean_gpu_util_percent:$mean_gpu_util_percent,
      sampled_peak_rss_kib:$sampled_peak_rss_kib,
      sampled_peak_swap_kib:$sampled_peak_swap_kib,
      time_peak_rss_kib:$time_peak_rss_kib},
    command_wall_s:$wall[0].command_wall_s,
    claim_limit:"Directional small-grid scheduler screen only; not full-nine admission and not a profiler-backed final performance claim."}' \
  >"$summary_json"

echo "B2_EVENT_AWARE_SHORT_COMPLETE summary=$summary_json status=$classification gain=$gain"
