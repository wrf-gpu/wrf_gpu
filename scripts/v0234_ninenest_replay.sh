#!/usr/bin/env bash
# Exactly one manager-authorized all-seven-island max_dom=9 GPU replay.

set -euo pipefail

if [[ ( $# -lt 2 || $# -gt 4 ) || ! $1 =~ ^[0-9a-f]{64}$ || ! $2 =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <64-hex-nonce> <40-hex-authorized-head> [aot-seed-manifest.json] [sealed-input-stage]" >&2
  exit 64
fi

nonce=$1
authorized_head=$2
seed_manifest=${3:-}
replay_input_stage=${4:-}
cache_mode=cold_fresh
if [[ -n $seed_manifest ]]; then
  cache_mode=aot_seeded_warm
fi
input_mode=fresh_namespace_stage
if [[ -n $replay_input_stage ]]; then
  input_mode=sealed_aot_source_stage
  [[ -n $seed_manifest ]] || { echo "sealed input stage requires an AOT seed manifest" >&2; exit 64; }
fi
prefix=${nonce:0:16}
repo=<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-ninenest-replay
sprint=${repo}/.agent/sprints/2026-07-22-v0234-gpt-ninenest-replay
source_input=<DATA_ROOT>/alisios/registry/wrf_cases/payload/canary_all7/run
truth=<DATA_ROOT>/alisios/registry/wrf_cases/payload/canary_all7/run_cadvariant
wrf_root=<USER_HOME>/src/wrf_pristine/WRF
namespace=<DATA_ROOT>/wrf_gpu2/v0234_gpt_ninenest_replay_${prefix}
stage=${namespace}/input
output=${namespace}/gpu_output
proofs=${namespace}/pipeline_proofs
scratch=${namespace}/scratch
cache=${namespace}/jax_cache
monitor=${namespace}/monitor
preflight=${sprint}/STATIC_PREFLIGHT_${prefix}.json
authorization=${sprint}/GPU_AUTHORIZATION_${prefix}.json

if [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]]; then
  echo "refusing: canonical GPU lock is not held" >&2
  exit 65
fi
if [[ $(git -C "$repo" rev-parse HEAD) != "$authorized_head" ]]; then
  echo "refusing: HEAD differs from manager-authorized head" >&2
  exit 66
fi
if [[ -n $(git -C "$repo" status --short -- src/gpuwrf scripts/v0234_ninenest_replay.sh scripts/v0234_ninenest_static_preflight.py scripts/v0234_ninenest_score.py scripts/v0234_ninenest_aot_phase_audit.py scripts/v0234_ninenest_aot_seed_manifest.py) ]]; then
  echo "refusing: model or replay harness is dirty" >&2
  exit 66
fi
if [[ -e $namespace ]]; then
  echo "refusing non-fresh namespace: $namespace" >&2
  exit 67
fi
for required in "$preflight" "$authorization"; do
  [[ -f $required ]] || { echo "missing authority file: $required" >&2; exit 68; }
done
jq -e --arg nonce "$nonce" --arg head "$authorized_head" --arg ns "$namespace" \
  --arg cache_mode "$cache_mode" --arg input_mode "$input_mode" \
  '.verdict == "STATIC_PREFLIGHT_PASS" and .nonce == $nonce and
   .authorized_head == $head and .namespace == $ns and
   .configuration.cache_mode == $cache_mode and
   (.configuration.input_mode // "fresh_namespace_stage") == $input_mode and
   .configuration.hours == 1 and .configuration.max_dom == 9 and
   .configuration.expected_wrfout_count == 27 and
   .checks.namespace_absent == true and .checks.all_pinned_hashes_match == true and
   .checks.launcher_default_expression_present == true and
   .checks.nonce_not_consumed == true and .checks.nonce_absent_from_prior_sprints == true' \
  "$preflight" >/dev/null
jq -e --arg nonce "$nonce" --arg head "$authorized_head" --arg ns "$namespace" \
  --arg cache_mode "$cache_mode" --arg input_mode "$input_mode" \
  '.manager_pane == "0:1" and .nonce == $nonce and .authorized_head == $head and
   .namespace == $ns and .authorized_arm_count == 1 and
   .max_dom == 9 and .forecast_hours == 1 and .ratifies_v019_one_hour_gate == true and
   .cache_mode == $cache_mode and (.input_mode // "fresh_namespace_stage") == $input_mode' \
  "$authorization" >/dev/null

mkdir "$namespace"
mkdir "$output" "$proofs" "$scratch" "$cache" "$monitor"

if [[ $cache_mode == aot_seeded_warm ]]; then
  [[ -f $seed_manifest ]] || { echo "missing AOT seed manifest: $seed_manifest" >&2; exit 68; }
  seed_manifest_sha=$(sha256sum "$seed_manifest" | awk '{print $1}')
  jq -e --arg source_head "471daa439d0328efed485be653ceb0e218169fb2" \
    '.verdict == "AOT_SEED_MANIFEST_PASS" and .source_head == $source_head and
     .checks.exactly_three_aot_metas == true and .checks.exactly_six_seed_files == true and
     .checks.all_blob_hashes_match_authenticated_meta == true and
     .checks.source_gpuwrf_tree_matches_current == true and
     (.files | length) == 6' "$seed_manifest" >/dev/null
  jq -e --arg seed_sha "$seed_manifest_sha" \
    '.aot_seed_manifest_sha256 == $seed_sha and
     .aot_seed_manifest_canonical_payload_sha256 != null' "$authorization" >/dev/null

  seed_source=$(jq -r '.source_cache' "$seed_manifest")
  [[ -d $seed_source ]] || { echo "missing AOT seed cache: $seed_source" >&2; exit 68; }
  seed_rows=${namespace}/aot_seed_copy_rows.jsonl
  : >"$seed_rows"
  while IFS=$'\t' read -r relative expected bytes; do
    if [[ ! $relative =~ ^aot/[A-Za-z0-9._/-]+$ || $relative == *..* ]]; then
      echo "unsafe AOT seed relative path: $relative" >&2
      exit 68
    fi
    source_path=${seed_source}/${relative}
    destination_path=${cache}/${relative}
    [[ -f $source_path ]] || { echo "missing AOT seed file: $source_path" >&2; exit 68; }
    [[ $(stat -c %s "$source_path") == "$bytes" ]] || { echo "AOT seed size mismatch: $source_path" >&2; exit 68; }
    observed_source=$(sha256sum "$source_path" | awk '{print $1}')
    [[ $observed_source == "$expected" ]] || { echo "AOT seed source hash mismatch: $source_path" >&2; exit 68; }
    mkdir -p "$(dirname "$destination_path")"
    cp --reflink=auto --preserve=mode,timestamps "$source_path" "$destination_path"
    observed_destination=$(sha256sum "$destination_path" | awk '{print $1}')
    [[ $observed_destination == "$expected" ]] || { echo "AOT seed destination hash mismatch: $destination_path" >&2; exit 68; }
    jq -cn --arg relative "$relative" --arg sha "$expected" --argjson bytes "$bytes" \
      '{relative_path:$relative,bytes:$bytes,sha256:$sha,source_verified:true,destination_verified:true}' \
      >>"$seed_rows"
  done < <(jq -r '.files[] | [.relative_path,.sha256,(.bytes|tostring)] | @tsv' "$seed_manifest")

  jq -s --arg seed_manifest "$seed_manifest" --arg seed_manifest_sha "$seed_manifest_sha" \
    --arg source_cache "$seed_source" --arg destination_cache "$cache" \
    '{schema:"wrfgpu2.v0234.ninenest-aot-seed-receipt.v1",seed_manifest:$seed_manifest,
      seed_manifest_sha256:$seed_manifest_sha,source_cache:$source_cache,
      destination_cache:$destination_cache,files:.,all_source_hashes_verified:true,
      all_destination_hashes_verified:true}' "$seed_rows" >"${namespace}/aot_seed_receipt.json"
  rm "$seed_rows"
fi

if [[ $input_mode == sealed_aot_source_stage ]]; then
  source_namespace=$(jq -r '.source_namespace' "$seed_manifest")
  expected_input=${source_namespace}/input
  expected_manifest=${source_namespace}/stage_manifest.json
  [[ -d $replay_input_stage && -f $expected_manifest ]] || {
    echo "missing sealed source input stage or manifest" >&2
    exit 68
  }
  [[ $(readlink -f "$replay_input_stage") == $(readlink -f "$expected_input") ]] || {
    echo "sealed input stage does not match AOT source provenance" >&2
    exit 68
  }
  expected_manifest_sha=$(sha256sum "$expected_manifest" | awk '{print $1}')
  jq -e --arg stage "$(readlink -f "$expected_input")" --arg sha "$expected_manifest_sha" \
    '.replay_input_stage == $stage and .replay_input_stage_manifest_sha256 == $sha and
     .namespace_input_is_symlink == true' "$authorization" >/dev/null
  ln -s "$(readlink -f "$expected_input")" "$stage"
else
  mkdir "$stage"
  ln -s "$truth/namelist.input" "$stage/namelist.input"
  ln -s "$source_input/wrfbdy_d01" "$stage/wrfbdy_d01"
  for domain in 01 02 03 04 05 06 07 08 09; do
    ln -s "$source_input/wrfinput_d${domain}" "$stage/wrfinput_d${domain}"
  done
fi

python - "$stage" "$namespace/stage_manifest.json" "$input_mode" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
stage, out = map(Path, sys.argv[1:3])
input_mode = sys.argv[3]
rows = []
for path in sorted(stage.iterdir()):
    target = path.resolve(strict=True)
    h = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    rows.append({"name": path.name, "symlink": os.readlink(path), "resolved": str(target), "bytes": target.stat().st_size, "sha256": h.hexdigest()})
payload = {"schema": "wrfgpu2.v0234.ninenest-stage-manifest.v1", "read_only_symlink_stage": True,
           "input_mode": input_mode, "stage_argument": str(stage),
           "stage_resolved": str(stage.resolve(strict=True)), "files": rows}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["canonical_payload_sha256"] = hashlib.sha256(raw).hexdigest()
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

# Exercise production defaults. Only the repaired boundary variable is called
# out separately because its absence is itself an acceptance condition.
unset JAX_PLATFORM_NAME JAX_PLATFORMS XLA_FLAGS XLA_PYTHON_CLIENT_ALLOCATOR
unset GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE GPUWRF_NESTED_FUSE
unset GPUWRF_NESTED_DEFUSE_COMPILE GPUWRF_BITWISE GPUWRF_NESTED_AOT
unset GPUWRF_AOT_VERIFY GPUWRF_PREPARED_RUNTIME_REUSE GPUWRF_NESTED_ASYNC_OUTPUT
unset GPUWRF_NEST_OUTPUT_PIPELINE GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH
unset GPUWRF_NESTED_EVENT_AWARE_FUSION_K GPUWRF_NESTED_SYNC_MODE
unset GPUWRF_TRAINING_OUTPUT_SUBSET GPUWRF_FULL_WRFOUT_VARIABLES GPUWRF_FULL_WRFOUT
unset GPUWRF_BATCH_ENSEMBLE GPUWRF_BATCH_INPUT_DIRS GPUWRF_MYNN_BOULAC_ONZ
unset GPUWRF_MYNN_BOULAC_FP32 GPUWRF_ACOUSTIC_PRECISION_MODE
unset GPUWRF_GWD_NESTED GPUWRF_EMIT_INITIAL_HISTORY

export PYTHONPATH=${repo}/src
export GPUWRF_WRF_ROOT=$wrf_root
export GPUWRF_ALLOCATOR=cuda_async
export GPUWRF_FINITE_CHECK=1
export GPUWRF_FORCE_FP64=1
export GPUWRF_NEST_PERF_TIMERS=1
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=$cache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -l 1 >"$monitor/nvidia_smi.csv" &
sampler_pid=$!
stop_sampler() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap stop_sampler EXIT

start_ns=$(date +%s%N)
set +e
timeout --foreground --signal=INT --kill-after=60s 7200s \
  /usr/bin/time -v -o "$monitor/time_v.txt" \
  <USER_HOME>/miniconda3/bin/python -m gpuwrf run \
    --input-dir "$stage" \
    --output-dir "$output" \
    --proof-dir "$proofs" \
    --scratch-dir "$scratch" \
    --max-dom 9 \
    --hours 1 \
    >"$monitor/run.stdout.log" 2>"$monitor/run.stderr.log"
rc=$?
set -e
end_ns=$(date +%s%N)
stop_sampler
trap - EXIT

jq -n --argjson rc "$rc" --argjson start "$start_ns" --argjson end "$end_ns" \
  --arg nonce "$nonce" --arg namespace "$namespace" \
  '{schema:"wrfgpu2.v0234.ninenest-arm-wall.v1",returncode:$rc,nonce:$nonce,
    namespace:$namespace,start_epoch_ns:$start,end_epoch_ns:$end,
    command_wall_s:(($end-$start)/1000000000),model_timeout_s:7200,kill_grace_s:60}' \
  >"$monitor/arm_wall.json"

if [[ $rc -ne 0 ]]; then
  echo "nine-nest replay failed or timed out rc=$rc namespace=$namespace" >&2
  exit "$rc"
fi

if [[ $cache_mode == aot_seeded_warm ]]; then
  grep -F '[gpuwrf:nested-aot] domain=d01 loaded=true source=aot_blob' "$monitor/run.stderr.log" >/dev/null
  while IFS= read -r cheap_key; do
    grep -F "[gpuwrf:nested-aot] domain=fused/d02 loaded=true source=aot_blob" "$monitor/run.stderr.log" \
      | grep -F "cheap_key=${cheap_key:0:12}" >/dev/null
  done < <(jq -r '.files[] | select(.kind == "aot_meta" and (.relative_path | contains("/fused_d02/"))) | .cheap_key' "$seed_manifest")
  if grep -Eq 'fallback:.*(compiled|compile-exception|cached-call-error)|Compiling module jit_(_advance_chunk_fori|fused_jit)' "$monitor/run.stderr.log"; then
    echo "warm AOT replay unexpectedly compiled or hit a cached-call error" >&2
    exit 69
  fi
fi

pipeline=${proofs}/nested_pipeline_run.json
jq -e \
  '.verdict == "PIPELINE_GREEN" and .hours == 1 and .max_dom == 9 and
   .domains == ["d01","d02","d03","d04","d05","d06","d07","d08","d09"] and
   .all_domains_finite == true and .all_outputs_present == true and
   ([.per_domain[] | .history_interval_min] | all(. == 20)) and
   ([.per_domain[] | .wrfout_count] | all(. == 3)) and
   ([.per_domain[] | .expected_wrfout_count] | all(. == 3)) and
   .metadata.domains.d01.namelist.nested_frozen_wrf_boundary_bundle == false and
   ([.metadata.domains.d02.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d03.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d04.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d05.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d06.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d07.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d08.namelist.nested_frozen_wrf_boundary_bundle,
     .metadata.domains.d09.namelist.nested_frozen_wrf_boundary_bundle] | all(. == true)) and
   .metadata.nested_runtime.prepared_runtime_reuse == true and
   .metadata.parallel_compile.nested_precompile.source == "skip:fused-default" and
   .metadata.nested_aot.enabled == true and
   .per_domain.d01.own_steps == 200 and .per_domain.d02.own_steps == 600 and
   ([.per_domain.d03.own_steps,.per_domain.d04.own_steps,.per_domain.d05.own_steps,
     .per_domain.d06.own_steps,.per_domain.d07.own_steps,.per_domain.d08.own_steps,
     .per_domain.d09.own_steps] | all(. == 1800)) and
   .metadata.domains.d01.namelist.radiation_cadence_steps == 100 and
   .metadata.domains.d02.namelist.radiation_cadence_steps == 300 and
   ([.metadata.domains.d03.namelist.radiation_cadence_steps,
     .metadata.domains.d04.namelist.radiation_cadence_steps,
     .metadata.domains.d05.namelist.radiation_cadence_steps,
     .metadata.domains.d06.namelist.radiation_cadence_steps,
     .metadata.domains.d07.namelist.radiation_cadence_steps,
     .metadata.domains.d08.namelist.radiation_cadence_steps,
     .metadata.domains.d09.namelist.radiation_cadence_steps] | all(. == 900)) and
   ([.metadata.domains[].namelist.gwd_opt] | all(. == 0))' "$pipeline" >/dev/null

artifacts=("$pipeline" "$namespace/stage_manifest.json")
if [[ $cache_mode == aot_seeded_warm ]]; then
  artifacts+=("$namespace/aot_seed_receipt.json" "$seed_manifest")
fi
sha256sum "${artifacts[@]}" >"$namespace/gpu_arm_artifact_hashes.txt"
echo "NINE_NEST_GPU_TRAJECTORY_COMPLETE namespace=$namespace pipeline=$pipeline"
