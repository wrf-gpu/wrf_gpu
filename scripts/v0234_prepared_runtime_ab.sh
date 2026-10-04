#!/usr/bin/env bash
# Controlled v0.23.4 three-hour prepared-runtime A/B.
#
# Plan mode is CPU-only and is the default. Execution is fail-closed until the
# committed benchmark manifest carries a PRE_ARM_READY selection revision. The
# whole A/B must be wrapped by scripts/with_gpu_lock.sh so no other GPU job can
# enter between Arm A and Arm B.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/.agent/decisions/V0234_BENCHMARK_MANIFEST.json"
MODE="${1:---plan}"
RECORD='.prepared_runtime_ab_prearm'
NVIDIA_SMI_PID=0
PROCESS_SAMPLE_PID=0
PIDSTAT_PID=0
PROFILE_PID=0
ACTIVE_ARM_DIR=""

q() {
  jq -er "$RECORD$1" "$MANIFEST"
}

die() {
  echo "v0234-prepared-ab: $*" >&2
  exit 2
}

inventory_sha256() {
  local root="$1"
  [[ -d "$root" ]] || die "cache root is missing: $root"
  (
    cd "$root"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

source_path_set_sha256() {
  local -a paths=()
  mapfile -t paths < <(jq -r "$RECORD.code.source_paths[]" "$MANIFEST")
  (
    cd "$ROOT"
    for path in "${paths[@]}"; do
      tracked="$(git ls-files -- "$path")"
      [[ -n "$tracked" ]] || die "source path has no tracked files: $path"
      printf '%s\n' "$tracked"
    done | LC_ALL=C sort -u | while IFS= read -r path; do
      printf '%s  %s\n' "$(sha256sum "$path" | awk '{print $1}')" "$path"
    done
  ) | sha256sum | awk '{print $1}'
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

validate_record_shape() {
  jq -e "$RECORD" "$MANIFEST" >/dev/null || die "manifest has no prepared A/B record"
  if jq -e "$RECORD | .. | strings | select(test(\"TBD\"; \"i\"))" "$MANIFEST" >/dev/null; then
    die "prepared A/B selection contains a forbidden TBD string"
  fi
  if jq -e "$RECORD | .. | select(. == null)" "$MANIFEST" >/dev/null; then
    die "prepared A/B selection contains null; use an explicit blocked status"
  fi
}

print_plan() {
  validate_record_shape
  local output_root selection_status expected
  output_root="$(q '.output_root')"
  selection_status="$(q '.selection_status')"
  expected="$(q '.expected_duration.classification'): lock $(q '.expected_duration.lock_wall_s.min')-$(q '.expected_duration.lock_wall_s.max') s; total $(q '.expected_duration.total_wall_s.min')-$(q '.expected_duration.total_wall_s.max') s"
  jq -n \
    --arg status "$selection_status" \
    --arg manifest "$MANIFEST" \
    --arg command "/usr/bin/taskset -c 12-15 $ROOT/scripts/with_gpu_lock.sh --timeout 10800 --label v0234-prepared-runtime-ab -- /usr/bin/taskset -c 12-15 $ROOT/scripts/v0234_prepared_runtime_ab.sh --execute" \
    --arg output_root "$output_root" \
    --arg expected "$expected" \
    '{selection_status:$status, selection_manifest:$manifest,
      exact_lock_wrapped_command:$command, output_root:$output_root,
      expected_duration:$expected,
      note:"Execution remains prohibited unless selection_status is PRE_ARM_READY and all frozen hashes verify."}'
}

start_resource_samplers() {
  local arm_dir="$1" pid_file="$2"
  nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits -l 1 >"$arm_dir/nvidia_smi.csv" 2>"$arm_dir/nvidia_smi.stderr" &
  NVIDIA_SMI_PID=$!

  (
    while [[ ! -s "$pid_file" ]]; do sleep 0.2; done
    local pid
    pid="$(<"$pid_file")"
    printf 'epoch_s\tpid\tvmrss_kib\tvmswap_kib\tmemavailable_kib\n'
    while [[ -r "/proc/$pid/status" ]]; do
      local rss swap available
      rss="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")"
      swap="$(awk '/^VmSwap:/ {print $2}' "/proc/$pid/status")"
      available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$pid" "${rss:-0}" "${swap:-0}" "${available:-0}"
      sleep 1
    done
  ) >"$arm_dir/process_resources.tsv" &
  PROCESS_SAMPLE_PID=$!

  (
    while [[ ! -s "$pid_file" ]]; do sleep 0.2; done
    exec pidstat -t -rudw -h -p "$(<"$pid_file")" 1
  ) >"$arm_dir/pidstat.log" 2>"$arm_dir/pidstat.stderr" &
  PIDSTAT_PID=$!
}

stop_resource_samplers() {
  for pid in "$NVIDIA_SMI_PID" "$PROCESS_SAMPLE_PID" "$PIDSTAT_PID"; do
    if [[ "$pid" -gt 0 ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  NVIDIA_SMI_PID=0
  PROCESS_SAMPLE_PID=0
  PIDSTAT_PID=0
}

invalidate_preempted_arm() {
  local signal="$1"
  trap - HUP INT TERM
  if [[ -n "$ACTIVE_ARM_DIR" ]]; then
    jq -n --arg signal "$signal" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{status:"INVALID_PREEMPTED",signal:$signal,at_utc:$at,resume_forbidden:true}' \
      >"$ACTIVE_ARM_DIR/INVALID_PREEMPTED.json"
  fi
  if [[ "$PROFILE_PID" -gt 0 ]] && kill -0 "$PROFILE_PID" 2>/dev/null; then
    kill -TERM -- "-$PROFILE_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$PROFILE_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PROFILE_PID" 2>/dev/null; then
      kill -KILL -- "-$PROFILE_PID" 2>/dev/null || true
    fi
    wait "$PROFILE_PID" 2>/dev/null || true
  fi
  PROFILE_PID=0
  stop_resource_samplers
  exit 143
}

quick_arm_gate() {
  local arm="$1" arm_dir="$2" expected_loads="$3"
  local proof="$arm_dir/proof/nested_pipeline_run.json"
  [[ -s "$proof" ]] || die "Arm $arm has no nested pipeline proof"
  jq -e \
    --argjson loads "$expected_loads" \
    '.verdict == "PIPELINE_GREEN" and .all_domains_finite == true and
     .all_outputs_present == true and
     .metadata.nested_aot.load_count == $loads' "$proof" >/dev/null \
    || die "Arm $arm failed finite/output/load-count gate (expected $expected_loads loads)"
  local logged_loads
  logged_loads="$(awk '/real_load=true/ {count += 1} END {print count + 0}' "$arm_dir/run.stderr")"
  [[ "$logged_loads" -eq "$expected_loads" ]] \
    || die "Arm $arm log has $logged_loads real loads; expected $expected_loads"

  local peak_rss peak_swap min_available peak_vram
  peak_rss="$(awk 'NR>1 {v=$3+0; if (v>m) m=v} END {print m+0}' "$arm_dir/process_resources.tsv")"
  peak_swap="$(awk 'NR>1 {v=$4+0; if (v>m) m=v} END {print m+0}' "$arm_dir/process_resources.tsv")"
  min_available="$(awk 'NR>1 {v=$5+0; if (!seen || v<m) m=v; seen=1} END {print m+0}' "$arm_dir/process_resources.tsv")"
  peak_vram="$(awk -F, '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}' "$arm_dir/nvidia_smi.csv")"
  [[ "$peak_rss" -le "$(q '.resource_gates.maximum_process_rss_kib')" ]] || die "Arm $arm exceeded RSS gate"
  [[ "$peak_swap" -le "$(q '.resource_gates.maximum_process_swap_kib')" ]] || die "Arm $arm exceeded swap gate"
  [[ "$min_available" -ge "$(q '.resource_gates.minimum_host_available_kib')" ]] || die "Arm $arm violated host-available gate"
  [[ "$peak_vram" -le "$(q '.resource_gates.maximum_gpu_used_mib')" ]] || die "Arm $arm exceeded VRAM gate"
}

required_subset_gate() {
  local python_bin="$1" arm_dir="$2"
  "$python_bin" - "$arm_dir/out" "$arm_dir/required_subset.json" <<'PY'
from __future__ import annotations
import glob
import json
from pathlib import Path
import sys
from netCDF4 import Dataset

root, output = sys.argv[1:]
required = {"U10", "V10", "T2", "Q2", "PSFC", "TSK", "HGT", "LANDMASK", "XLAT", "XLONG", "CLDFRA", "PH", "PHB"}
files = sorted(glob.glob(str(Path(root) / "**" / "wrfout_d0*"), recursive=True))
missing = {}
for path in files:
    with Dataset(path) as dataset:
        absent = sorted(required.difference(dataset.variables))
    if absent:
        missing[str(Path(path).relative_to(root))] = absent
payload = {"file_count": len(files), "required": sorted(required), "missing": missing,
           "verdict": "PASS" if files and not missing else "FAIL"}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["verdict"] == "PASS" else 1)
PY
}

run_arm() {
  local arm="$1" reuse="$2" expected_loads="$3" output_root="$4" input_root="$5" python_bin="$6" cpuset="$7"
  local arm_dir="$output_root/arm_${arm,,}" pid_file cache_root cache_before cache_expected
  arm_dir="$output_root/arm_${arm,,}"
  mkdir "$arm_dir" "$arm_dir/out" "$arm_dir/proof" "$arm_dir/scratch"
  pid_file="$arm_dir/python.pid"
  cache_root="$(q ".cache.arm_${arm,,}_root")"
  cache_expected="$(q ".cache.arm_${arm,,}_inventory_sha256")"
  cache_before="$(inventory_sha256 "$cache_root")"
  [[ "$cache_before" == "$cache_expected" ]] || die "Arm $arm cache inventory differs before launch"

  local -a env_args=()
  while IFS= read -r kv; do env_args+=("$kv"); done < <(
    jq -r "$RECORD.environment | to_entries[] | \"\\(.key)=\\(.value)\"" "$MANIFEST"
  )
  env_args+=(
    "GPUWRF_PREPARED_RUNTIME_REUSE=$reuse"
    "GPUWRF_JAX_CACHE_DIR=$cache_root"
    "JAX_COMPILATION_CACHE_DIR=$cache_root"
  )

  local -a model_cmd=(
    taskset -c "$cpuset" env -i "${env_args[@]}"
    "$python_bin" -m gpuwrf.cli run
    --input-dir "$input_root"
    --output-dir "$arm_dir/out"
    --proof-dir "$arm_dir/proof"
    --namelist "$input_root/namelist.input"
    --domains-from-namelist
    --hours 3
    --scratch-dir "$arm_dir/scratch"
  )
  local -a profile_cmd=(
    "$(q '.toolchain.nsys')" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
    --force-overwrite=true --output="$arm_dir/nsys_real1km_3h"
    bash -c 'echo "$$" > "$1"; shift; exec "$@"' _ "$pid_file"
    "${model_cmd[@]}"
  )
  printf '%q ' "${profile_cmd[@]}" >"$arm_dir/command.txt"
  printf '\n' >>"$arm_dir/command.txt"
  printf '%s\n' "${env_args[@]}" | LC_ALL=C sort >"$arm_dir/environment.txt"

  start_resource_samplers "$arm_dir" "$pid_file"
  local start_ns end_ns rc
  start_ns="$(date +%s%N)"
  ACTIVE_ARM_DIR="$arm_dir"
  set +e
  setsid "${profile_cmd[@]}" >"$arm_dir/run.stdout" 2>"$arm_dir/run.stderr" &
  PROFILE_PID=$!
  wait "$PROFILE_PID"
  rc=$?
  set -e
  PROFILE_PID=0
  ACTIVE_ARM_DIR=""
  end_ns="$(date +%s%N)"
  stop_resource_samplers
  jq -n --argjson rc "$rc" --argjson start_ns "$start_ns" --argjson end_ns "$end_ns" \
    '{returncode:$rc,start_epoch_ns:$start_ns,end_epoch_ns:$end_ns,
      command_wall_s:(($end_ns-$start_ns)/1000000000)}' >"$arm_dir/wall.json"
  [[ "$rc" -eq 0 ]] || die "Arm $arm failed rc=$rc"

  find "$arm_dir/out" -type f -name 'wrfout_d0*' -printf '%P\t%T@\t%s\n' | LC_ALL=C sort >"$arm_dir/wrfout_mtimes.tsv"
  (
    cd "$arm_dir/out"
    find . -type f -name 'wrfout_d0*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) >"$arm_dir/wrfout_sha256.txt"
  "$(q '.toolchain.nsys')" stats --force-overwrite=true --force-export=true \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum \
    --format csv --output "$arm_dir/nsys_stats" "$arm_dir/nsys_real1km_3h.nsys-rep" \
    >"$arm_dir/nsys_stats.stdout" 2>"$arm_dir/nsys_stats.stderr"
  "$python_bin" "$ROOT/scripts/m7_nsys_stats_extract.py" \
    "$arm_dir/nsys_real1km_3h.nsys-rep" --output "$arm_dir/nsys_summary.json"

  quick_arm_gate "$arm" "$arm_dir" "$expected_loads"
  required_subset_gate "$python_bin" "$arm_dir"
  [[ "$(inventory_sha256 "$cache_root")" == "$cache_before" ]] || die "Arm $arm mutated its immutable cache"
}

execute_ab() {
  validate_record_shape
  [[ "$(q '.selection_status')" == "PRE_ARM_READY" ]] || die "pre-arm selection is not ready"
  [[ "${GPUWRF_GPU_LOCK_HELD:-0}" == "1" ]] || die "execute through scripts/with_gpu_lock.sh"
  [[ "${GPUWRF_GPU_LOCK_LABEL:-}" == "v0234-prepared-runtime-ab" ]] || die "unexpected GPU lock label"

  local cpuset allowed_cpus
  cpuset="$(q '.cpuset')"
  allowed_cpus="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
  [[ "$allowed_cpus" == "$cpuset" ]] || die "runner affinity is $allowed_cpus; expected $cpuset"
  export PATH="$(q '.environment.PATH')"

  local source_commit source_tree actual_source_tree source_hash expected_source_hash
  source_commit="$(q '.code.source_commit')"
  source_tree="$(q '.code.source_tree')"
  git -C "$ROOT" cat-file -e "$source_commit^{commit}" 2>/dev/null || die "frozen source commit does not exist"
  actual_source_tree="$(git -C "$ROOT" rev-parse "$source_commit^{tree}")"
  [[ "$actual_source_tree" == "$source_tree" ]] || die "frozen source tree differs"
  git -C "$ROOT" merge-base --is-ancestor "$source_commit" HEAD || die "selection commit does not descend from source commit"
  while IFS= read -r path; do
    jq -e --arg path "$path" "$RECORD.code.selection_commit_allowlist | index($path) != null" "$MANIFEST" >/dev/null \
      || die "selection commit changed non-allowlisted path: $path"
  done < <(git -C "$ROOT" diff --name-only "$source_commit"..HEAD)
  source_hash="$(source_path_set_sha256)"
  expected_source_hash="$(q '.code.source_path_set_sha256')"
  [[ "$source_hash" == "$expected_source_hash" ]] || die "runtime source path-set hash differs"
  [[ -z "$(git -C "$ROOT" status --porcelain)" ]] || die "worktree is not clean"

  local input_root output_root python_bin
  input_root="$(q '.fixture.input_root')"
  output_root="$(q '.output_root')"
  python_bin="$(q '.toolchain.python')"
  [[ ! -e "$output_root" ]] || die "output root already exists: $output_root"
  [[ "$(q '.cache.arm_a_root')" != "$(q '.cache.arm_b_root')" ]] || die "arms must use distinct immutable cache copies"
  [[ "$(q '.cache.arm_a_inventory_sha256')" == "$(q '.cache.arm_b_inventory_sha256')" ]] \
    || die "Arm A and Arm B cache inventories are not byte-equivalent"
  for tool in jq sha256sum nvidia-smi pidstat taskset setsid; do command -v "$tool" >/dev/null || die "missing tool: $tool"; done
  [[ "$(file_sha256 "$python_bin")" == "$(q '.toolchain.python_sha256')" ]] || die "Python binary hash differs"
  [[ "$(file_sha256 "$(q '.toolchain.nsys')")" == "$(q '.toolchain.nsys_sha256')" ]] || die "Nsight binary hash differs"
  [[ "$($python_bin --version 2>&1)" == "Python $(q '.toolchain.python_version')" ]] || die "Python version differs"
  [[ "$("$(q '.toolchain.nsys')" --version | awk 'NR==1 {print $5}')" == "$(q '.toolchain.nsys_version')" ]] || die "Nsight version differs"
  local versions
  versions="$(env -i PATH="$(q '.environment.PATH')" HOME="$(q '.environment.HOME')" PYTHONPATH="$(q '.environment.PYTHONPATH')" "$python_bin" -c 'import jax,jaxlib; print(jax.__version__ + " " + jaxlib.__version__)')"
  [[ "$versions" == "$(q '.toolchain.jax_version') $(q '.toolchain.jaxlib_version')" ]] || die "JAX toolchain differs"
  while IFS=$'\t' read -r rel expected; do
    [[ "$(file_sha256 "$input_root/$rel")" == "$expected" ]] || die "fixture hash mismatch: $rel"
  done < <(jq -r "$RECORD.fixture.files[] | [.path,.sha256] | @tsv" "$MANIFEST")

  mkdir "$output_root"
  cp "$MANIFEST" "$output_root/selection_manifest.json"
  jq -n --arg selection_commit "$(git -C "$ROOT" rev-parse HEAD)" \
    --arg selection_tree "$(git -C "$ROOT" rev-parse HEAD^{tree})" \
    --arg source_commit "$source_commit" --arg source_tree "$source_tree" \
    --arg source_path_set_sha256 "$source_hash" \
    '{selection_commit_at_launch:$selection_commit,selection_tree_at_launch:$selection_tree,
      source_commit:$source_commit,source_tree:$source_tree,
      source_path_set_sha256:$source_path_set_sha256}' >"$output_root/code_identity.json"
  trap 'invalidate_preempted_arm HUP' HUP
  trap 'invalidate_preempted_arm INT' INT
  trap 'invalidate_preempted_arm TERM' TERM
  run_arm A 0 27 "$output_root" "$input_root" "$python_bin" "$cpuset"
  run_arm B 1 9 "$output_root" "$input_root" "$python_bin" "$cpuset"

  "$python_bin" "$ROOT/scripts/v020_wrfout_byte_compare.py" \
    "$output_root/arm_a/out" "$output_root/arm_b/out" --label-a A --label-b B \
    >"$output_root/wrfout_variable_identity.txt"
  find "$output_root/arm_a/out" -type f -name 'wrfout_d0*' -printf '%P\n' | LC_ALL=C sort >"$output_root/arm_a/wrfout_paths.txt"
  find "$output_root/arm_b/out" -type f -name 'wrfout_d0*' -printf '%P\n' | LC_ALL=C sort >"$output_root/arm_b/wrfout_paths.txt"
  diff -u "$output_root/arm_a/wrfout_paths.txt" "$output_root/arm_b/wrfout_paths.txt" >"$output_root/wrfout_paths.diff"
  diff -u "$output_root/arm_a/wrfout_sha256.txt" "$output_root/arm_b/wrfout_sha256.txt" \
    >"$output_root/wrfout_file_sha256.diff" || true
  "$python_bin" "$ROOT/scripts/v0234_prepared_runtime_ab_analyze.py" "$output_root" \
    --manifest "$MANIFEST" --output "$output_root/automated_ab_analysis.json"
  echo "A/B capture complete; profiler transfer-window and segment-growth review remain manager gates." >"$output_root/POST_CAPTURE_REVIEW_REQUIRED"
  trap - HUP INT TERM
}

case "$MODE" in
  --plan) print_plan ;;
  --execute) execute_ab ;;
  *) die "usage: $0 [--plan|--execute]" ;;
esac
