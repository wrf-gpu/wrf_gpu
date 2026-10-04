#!/usr/bin/env bash
# Shared helpers for v0.23 close validation scripts.
#
# Source this file from the gate scripts. It centralizes CPU/GPU env, dry-run
# behavior, git-worktree paired runs, GPU locking, and metric capture. Every
# command that can touch CUDA is routed through scripts/with_gpu_lock.sh unless
# V023_DRYRUN=1.

set -uo pipefail

V023_COMMON="${BASH_SOURCE[0]}"
V023_VALIDATION_DIR="$(cd "$(dirname "$V023_COMMON")" && pwd)"
V023_ROOT="$(cd "$V023_VALIDATION_DIR/../../.." && pwd)"
V023_GPU_LOCK="$V023_ROOT/scripts/with_gpu_lock.sh"
V023_FIELD_COMPARE="$V023_VALIDATION_DIR/paired_baseline_field_compare.py"

: "${BASE_REF:=v0.22.2}"
: "${HEAD:=HEAD}"
: "${CASE:=<DATA_ROOT>/wrf_downscale/canary_all7/run}"
: "${HOURS:=1}"
: "${MAX_DOM:=3}"
: "${DOMAIN:=d01}"
: "${DOMAINS:=d01 d02 d03}"
: "${V023_TASKSET:=4-31}"
: "${V023_OMP:=4}"
: "${V023_DRYRUN:=0}"
: "${V023_LOCK_TIMEOUT:=21600}"
: "${V023_WORK_ROOT:=<DATA_ROOT>/wrf_gpu_validation/v023_validation}"
: "${V023_RESULTS_DIR:=$V023_VALIDATION_DIR/results}"
: "${V023_JAX_CACHE_ROOT:=<DATA_ROOT>/gpuwrf_jax_cache/v023_validation}"
: "${V023_AUTOTUNE_FLOOR:=1e-8}"
: "${V023_PROFILE_NSYS:=0}"
: "${V023_FORCE_GPU_RUN:=0}"
: "${V023_GPUWRF_ROOT:=<DATA_ROOT>/canairy_meteo}"
# Pristine WRF root that provides the physics .TBL / data lookup files. Fresh git
# worktrees do not carry the gitignored data/wrf_pristine symlink, so every gpuwrf
# forecast in the harness must be told where the real WRF run/ tables live (honored
# via GPUWRF_WRF_ROOT; see wrf_gpu2 memory item #115).
: "${V023_WRF_ROOT:=<USER_HOME>/src/wrf_pristine/WRF}"
: "${V023_KEEP_WORKTREES:=1}"
: "${V023_FAIL_ON_PERF_REGRESSION:=1}"
: "${V023_PERF_WALL_TOL:=1.05}"

export V023_COMMON V023_ROOT V023_VALIDATION_DIR V023_GPU_LOCK V023_FIELD_COMPARE
export V023_FAIL_ON_PERF_REGRESSION V023_PERF_WALL_TOL

v023_log() {
  printf '[v023-validation][%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2
}

v023_usage_common() {
  cat <<EOF
Common env:
  CASE=$CASE
  BASE_REF=$BASE_REF
  HEAD=$HEAD
  HOURS=$HOURS
  MAX_DOM=$MAX_DOM
  DOMAINS="$DOMAINS"
  V023_DRYRUN=$V023_DRYRUN
  V023_TASKSET=$V023_TASKSET
  V023_WORK_ROOT=$V023_WORK_ROOT
  V023_RESULTS_DIR=$V023_RESULTS_DIR
  V023_AUTOTUNE_FLOOR=$V023_AUTOTUNE_FLOOR
EOF
}

v023_parse_common_flag() {
  case "${1:-}" in
    --dry-run)
      V023_DRYRUN=1
      export V023_DRYRUN
      return 0
      ;;
    --help|-h)
      return 2
      ;;
    *)
      return 1
      ;;
  esac
}

v023_require_file() {
  local path="${1:?path required}"
  if [[ ! -f "$path" ]]; then
    v023_log "missing file: $path"
    return 1
  fi
}

v023_require_dir() {
  local path="${1:?path required}"
  if [[ ! -d "$path" ]]; then
    v023_log "missing directory: $path"
    return 1
  fi
}

v023_require_ref() {
  local ref="${1:?ref required}"
  git -C "$V023_ROOT" rev-parse --verify "${ref}^{commit}" >/dev/null
}

v023_gate_paths() {
  local gate="${1:?gate required}"
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  V023_GATE="$gate"
  V023_RUN_DIR="${V023_RUN_DIR:-$V023_WORK_ROOT/${gate}_${ts}}"
  V023_VERDICT_JSON="${V023_VERDICT_JSON:-$V023_RESULTS_DIR/${gate}_${ts}.json}"
  V023_VERDICT_MD="${V023_VERDICT_MD:-$V023_RESULTS_DIR/${gate}_${ts}.md}"
  mkdir -p "$V023_RUN_DIR" "$V023_RESULTS_DIR"
  export V023_GATE V023_RUN_DIR V023_VERDICT_JSON V023_VERDICT_MD
}

v023_cpu_prefix() {
  printf 'taskset -c %q env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=%q PYTHONPATH=%q' \
    "$V023_TASKSET" "$V023_OMP" "$V023_ROOT/src"
}

v023_write_status_json() {
  local out="${1:?out required}"
  local gate="${2:?gate required}"
  local verdict="${3:?verdict required}"
  local status="${4:?status required}"
  local message="${5:-}"
  python3 - "$out" "$gate" "$verdict" "$status" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out, gate, verdict, status, message = sys.argv[1:6]
payload = {
    "schema": "v023-validation-gate-status-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": gate,
    "verdict": verdict,
    "status": status,
    "message": message,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"gate": gate, "verdict": verdict, "output": out}, sort_keys=True))
PY
}

v023_json_combine_pair_gate() {
  local out="${1:?out required}"
  local gate="${2:?gate required}"
  local compare_json="${3:?compare json required}"
  local base_metrics="${4:?base metrics required}"
  local cand_metrics="${5:?candidate metrics required}"
  local acceptance="${6:?acceptance required}"
  python3 - "$out" "$gate" "$compare_json" "$base_metrics" "$cand_metrics" "$acceptance" <<'PY'
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out, gate, compare_path, base_metrics_path, cand_metrics_path, acceptance = sys.argv[1:7]
compare = json.loads(Path(compare_path).read_text())
base_metrics = json.loads(Path(base_metrics_path).read_text())
cand_metrics = json.loads(Path(cand_metrics_path).read_text())

failures = []
if compare.get("verdict") != "PASS":
    failures.append({"kind": "field_compare", "verdict": compare.get("verdict"), "path": compare_path})
for label, metrics in (("baseline", base_metrics), ("candidate", cand_metrics)):
    if int(metrics.get("returncode", 999)) != 0:
        failures.append({"kind": "run_returncode", "label": label, "returncode": metrics.get("returncode")})

base_wall = base_metrics.get("wall_seconds")
cand_wall = cand_metrics.get("wall_seconds")
wall_ratio = None
if isinstance(base_wall, (int, float)) and isinstance(cand_wall, (int, float)) and base_wall > 0:
    wall_ratio = cand_wall / base_wall

base_kernel = (base_metrics.get("nsys") or {}).get("kernel_count")
cand_kernel = (cand_metrics.get("nsys") or {}).get("kernel_count")
kernel_delta = None
if isinstance(base_kernel, int) and isinstance(cand_kernel, int):
    kernel_delta = cand_kernel - base_kernel

# FIX 2 (v0.23 close): consume V023_FAIL_ON_PERF_REGRESSION so a slower / more-kernel
# candidate fails the B perf-gate, instead of a green verdict on correctness alone.
fail_on_perf = str(os.environ.get("V023_FAIL_ON_PERF_REGRESSION", "1")).strip().lower() not in ("0", "", "false", "no")
try:
    wall_tol = float(os.environ.get("V023_PERF_WALL_TOL", "1.05") or "1.05")
except ValueError:
    wall_tol = 1.05
if fail_on_perf:
    if isinstance(wall_ratio, (int, float)) and wall_ratio > wall_tol:
        failures.append({"kind": "perf_wall_regression", "wall_ratio": wall_ratio, "tolerance": wall_tol})
    if isinstance(kernel_delta, int) and kernel_delta > 0:
        failures.append({"kind": "perf_kernel_regression", "kernel_delta": kernel_delta})

payload = {
    "schema": "v023-validation-paired-gate-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": gate,
    "acceptance": acceptance,
    "verdict": "PASS" if not failures else "FAIL",
    "failures": failures,
    "field_compare": compare,
    "baseline_metrics": base_metrics,
    "candidate_metrics": cand_metrics,
    "derived": {
        "candidate_over_baseline_wall_ratio": wall_ratio,
        "kernel_count_delta": kernel_delta,
        "perf_regression_enforced": fail_on_perf,
        "wall_tolerance": wall_tol,
    },
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"gate": gate, "verdict": payload["verdict"], "output": out}, sort_keys=True))
sys.exit(0 if payload["verdict"] == "PASS" else 1)
PY
}

v023_compare_dirs() {
  local baseline_dir="${1:?baseline dir required}"
  local candidate_dir="${2:?candidate dir required}"
  local mode="${3:?mode required}"
  local out_json="${4:?out json required}"
  shift 4
  local -a cmd=(
    taskset -c "$V023_TASKSET"
    env
    JAX_PLATFORMS=cpu
    JAX_PLATFORM_NAME=cpu
    CUDA_VISIBLE_DEVICES=
    OMP_NUM_THREADS="$V023_OMP"
    PYTHONPATH="$V023_ROOT/src"
    python3 "$V023_FIELD_COMPARE"
    --baseline-dir "$baseline_dir"
    --candidate-dir "$candidate_dir"
    --domains $DOMAINS
    --mode "$mode"
    --autotune-floor "$V023_AUTOTUNE_FLOOR"
    --out-json "$out_json"
  )
  cmd+=("$@")
  v023_log "CPU field compare: ${cmd[*]}"
  "${cmd[@]}"
}

v023_run_paired_gpuwrf() {
  local gate="${1:?gate required}"
  local session="${2:?session required}"
  local base_ref="${3:?base ref required}"
  local cand_ref="${4:?candidate ref required}"
  local base_env="${5:-}"
  local cand_env="${6:-}"
  local base_label="${7:-baseline}"
  local cand_label="${8:-candidate}"

  mkdir -p "$session"
  {
    echo "gate=$gate"
    echo "session=$session"
    echo "base_ref=$base_ref"
    echo "candidate_ref=$cand_ref"
    echo "case=$CASE"
    echo "hours=$HOURS"
    echo "max_dom=$MAX_DOM"
    echo "domains=$DOMAINS"
    echo "base_env=$base_env"
    echo "candidate_env=$cand_env"
    echo "dryrun=$V023_DRYRUN"
  } > "$session/run_config.txt"

  if [[ "$V023_DRYRUN" == "1" ]]; then
    v023_log "DRY-RUN: would run paired GPU forecast for $gate under $V023_GPU_LOCK"
    v023_write_status_json "$session/run_pair.json" "$gate" "DRY_RUN" "NO_GPU_EXECUTED" \
      "Pair commands were staged only; rerun without V023_DRYRUN=1 when the GPU is free."
    return 0
  fi

  v023_require_file "$V023_GPU_LOCK" || return 1
  v023_require_ref "$base_ref" || return 1
  v023_require_ref "$cand_ref" || return 1
  v023_require_dir "$CASE" || return 1
  v023_require_file "$CASE/namelist.input" || return 1

  export V023_PAIR_GATE="$gate"
  export V023_PAIR_SESSION="$session"
  export V023_PAIR_BASE_REF="$base_ref"
  export V023_PAIR_CAND_REF="$cand_ref"
  export V023_PAIR_BASE_ENV="$base_env"
  export V023_PAIR_CAND_ENV="$cand_env"
  export V023_PAIR_BASE_LABEL="$base_label"
  export V023_PAIR_CAND_LABEL="$cand_label"
  export V023_PAIR_CASE="$CASE"
  export V023_PAIR_HOURS="$HOURS"
  export V023_PAIR_MAX_DOM="$MAX_DOM"
  export V023_PAIR_DOMAIN="$DOMAIN"
  export V023_PAIR_PROFILE_NSYS="$V023_PROFILE_NSYS"

  "$V023_GPU_LOCK" --timeout "$V023_LOCK_TIMEOUT" --label "v023-${gate}" -- \
    bash -c 'set -euo pipefail; source "$V023_COMMON"; _v023_run_paired_gpuwrf_unlocked'
}

_v023_git_worktree_add() {
  local ref="${1:?ref required}"
  local dst="${2:?dst required}"
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  git -C "$V023_ROOT" worktree add --detach "$dst" "$ref" >/dev/null
}

_v023_run_one_forecast() {
  local label="${1:?label required}"
  local wt="${2:?worktree required}"
  local out_dir="${3:?out dir required}"
  local proof_dir="${4:?proof dir required}"
  local scratch_dir="${5:?scratch dir required}"
  local cache_dir="${6:?cache dir required}"
  local extra_env="${7:-}"
  local run_dir
  run_dir="$(dirname "$out_dir")"
  mkdir -p "$out_dir" "$proof_dir" "$scratch_dir" "$cache_dir" "$run_dir"

  local -a env_args=()
  if [[ -n "$extra_env" ]]; then
    # shellcheck disable=SC2206
    env_args=($extra_env)
  fi

  local -a gpuwrf_cmd=(
    env
    PYTHONPATH="$wt/src"
    JAX_ENABLE_X64=true
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_ENABLE_COMPILATION_CACHE=true
    JAX_COMPILATION_CACHE_DIR="$cache_dir"
    GPUWRF_JAX_CACHE_DIR="$cache_dir"
    GPUWRF_CANAIRY_ROOT="$V023_GPUWRF_ROOT"
    GPUWRF_WRF_ROOT="$V023_WRF_ROOT"
    GPUWRF_SCRATCH="$scratch_dir"
    OMP_NUM_THREADS="$V023_OMP"
    "${env_args[@]}"
    taskset -c "$V023_TASKSET"
    python -m gpuwrf run
      --input-dir "$V023_PAIR_CASE"
      --namelist "$V023_PAIR_CASE/namelist.input"
      --output-dir "$out_dir"
      --proof-dir "$proof_dir"
      --scratch-dir "$scratch_dir"
      --hours "$V023_PAIR_HOURS"
  )
  if (( V023_PAIR_MAX_DOM > 1 )); then
    gpuwrf_cmd+=(--max-dom "$V023_PAIR_MAX_DOM")
  else
    gpuwrf_cmd+=(--domain "$V023_PAIR_DOMAIN")
  fi
  if [[ "$V023_FORCE_GPU_RUN" == "1" ]]; then
    gpuwrf_cmd+=(--force-gpu-run)
  fi

  local -a final_cmd=("${gpuwrf_cmd[@]}")
  local nsys_report="$run_dir/nsys"
  if [[ "${V023_PAIR_PROFILE_NSYS:-0}" == "1" ]]; then
    if command -v nsys >/dev/null 2>&1; then
      final_cmd=(
        nsys profile
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=none
        --force-overwrite=true
        --output "$nsys_report"
        "${gpuwrf_cmd[@]}"
      )
    else
      echo "nsys requested but not found" > "$run_dir/nsys_missing.txt"
    fi
  fi

  printf '%q ' "${final_cmd[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"

  local vram_csv="$run_dir/vram.csv"
  local vpid=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null >> "$vram_csv"
        sleep 2
      done ) &
    vpid=$!
  fi

  local start end rc
  start=$(date -u +%s)
  set +e
  (cd "$wt" && "${final_cmd[@]}") > "$run_dir/run.log" 2>&1
  rc=$?
  set -e
  end=$(date -u +%s)
  if [[ -n "$vpid" ]]; then
    kill "$vpid" 2>/dev/null || true
    wait "$vpid" 2>/dev/null || true
  fi
  echo "$rc" > "$run_dir/rc.txt"
  echo "$((end - start))" > "$run_dir/wall_s.txt"

  if [[ -f "$nsys_report.nsys-rep" ]]; then
    taskset -c "$V023_TASKSET" env PYTHONPATH="$V023_ROOT/src" python "$V023_ROOT/scripts/m7_nsys_stats_extract.py" \
      "$nsys_report.nsys-rep" --output "$run_dir/nsys_summary.json" \
      > "$run_dir/nsys_extract.log" 2>&1 || true
  fi

  _v023_write_run_metrics "$label" "$run_dir" "$out_dir" "$proof_dir" "$rc" "$((end - start))"
  return "$rc"
}

_v023_write_run_metrics() {
  local label="${1:?label required}"
  local run_dir="${2:?run dir required}"
  local out_dir="${3:?out dir required}"
  local proof_dir="${4:?proof dir required}"
  local rc="${5:?rc required}"
  local wall_s="${6:?wall required}"
  python3 - "$label" "$run_dir" "$out_dir" "$proof_dir" "$rc" "$wall_s" <<'PY'
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

label, run_dir_s, out_dir, proof_dir, rc_s, wall_s_s = sys.argv[1:7]
run_dir = Path(run_dir_s)
peak_vram = None
vram = run_dir / "vram.csv"
if vram.is_file():
    for row in csv.reader(vram.open()):
        for cell in row:
            match = re.search(r"(\d+(?:\.\d+)?)\s*MiB", cell)
            if match:
                value = float(match.group(1))
                peak_vram = value if peak_vram is None else max(peak_vram, value)
nsys = None
nsys_path = run_dir / "nsys_summary.json"
if nsys_path.is_file():
    try:
        nsys = json.loads(nsys_path.read_text())
    except Exception as exc:
        nsys = {"status": "UNREADABLE", "error": str(exc), "path": str(nsys_path)}
payload = {
    "schema": "v023-run-metrics-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "label": label,
    "returncode": int(rc_s),
    "wall_seconds": int(wall_s_s),
    "output_dir": out_dir,
    "proof_dir": proof_dir,
    "run_dir": str(run_dir),
    "peak_vram_mib": peak_vram,
    "nsys": nsys,
}
(run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

_v023_run_paired_gpuwrf_unlocked() {
  local session="$V023_PAIR_SESSION"
  mkdir -p "$session/worktrees" "$session/cache"
  local base_wt="$session/worktrees/$V023_PAIR_BASE_LABEL"
  local cand_wt="$session/worktrees/$V023_PAIR_CAND_LABEL"
  _v023_git_worktree_add "$V023_PAIR_BASE_REF" "$base_wt"
  _v023_git_worktree_add "$V023_PAIR_CAND_REF" "$cand_wt"

  v023_log "warmup $V023_PAIR_BASE_LABEL ($V023_PAIR_BASE_REF)"
  _v023_run_one_forecast "$V023_PAIR_BASE_LABEL-warmup" "$base_wt" \
    "$session/${V023_PAIR_BASE_LABEL}_warmup/out" "$session/${V023_PAIR_BASE_LABEL}_warmup/proof" \
    "$session/${V023_PAIR_BASE_LABEL}_warmup/scratch" "$session/cache/$V023_PAIR_BASE_LABEL" \
    "$V023_PAIR_BASE_ENV"

  v023_log "warmup $V023_PAIR_CAND_LABEL ($V023_PAIR_CAND_REF)"
  _v023_run_one_forecast "$V023_PAIR_CAND_LABEL-warmup" "$cand_wt" \
    "$session/${V023_PAIR_CAND_LABEL}_warmup/out" "$session/${V023_PAIR_CAND_LABEL}_warmup/proof" \
    "$session/${V023_PAIR_CAND_LABEL}_warmup/scratch" "$session/cache/$V023_PAIR_CAND_LABEL" \
    "$V023_PAIR_CAND_ENV"

  v023_log "measured $V023_PAIR_BASE_LABEL ($V023_PAIR_BASE_REF)"
  _v023_run_one_forecast "$V023_PAIR_BASE_LABEL" "$base_wt" \
    "$session/$V023_PAIR_BASE_LABEL/out" "$session/$V023_PAIR_BASE_LABEL/proof" \
    "$session/$V023_PAIR_BASE_LABEL/scratch" "$session/cache/$V023_PAIR_BASE_LABEL" \
    "$V023_PAIR_BASE_ENV"

  v023_log "measured $V023_PAIR_CAND_LABEL ($V023_PAIR_CAND_REF)"
  _v023_run_one_forecast "$V023_PAIR_CAND_LABEL" "$cand_wt" \
    "$session/$V023_PAIR_CAND_LABEL/out" "$session/$V023_PAIR_CAND_LABEL/proof" \
    "$session/$V023_PAIR_CAND_LABEL/scratch" "$session/cache/$V023_PAIR_CAND_LABEL" \
    "$V023_PAIR_CAND_ENV"

  if [[ "$V023_KEEP_WORKTREES" != "1" ]]; then
    git -C "$V023_ROOT" worktree remove --force "$base_wt" >/dev/null 2>&1 || true
    git -C "$V023_ROOT" worktree remove --force "$cand_wt" >/dev/null 2>&1 || true
  fi

  v023_write_status_json "$session/run_pair.json" "$V023_PAIR_GATE" "RAN" "GPU_SEQUENCE_COMPLETE" \
    "Warmup and measured baseline/candidate runs completed under one GPU lock."
}
