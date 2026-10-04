#!/usr/bin/env bash
# C2: V2 24-120h skill gate.
#
# Reuses the existing CPU-only skill comparator scripts/v020_skill_eval.py
# against fresh or pre-existing GPU/CPU wrfout directories. Forecast execution,
# when requested, is wrapped in with_gpu_lock.sh.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  C2_CASES           space list, default: canary
  C2_<CASE>_GPU_DIR  existing GPU wrfout dir, optional
  C2_<CASE>_CPU_DIR  CPU-WRF truth wrfout dir, required per case
  C2_<CASE>_INPUT    input case dir if GPU_DIR is absent
  C2_HOURS           default: 120

Acceptance: scripts/v020_skill_eval.py PASS for each case over requested domains.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

GATE="c2_skill_gate"
v023_gate_paths "$GATE"
C2_CASES="${C2_CASES:-canary}"
C2_HOURS="${C2_HOURS:-120}"
C2_DOMAINS="${C2_DOMAINS:-d01 d02 d03}"
RESULTS_JSONL="$V023_RUN_DIR/c2_results.jsonl"
: > "$RESULTS_JSONL"

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "Would run or compare C2 cases: $C2_CASES"
  exit 0
fi

for raw_case in $C2_CASES; do
  case_label="$(printf '%s' "$raw_case" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9_' '_')"
  gpu_var="C2_${case_label}_GPU_DIR"
  cpu_var="C2_${case_label}_CPU_DIR"
  input_var="C2_${case_label}_INPUT"
  gpu_dir="${!gpu_var:-}"
  cpu_dir="${!cpu_var:-}"
  input_dir="${!input_var:-}"
  case_dir="$V023_RUN_DIR/$raw_case"
  mkdir -p "$case_dir"

  if [[ -z "$cpu_dir" ]]; then
    v023_write_status_json "$case_dir/result.json" "$GATE-$raw_case" "BLOCKED" "MISSING_CPU_TRUTH" \
      "Set $cpu_var to CPU-WRF truth wrfout directory."
    cat "$case_dir/result.json" >> "$RESULTS_JSONL"; echo >> "$RESULTS_JSONL"
    continue
  fi

  if [[ -z "$gpu_dir" ]]; then
    if [[ -z "$input_dir" ]]; then
      v023_write_status_json "$case_dir/result.json" "$GATE-$raw_case" "BLOCKED" "MISSING_GPU_OR_INPUT" \
        "Set $gpu_var or $input_var."
      cat "$case_dir/result.json" >> "$RESULTS_JSONL"; echo >> "$RESULTS_JSONL"
      continue
    fi
    "$V023_GPU_LOCK" --timeout "$V023_LOCK_TIMEOUT" --label "v023-c2-$raw_case" -- \
      taskset -c "$V023_TASKSET" env \
        PYTHONPATH="$V023_ROOT/src" JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS="$V023_OMP" \
        python -m gpuwrf run \
          --input-dir "$input_dir" \
          --namelist "$input_dir/namelist.input" \
          --output-dir "$case_dir/gpu_out" \
          --proof-dir "$case_dir/gpu_proof" \
          --hours "$C2_HOURS" \
          --max-dom "$MAX_DOM" \
      > "$case_dir/gpu_run.log" 2>&1
    gpu_rc=$?
    if [[ "$gpu_rc" != "0" ]]; then
      v023_write_status_json "$case_dir/result.json" "$GATE-$raw_case" "FAIL" "GPU_RUN_FAILED" "rc=$gpu_rc"
      cat "$case_dir/result.json" >> "$RESULTS_JSONL"; echo >> "$RESULTS_JSONL"
      continue
    fi
    gpu_dir="$case_dir/gpu_out"
  fi

  taskset -c "$V023_TASKSET" env PYTHONPATH="$V023_ROOT/src" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
    python "$V023_ROOT/scripts/v020_skill_eval.py" \
      --candidate-dir "$gpu_dir" \
      --oracle-dir "$cpu_dir" \
      --domains $C2_DOMAINS \
      --out "$case_dir/skill.json" \
      > "$case_dir/skill.stdout.txt" 2> "$case_dir/skill.stderr.txt"
  skill_rc=$?

  python3 - "$case_dir/result.json" "$raw_case" "$gpu_dir" "$cpu_dir" "$case_dir/skill.json" "$skill_rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, case, gpu, cpu, skill_path, rc = sys.argv[1:7]
skill = json.loads(Path(skill_path).read_text()) if Path(skill_path).is_file() else {}
payload = {
    "schema": "v023-c2-case-skill-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "case": case,
    "gpu_dir": gpu,
    "cpu_truth_dir": cpu,
    "skill_json": skill_path,
    "returncode": int(rc),
    "verdict": "PASS" if int(rc) == 0 and skill.get("GATE_PASS") is True else "FAIL",
    "skill": skill,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  cat "$case_dir/result.json" >> "$RESULTS_JSONL"; echo >> "$RESULTS_JSONL"
done

python3 - "$V023_VERDICT_JSON" "$RESULTS_JSONL" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, path = sys.argv[1:3]
rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
failures = [row for row in rows if row.get("verdict") != "PASS"]
payload = {
    "schema": "v023-c2-skill-gate-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": "C2",
    "verdict": "PASS" if not failures else "FAIL",
    "case_count": len(rows),
    "failures": failures,
    "results": rows,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"gate": "C2", "verdict": payload["verdict"], "output": out}, sort_keys=True))
sys.exit(0 if not failures else 1)
PY
