#!/usr/bin/env bash
# C1: Full-suite regression.
#
# Reuses scripts/run_regression_suite.py. The whole suite invocation is wrapped
# in with_gpu_lock.sh because the suite can launch GPU subcommands internally.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  C1_MILESTONE          default: V0230
  C1_PREVIOUS_SNAPSHOT  required for ZERO-new-regressions check unless
                        C1_ALLOW_NO_PREVIOUS=1
  C1_EXPECTED_TESTS     default: 53
  C1_FORCE_GPU_RUN      default: 0

Acceptance: aggregate PASS, exactly C1_EXPECTED_TESTS field tests, and snapshot
comparison has no newly failing tests.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

GATE="c1_full_suite"
v023_gate_paths "$GATE"
C1_MILESTONE="${C1_MILESTONE:-V0230}"
C1_EXPECTED_TESTS="${C1_EXPECTED_TESTS:-53}"
C1_PROOF_DIR="${C1_PROOF_DIR:-$V023_RUN_DIR/regression}"
C1_ALLOW_NO_PREVIOUS="${C1_ALLOW_NO_PREVIOUS:-0}"
C1_FORCE_GPU_RUN="${C1_FORCE_GPU_RUN:-0}"

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "Would run scripts/run_regression_suite.py under GPU lock."
  exit 0
fi

mkdir -p "$C1_PROOF_DIR"
force_arg=()
[[ "$C1_FORCE_GPU_RUN" == "1" ]] && force_arg=(--force-gpu-run)

"$V023_GPU_LOCK" --timeout "$V023_LOCK_TIMEOUT" --label "v023-c1-full-suite" -- \
  taskset -c "$V023_TASKSET" env \
    PYTHONPATH="$V023_ROOT/src" JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS="$V023_OMP" \
    python "$V023_ROOT/scripts/run_regression_suite.py" \
      --proof-dir "$C1_PROOF_DIR" \
      --milestone-snapshot "$C1_MILESTONE" \
      --strict \
      "${force_arg[@]}" \
  > "$V023_RUN_DIR/c1_suite.stdout.txt" 2> "$V023_RUN_DIR/c1_suite.stderr.txt"
suite_rc=$?

compare_args=(--proof-dir "$C1_PROOF_DIR" --compare-snapshot "$C1_MILESTONE")
if [[ -n "${C1_PREVIOUS_SNAPSHOT:-}" ]]; then
  compare_args+=(--previous-snapshot "$C1_PREVIOUS_SNAPSHOT")
fi
taskset -c "$V023_TASKSET" env PYTHONPATH="$V023_ROOT/src" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  python "$V023_ROOT/scripts/run_regression_suite.py" "${compare_args[@]}" \
  > "$V023_RUN_DIR/c1_compare.stdout.txt" 2> "$V023_RUN_DIR/c1_compare.stderr.txt"
compare_rc=$?

python3 - "$V023_VERDICT_JSON" "$C1_PROOF_DIR" "$C1_MILESTONE" "$C1_EXPECTED_TESTS" "$suite_rc" "$compare_rc" "$C1_ALLOW_NO_PREVIOUS" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

out, proof_dir, milestone, expected_s, suite_rc, compare_rc, allow_no_prev = sys.argv[1:8]
proof = Path(proof_dir)
aggregate = json.loads((proof / f"aggregate_{milestone}.json").read_text()) if (proof / f"aggregate_{milestone}.json").is_file() else {}
check_files = sorted(proof.glob(f"regression_check_*_to_{milestone}.json"))
check = json.loads(check_files[-1].read_text()) if check_files else {}
failures = []
if int(suite_rc) != 0 or aggregate.get("aggregate_status") != "PASS":
    failures.append({"kind": "aggregate", "suite_rc": int(suite_rc), "aggregate_status": aggregate.get("aggregate_status")})
if int(aggregate.get("field_test_count", -1)) != int(expected_s):
    failures.append({"kind": "field_test_count", "expected": int(expected_s), "actual": aggregate.get("field_test_count")})
if check.get("status") == "NO_PREVIOUS_SNAPSHOT" and allow_no_prev != "1":
    failures.append({"kind": "no_previous_snapshot", "status": check.get("status")})
elif int(compare_rc) != 0 or check.get("status") not in {"PASS", "NO_PREVIOUS_SNAPSHOT"}:
    failures.append({"kind": "snapshot_compare", "compare_rc": int(compare_rc), "status": check.get("status")})
payload = {
    "schema": "v023-c1-full-suite-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": "C1",
    "verdict": "PASS" if not failures else "FAIL",
    "failures": failures,
    "aggregate": aggregate,
    "snapshot_check": check,
    "proof_dir": proof_dir,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"gate": "C1", "verdict": payload["verdict"], "output": out}, sort_keys=True))
sys.exit(0 if not failures else 1)
PY
