#!/usr/bin/env bash
# A3: M1 fp32 release-gate items.
#
# Subgate A: fp64_default wrfout field identity vs branch base.
# Subgate B: exact killgate cold repro, if M1_KILLGATE_CMD is supplied, or a
#            closure JSON is supplied via M1_KILLGATE_CLOSED_JSON.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  BRANCH_BASE_REF             base ref for fp64_default compare (default: BASE_REF=$BASE_REF)
  HEAD                        candidate ref (default: $HEAD)
  M1_KILLGATE_CMD             optional cold killgate command to run under GPU lock
  M1_KILLGATE_CLOSED_JSON     optional JSON proving killgate already closed
  M1_KILLGATE_FORENSICS_JSON  optional input note, default proofs/.../m1_killgate_forensics.json

Acceptance: fp64_default field compare PASS, plus killgate command exits 0 or
closed JSON declares PASS/CLOSED.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then
    continue
  fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    usage
    exit 0
  fi
  echo "unknown argument: $arg" >&2
  usage >&2
  exit 2
done

GATE="a3_m1_fp32_release_gates"
BRANCH_BASE_REF="${BRANCH_BASE_REF:-$BASE_REF}"
M1_KILLGATE_FORENSICS_JSON="${M1_KILLGATE_FORENSICS_JSON:-$V023_VALIDATION_DIR/m1_killgate_forensics.json}"
v023_gate_paths "$GATE"
SESSION="$V023_RUN_DIR/fp64_default_pair"
COMPARE_JSON="$V023_RUN_DIR/a3_fp64_default_compare.json"

FP64_ENV="GPUWRF_ACOUSTIC_PRECISION_MODE=fp64_default GPUWRF_FP32_MODE= GPUWRF_FORCE_FP64=1"
v023_run_paired_gpuwrf "$GATE" "$SESSION" "$BRANCH_BASE_REF" "$HEAD" "$FP64_ENV" "$FP64_ENV" "branch_base" "head" || exit $?

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "A3 fp64_default pair and optional killgate were staged only."
  exit 0
fi

v023_compare_dirs "$SESSION/branch_base/out" "$SESSION/head/out" "exact" "$COMPARE_JSON" --no-spatial-splits
COMPARE_RC=$?

KILLGATE_JSON="$V023_RUN_DIR/a3_killgate.json"
KILLGATE_RC=0
if [[ -n "${M1_KILLGATE_CMD:-}" ]]; then
  v023_log "running M1 killgate under GPU lock: $M1_KILLGATE_CMD"
  "$V023_GPU_LOCK" --timeout "$V023_LOCK_TIMEOUT" --label "v023-a3-m1-killgate" -- \
    bash -lc "$M1_KILLGATE_CMD" > "$V023_RUN_DIR/m1_killgate.log" 2>&1
  KILLGATE_RC=$?
  python3 - "$KILLGATE_JSON" "$KILLGATE_RC" "$M1_KILLGATE_CMD" "$V023_RUN_DIR/m1_killgate.log" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, rc, cmd, log = sys.argv[1:5]
payload = {
  "schema": "v023-a3-m1-killgate-v1",
  "generated_utc": datetime.now(timezone.utc).isoformat(),
  "mode": "command",
  "command": cmd,
  "log": log,
  "returncode": int(rc),
  "verdict": "PASS" if int(rc) == 0 else "FAIL",
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
elif [[ -n "${M1_KILLGATE_CLOSED_JSON:-}" && -f "$M1_KILLGATE_CLOSED_JSON" ]]; then
  cp "$M1_KILLGATE_CLOSED_JSON" "$KILLGATE_JSON"
else
  KILLGATE_RC=2
  v023_write_status_json "$KILLGATE_JSON" "$GATE" "BLOCKED" "MISSING_M1_KILLGATE" \
    "Set M1_KILLGATE_CMD or M1_KILLGATE_CLOSED_JSON. Forensics path checked: $M1_KILLGATE_FORENSICS_JSON"
fi

python3 - "$V023_VERDICT_JSON" "$COMPARE_JSON" "$KILLGATE_JSON" "$COMPARE_RC" "$KILLGATE_RC" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, cmp_path, kill_path, cmp_rc, kill_rc = sys.argv[1:6]
compare = json.loads(Path(cmp_path).read_text())
kill = json.loads(Path(kill_path).read_text())
failures = []
if int(cmp_rc) != 0 or compare.get("verdict") != "PASS":
    failures.append({"kind": "fp64_default_compare", "path": cmp_path, "verdict": compare.get("verdict")})
if int(kill_rc) != 0 and str(kill.get("verdict")) not in {"PASS", "CLOSED"}:
    failures.append({"kind": "m1_killgate", "path": kill_path, "verdict": kill.get("verdict")})
payload = {
    "schema": "v023-a3-m1-fp32-release-gates-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": "A3",
    "verdict": "PASS" if not failures else "FAIL",
    "failures": failures,
    "fp64_default_compare": compare,
    "killgate": kill,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"gate": "A3", "verdict": payload["verdict"], "output": out}, sort_keys=True))
sys.exit(0 if not failures else 1)
PY
