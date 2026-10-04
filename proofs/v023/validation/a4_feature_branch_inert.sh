#!/usr/bin/env bash
# A4: Feature branch default-inert check.
#
# For each feature ref, run default namelist with no opt-in env and compare the
# fresh wrfout against v0.22.2. Optional fail-closed commands can be supplied per
# feature as A4_FAILCLOSED_CMD_<LABEL>.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  FEATURE_REFS   space list of LABEL=REF entries, e.g. "G2=branch1 F2=branch2"
  G2_REF F2_REF F3_REF G3_REF are also accepted if FEATURE_REFS is unset.
  A4_FAILCLOSED_CMD_<LABEL> optional fail-closed smoke command for that feature.

Acceptance: every feature default path field-compares PASS vs BASE_REF and every
feature has a supplied fail-closed/opt-in smoke that exits 0.
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

if [[ -z "${FEATURE_REFS:-}" ]]; then
  FEATURE_REFS=""
  [[ -n "${G2_REF:-}" ]] && FEATURE_REFS+=" G2=$G2_REF"
  [[ -n "${F2_REF:-}" ]] && FEATURE_REFS+=" F2=$F2_REF"
  [[ -n "${F3_REF:-}" ]] && FEATURE_REFS+=" F3=$F3_REF"
  [[ -n "${G3_REF:-}" ]] && FEATURE_REFS+=" G3=$G3_REF"
fi

GATE="a4_feature_branch_inert"
v023_gate_paths "$GATE"

if [[ -z "${FEATURE_REFS// }" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "BLOCKED" "NO_FEATURE_REFS" \
    "Set FEATURE_REFS or G2_REF/F2_REF/F3_REF/G3_REF."
  exit 2
fi

RESULTS_JSONL="$V023_RUN_DIR/a4_results.jsonl"
: > "$RESULTS_JSONL"

for item in $FEATURE_REFS; do
  label="${item%%=*}"
  ref="${item#*=}"
  if [[ -z "$label" || -z "$ref" || "$label" == "$ref" ]]; then
    v023_write_status_json "$V023_RUN_DIR/a4_${item}_badref.json" "$GATE" "FAIL" "BAD_FEATURE_REF" "$item"
    continue
  fi
  safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')"
  session="$V023_RUN_DIR/$safe_label"
  compare_json="$session/field_compare.json"
  if ! v023_run_paired_gpuwrf "$GATE-$safe_label" "$session" "$BASE_REF" "$ref" "" "" "v0222" "$safe_label"; then
    v023_write_status_json "$session/result.json" "$GATE-$safe_label" "FAIL" "PAIRED_RUN_FAILED" "$ref"
    cat "$session/result.json" >> "$RESULTS_JSONL"; echo >> "$RESULTS_JSONL"
    continue
  fi
  if [[ "$V023_DRYRUN" == "1" ]]; then
    python3 - "$RESULTS_JSONL" "$label" "$ref" <<'PY'
import json, sys
path, label, ref = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"label": label, "ref": ref, "verdict": "DRY_RUN"}) + "\n")
PY
    continue
  fi
  v023_compare_dirs "$session/v0222/out" "$session/$safe_label/out" "autotune-floor" "$compare_json" --no-spatial-splits
  compare_rc=$?

  fail_var="A4_FAILCLOSED_CMD_$label"
  fail_cmd="${!fail_var:-}"
  fail_json="$session/failclosed.json"
  fail_rc=0
  if [[ -n "$fail_cmd" ]]; then
    "$V023_GPU_LOCK" --timeout "$V023_LOCK_TIMEOUT" --label "v023-a4-${safe_label}-failclosed" -- \
      bash -lc "$fail_cmd" > "$session/failclosed.log" 2>&1
    fail_rc=$?
    v023_write_status_json "$fail_json" "$GATE-$safe_label" "$([[ "$fail_rc" == "0" ]] && echo PASS || echo FAIL)" \
      "FAILCLOSED_CMD_RC_$fail_rc" "$fail_cmd"
  else
    fail_rc=2
    v023_write_status_json "$fail_json" "$GATE-$safe_label" "BLOCKED" "MISSING_FAILCLOSED_CMD" \
      "Set $fail_var to confirm opt-in/fail-closed behavior."
  fi

  python3 - "$RESULTS_JSONL" "$label" "$ref" "$compare_json" "$compare_rc" "$fail_json" "$fail_rc" <<'PY'
import json, sys
path, label, ref, cmp_path, cmp_rc, fail_path, fail_rc = sys.argv[1:8]
cmp_payload = json.loads(open(cmp_path).read()) if cmp_path and int(cmp_rc) == 0 else {"verdict": "FAIL_OR_MISSING"}
fail_payload = json.loads(open(fail_path).read())
verdict = "PASS" if cmp_payload.get("verdict") == "PASS" and int(fail_rc) == 0 else "FAIL"
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "label": label,
        "ref": ref,
        "verdict": verdict,
        "field_compare": cmp_payload,
        "failclosed": fail_payload,
    }) + "\n")
PY
done

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "A4 feature commands staged only."
  exit 0
fi

python3 - "$V023_VERDICT_JSON" "$RESULTS_JSONL" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, path = sys.argv[1:3]
rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
failures = [row for row in rows if row.get("verdict") != "PASS"]
payload = {
    "schema": "v023-a4-feature-branch-inert-v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "gate": "A4",
    "verdict": "PASS" if not failures else "FAIL",
    "feature_count": len(rows),
    "failures": failures,
    "results": rows,
}
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"gate": "A4", "verdict": payload["verdict"], "output": out}, sort_keys=True))
sys.exit(0 if not failures else 1)
PY
