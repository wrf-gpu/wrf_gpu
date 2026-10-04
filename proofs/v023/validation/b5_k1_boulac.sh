#!/usr/bin/env bash
# B5: K1 BouLac correctness + warm perf, if K1 lands.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  K1_OFF_ENV default: GPUWRF_MYNN_BOULAC_ONZ=0
  K1_ON_ENV  default: GPUWRF_MYNN_BOULAC_ONZ=1
  K1_REQUIRED=0 marks this optional when the implementation did not land.

Acceptance: field compare PASS plus warm perf/VRAM win in the on arm.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

GATE="b5_k1_boulac"
v023_gate_paths "$GATE"
K1_REQUIRED="${K1_REQUIRED:-0}"

if [[ "$K1_REQUIRED" != "1" && -z "${K1_ON_ENV:-}" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "SKIP" "K1_NOT_REQUIRED" \
    "K1 BouLac gate is conditional in V0230_CLOSE_VALIDATION_PLAN; set K1_REQUIRED=1 to enforce."
  exit 0
fi

SESSION="$V023_RUN_DIR/ab"
COMPARE_JSON="$V023_RUN_DIR/b5_field_compare.json"
V023_PROFILE_NSYS="${V023_PROFILE_NSYS:-1}"
K1_OFF_ENV="${K1_OFF_ENV:-GPUWRF_MYNN_BOULAC_ONZ=0}"
K1_ON_ENV="${K1_ON_ENV:-GPUWRF_MYNN_BOULAC_ONZ=1}"

v023_run_paired_gpuwrf "$GATE" "$SESSION" "$HEAD" "$HEAD" "$K1_OFF_ENV" "$K1_ON_ENV" "off" "on" || exit $?

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "B5 K1 BouLac A/B commands staged only."
  exit 0
fi

v023_compare_dirs "$SESSION/off/out" "$SESSION/on/out" "autotune-floor" "$COMPARE_JSON" --no-spatial-splits
v023_json_combine_pair_gate "$V023_VERDICT_JSON" "$GATE" "$COMPARE_JSON" \
  "$SESSION/off/metrics.json" "$SESSION/on/metrics.json" \
  "K1 BouLac on must field-compare green and reproduce the accepted warm perf/VRAM win."
