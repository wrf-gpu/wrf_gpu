#!/usr/bin/env bash
# B1: GATE-3 P1/P7b warm no-regress.
#
# Default A/B:
#   off: GPUWRF_SINGLE_SCAN=0 GPUWRF_FUSED_CASCADE_LOW_EFFORT=0
#   on : GPUWRF_SINGLE_SCAN=1 GPUWRF_FUSED_CASCADE_LOW_EFFORT=1
#
# Nsight Systems capture is enabled by default for this perf gate so launch
# counts can be extracted with scripts/m7_nsys_stats_extract.py.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  B1_OFF_ENV / B1_ON_ENV override A/B env assignments.
  V023_PROFILE_NSYS=1 by default for launch count extraction.

Acceptance: field compare PASS and candidate has no warm wall/kernel-count
regression relative to off arm.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

GATE="b1_gate3_p1_p7b"
v023_gate_paths "$GATE"
SESSION="$V023_RUN_DIR/ab"
COMPARE_JSON="$V023_RUN_DIR/b1_field_compare.json"
V023_PROFILE_NSYS="${V023_PROFILE_NSYS:-1}"

B1_OFF_ENV="${B1_OFF_ENV:-GPUWRF_SINGLE_SCAN=0 GPUWRF_FUSED_CASCADE_LOW_EFFORT=0}"
B1_ON_ENV="${B1_ON_ENV:-GPUWRF_SINGLE_SCAN=1 GPUWRF_FUSED_CASCADE_LOW_EFFORT=1}"
v023_run_paired_gpuwrf "$GATE" "$SESSION" "$HEAD" "$HEAD" "$B1_OFF_ENV" "$B1_ON_ENV" "off" "on" || exit $?

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "B1 P1/P7b A/B commands staged only."
  exit 0
fi

v023_compare_dirs "$SESSION/off/out" "$SESSION/on/out" "autotune-floor" "$COMPARE_JSON" --no-spatial-splits
v023_json_combine_pair_gate "$V023_VERDICT_JSON" "$GATE" "$COMPARE_JSON" \
  "$SESSION/off/metrics.json" "$SESSION/on/metrics.json" \
  "P1/P7b enabled must field-compare green and show no warm wall/kernel-count regression."
