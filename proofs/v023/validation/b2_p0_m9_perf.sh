#!/usr/bin/env bash
# B2: P0 M9-reduction perf gate.
#
# Runs a warm A/B for the output-boundary/M9 reduction lever and compares wrfout
# fields. The concrete P0 env is intentionally parameterized because the P0
# implementation may land on a different branch; defaults are fail-closed/off.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  P0_OFF_ENV  default: GPUWRF_P0_M9_REDUCTION=0
  P0_ON_ENV   default: GPUWRF_P0_M9_REDUCTION=1

Acceptance: byte/tolerance-identical wrfout plus lower device-time or peak-VRAM
transient in the on arm.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

GATE="b2_p0_m9_perf"
v023_gate_paths "$GATE"
SESSION="$V023_RUN_DIR/ab"
COMPARE_JSON="$V023_RUN_DIR/b2_field_compare.json"
V023_PROFILE_NSYS="${V023_PROFILE_NSYS:-1}"

P0_OFF_ENV="${P0_OFF_ENV:-GPUWRF_P0_M9_REDUCTION=0}"
P0_ON_ENV="${P0_ON_ENV:-GPUWRF_P0_M9_REDUCTION=1}"
v023_run_paired_gpuwrf "$GATE" "$SESSION" "$HEAD" "$HEAD" "$P0_OFF_ENV" "$P0_ON_ENV" "off" "on" || exit $?

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "B2 P0/M9 A/B commands staged only."
  exit 0
fi

v023_compare_dirs "$SESSION/off/out" "$SESSION/on/out" "exact" "$COMPARE_JSON" --no-spatial-splits
v023_json_combine_pair_gate "$V023_VERDICT_JSON" "$GATE" "$COMPARE_JSON" \
  "$SESSION/off/metrics.json" "$SESSION/on/metrics.json" \
  "P0 on must be output-identical and reduce M9 device-time or peak VRAM transient."
