#!/usr/bin/env bash
# A1: Default byte/tolerance identity, v0.23 default vs v0.22.2 fresh baseline.
#
# Runs warm-vs-warm paired forecasts under scripts/with_gpu_lock.sh, then compares
# all wrfout variables with the reusable fresh paired field comparator. No stored
# state digest is read.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  CASE        input case directory (default: $CASE)
  BASE_REF    baseline ref/tag (default: $BASE_REF)
  HEAD        candidate ref (default: $HEAD)
  HOURS       forecast hours (default: $HOURS)
  MAX_DOM     nested domains (default: $MAX_DOM)
  DOMAINS     domains to compare (default: "$DOMAINS")
  V023_AUTOTUNE_FLOOR fresh-paired tolerance floor (default: $V023_AUTOTUNE_FLOOR)

Acceptance: field compare PASS across all requested domains; no missing vars,
time metadata mismatch, masking, or stored digest.
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

GATE="a1_default_byte_identity"
v023_gate_paths "$GATE"
SESSION="$V023_RUN_DIR/paired"
COMPARE_JSON="$V023_RUN_DIR/a1_field_compare.json"

DEFAULT_OFF_ENV="GPUWRF_BATCH_ENSEMBLE= GPUWRF_SINGLE_SCAN= GPUWRF_FUSED_CASCADE_LOW_EFFORT= GPUWRF_ACOUSTIC_PRECISION_MODE=fp64_default"
v023_run_paired_gpuwrf "$GATE" "$SESSION" "$BASE_REF" "$HEAD" "$DEFAULT_OFF_ENV" "$DEFAULT_OFF_ENV" "v0222" "v023" || exit $?

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "A1 commands staged only. Run without --dry-run when GPU lane is free."
  exit 0
fi

v023_compare_dirs "$SESSION/v0222/out" "$SESSION/v023/out" "autotune-floor" "$COMPARE_JSON" --no-spatial-splits
v023_json_combine_pair_gate "$V023_VERDICT_JSON" "$GATE" "$COMPARE_JSON" \
  "$SESSION/v0222/metrics.json" "$SESSION/v023/metrics.json" \
  "v0.23 default equals fresh v0.22.2 baseline within max_abs <= V023_AUTOTUNE_FLOOR for all wrfout variables."
