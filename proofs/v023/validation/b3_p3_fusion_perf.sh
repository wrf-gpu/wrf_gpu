#!/usr/bin/env bash
# B3: P3 two-domain root fusion perf gate.
#
# Defaults to a 2-domain run. Set MAX_DOM=2 and DOMAINS="d01 d02" unless
# explicitly overridden by the caller.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v023_validation_common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run]

Env:
  P3_OFF_ENV  default: GPUWRF_NESTED_FUSE=0
  P3_ON_ENV   default: GPUWRF_NESTED_FUSE=1 GPUWRF_ROOT_FUSION=1
  MAX_DOM     default for this script: 2
  DOMAINS     default for this script: "d01 d02"

Acceptance: field compare PASS and launch-count/wall reduction.
EOF
  v023_usage_common
}

for arg in "$@"; do
  if v023_parse_common_flag "$arg"; then continue; fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then usage; exit 0; fi
  echo "unknown argument: $arg" >&2; usage >&2; exit 2
done

MAX_DOM="${B3_MAX_DOM:-2}"
DOMAINS="${B3_DOMAINS:-d01 d02}"
export MAX_DOM DOMAINS
GATE="b3_p3_fusion_perf"
v023_gate_paths "$GATE"
SESSION="$V023_RUN_DIR/ab"
COMPARE_JSON="$V023_RUN_DIR/b3_field_compare.json"
V023_PROFILE_NSYS="${V023_PROFILE_NSYS:-1}"

P3_OFF_ENV="${P3_OFF_ENV:-GPUWRF_NESTED_FUSE=0}"
P3_ON_ENV="${P3_ON_ENV:-GPUWRF_NESTED_FUSE=1 GPUWRF_ROOT_FUSION=1}"
v023_run_paired_gpuwrf "$GATE" "$SESSION" "$HEAD" "$HEAD" "$P3_OFF_ENV" "$P3_ON_ENV" "off" "on" || exit $?

if [[ "$V023_DRYRUN" == "1" ]]; then
  v023_write_status_json "$V023_VERDICT_JSON" "$GATE" "DRY_RUN" "NO_GPU_EXECUTED" \
    "B3 P3 fusion A/B commands staged only."
  exit 0
fi

v023_compare_dirs "$SESSION/off/out" "$SESSION/on/out" "autotune-floor" "$COMPARE_JSON" --no-spatial-splits
v023_json_combine_pair_gate "$V023_VERDICT_JSON" "$GATE" "$COMPARE_JSON" \
  "$SESSION/off/metrics.json" "$SESSION/on/metrics.json" \
  "P3 root fusion must be output-identical and reduce launch count / warm wall on 2-domain case."
