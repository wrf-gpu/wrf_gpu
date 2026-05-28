#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/run_milestone_regression.sh <milestone>" >&2
  exit 64
fi

milestone="$1"

taskset -c 0-3 python scripts/run_regression_suite.py --milestone-snapshot "${milestone}" --force-gpu-run
taskset -c 0-3 python scripts/run_regression_suite.py --compare-snapshot "${milestone}"
