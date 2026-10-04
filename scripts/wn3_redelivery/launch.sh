#!/usr/bin/env bash
# Usage: launch.sh ARM HOURS CASE... -> one GPU lock, 3 cases in parallel, QUIET held (re-delivery = gate run).
set -euo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W8; arm=${1:?}; hours=${2:?}; shift 2
[[ -d $W/source ]] || { echo "run setup.sh first" >&2; exit 2; }
[[ ! -e $W/$arm ]] || { echo "refuse existing $W/$arm" >&2; exit 2; }
mkdir -p $W/$arm; echo "queued $(date -u +%FT%TZ) $hours h $*" > $W/$arm/queued.txt
exec <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label wn3 --timeout 28800 -- \
  timeout --signal=TERM --kill-after=90 7200 env ROOT=$W bash <USER_HOME>/wrf_gpu2_lanes/wn3/W6/quiet_wrap.sh bash $W/parallel.sh $W $W/$arm "$hours" "$@"
