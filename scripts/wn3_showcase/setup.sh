#!/usr/bin/env bash
# Usage: setup.sh FINAL_COMMIT  -> detached W7/source + pristine symlink; prints src tree. Refuses if source exists.
set -euo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W7; c=${1:?}
[[ ! -e $W/source ]] || { echo "W7/source exists" >&2; exit 2; }
git -C <USER_HOME>/src/wrf_gpu2_wt/wn3 worktree add --detach $W/source "$c" >/dev/null
ln -s <USER_HOME>/src/wrf_pristine $W/source/data/wrf_pristine
test -f $W/source/scripts/wn3_forecast.py && test -f $W/source/scripts/bench/s0_nested_harness.py
grep -q WN3_CLI_EXTRA $W/source/scripts/wn3_forecast.py || git -C $W/source -c user.name=wn3 -c user.email=wn3@local cherry-pick 744a71e944b36fd1a78ffd1b4989362726672132 >/dev/null  # R07 restart hook (scripts only)
# Launcher (wn3, scripts only): cherry-pick the lane commits that touch it if the FINAL tree predates its merge.
if [[ ! -f $W/source/scripts/run_parallel_cases.sh ]]; then
  for h in $(git -C <USER_HOME>/src/wrf_gpu2_wt/wn3 log --reverse --format=%H main..lane/wn3 -- scripts/run_parallel_cases.sh scripts/parallel_cases.py scripts/compress_wrfout.py); do
    git -C $W/source -c user.name=wn3 -c user.email=wn3@local cherry-pick "$h" >/dev/null
  done
fi
test -x $W/source/scripts/run_parallel_cases.sh
echo "{\"commit\": \"$(git -C $W/source rev-parse HEAD)\", \"src_tree\": \"$(git -C $W/source rev-parse HEAD:src)\", \"utc\": \"$(date -u +%FT%TZ)\"}" | tee $W/source_pin.json
