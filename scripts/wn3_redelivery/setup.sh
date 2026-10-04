#!/usr/bin/env bash
# Usage: setup.sh TREE_COMMIT -> W8/source (detached) + pristine symlink + WN3_CLI_EXTRA hook if missing; prints pin.
set -euo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W8; c=${1:?}
[[ ! -e $W/source ]] || { echo "W8/source exists" >&2; exit 2; }
git -C <USER_HOME>/src/wrf_gpu2_wt/wn3 worktree add --detach $W/source "$c" >/dev/null
ln -s <USER_HOME>/src/wrf_pristine $W/source/data/wrf_pristine
grep -q WN3_CLI_EXTRA $W/source/scripts/wn3_forecast.py || git -C $W/source -c user.name=wn3 -c user.email=wn3@local cherry-pick 744a71e944b36fd1a78ffd1b4989362726672132 >/dev/null
test -f $W/source/scripts/bench/s0_nested_harness.py
echo 0.1675 > $W/mem_fraction
echo "{\"commit\": \"$(git -C $W/source rev-parse HEAD)\", \"base\": \"$c\", \"src_tree\": \"$(git -C $W/source rev-parse HEAD:src)\", \"utc\": \"$(date -u +%FT%TZ)\"}" | tee $W/source_pin.json
