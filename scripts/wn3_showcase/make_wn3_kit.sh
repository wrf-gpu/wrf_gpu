#!/usr/bin/env bash
# One command for a lane that needs WN3 3-nest GPU arms on the twin-delivered LW9 tree (fid-q2, fid-cloud, ...).
# Builds <USER_HOME>/wrf_gpu2_lanes/<lane>/wn3kit: source -> wn3 W6/source (immutable LW9 snapshot, symlink: NEVER edit it;
# make your own snapshot for code changes), cache_c0 = PRIVATE copy of the warm LW9 cache (~0.9 GB; your flag changes compile
# into it, delete it by exact path when done), arm.sh/parallel.sh copies with an ARM_ENV_OVERRIDE hook (sourced AFTER flags_lw9.env).
# Usage: bash <USER_HOME>/wrf_gpu2_lanes/wn3/W6/make_wn3_kit.sh <lane>      (CPU only, ~10 s). Then read <kit>/README.md.
set -euo pipefail
lane=${1:?usage: make_wn3_kit.sh <lane>}; K=<USER_HOME>/wrf_gpu2_lanes/$lane/wn3kit; W=<USER_HOME>/wrf_gpu2_lanes/wn3/W6
[[ ! -e $K ]] || { echo "exists: $K" >&2; exit 2; }
mkdir -p $K/arms
ln -s $W/source $K/source
cp -a $W/cache_c0 $K/cache_c0
echo 0.1675 > $K/mem_fraction
sed 's#^set -a; source $src/scripts/bench/flags_lw9.env; set +a$#set -a; source $src/scripts/bench/flags_lw9.env; if [[ -n "${ARM_ENV_OVERRIDE:-}" ]]; then source "$ARM_ENV_OVERRIDE"; fi; set +a#' $W/arm.sh > $K/arm.sh
sed "s#^W4=<USER_HOME>/wrf_gpu2_lanes/wn3/W6\$#W4=$K#" $W/parallel.sh > $K/parallel.sh
grep -q 'ARM_ENV_OVERRIDE' $K/arm.sh && grep -q "^W4=$K\$" $K/parallel.sh
sed "s#@K@#$K#g; s#@LANE@#$lane#g" $W/wn3kit_README.md > $K/README.md
echo "kit ready: $K (README.md)"
