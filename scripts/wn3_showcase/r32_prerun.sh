#!/usr/bin/env bash
# Pre-run of alisios' frozen twin scoring exactly as they run it (R32-31 wrapper -> pinned R32-01 v2, freeze v1), from a LANE
# COPY (it writes next to itself; never run in place). Only change in the copy: r3231_common.manifest_path honours
# WN3_PRERUN_MANIFEST so an undelivered manifest can be scored. CPU only, cores 12,13,28,29 (their pin()).
# Usage: r32_prerun.sh ISSUE(e.g. 20260227_18z) GPU_MANIFEST_JSON OUT_NAME [formal|attr_shim]
set -euo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/wn3/r32_copy
[[ $(sha256sum $L/R32-01/compare_gpu_cpu_R32_01_v2.py | cut -c1-8) == 22a73289 ]] || { echo "R32-01 copy is not 22a73289" >&2; exit 3; }
cd $L/R32-31
WN3_PRERUN_MANIFEST="$2" JAX_PLATFORMS=cpu taskset -c 12,13,28,29 nice -n 19 <USER_HOME>/miniconda3/bin/python3 r3231_score_twin.py "$1" "$3" "${4:-formal}"
