#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log="$(mktemp -t wrf_gpu_verify_reproducibility.XXXXXX.log)"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if taskset -c 0-3 pytest -q \
    tests/test_m7_honest_speedup.py \
    tests/test_m7_profiler_window.py \
    tests/test_m7_rca_helpers.py \
    tests/test_m7_wrfout_io_compat.py \
    >>"$log" 2>&1 \
  && taskset -c 0-3 bash scripts/m7_publication_audit.sh >>"$log" 2>&1; then
  echo "PASS verify_reproducibility log=$log"
else
  echo "FAIL verify_reproducibility log=$log"
  exit 1
fi
