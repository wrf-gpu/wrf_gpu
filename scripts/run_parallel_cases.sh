#!/usr/bin/env bash
# Run several WRF cases concurrently on ONE GPU, as many at a time as VRAM and host RAM admit.
#
#   scripts/run_parallel_cases.sh --out-root runs/batch CASE_DIR [CASE_DIR ...] -- --domains-from-namelist --hours 24
#   (on a host with the shared dev GPU lock: scripts/with_gpu_lock.sh --label <name> -- scripts/run_parallel_cases.sh ...)
#
# Everything after '--' is passed to every `python -m gpuwrf.cli run` (--input-dir/--output-dir are set per case).
# Per-case sizing = the product C-auto memory plan; a geometry without a plan runs alone first to record it.
# --dry-run prints the admission plan (JSON). Options and receipts: scripts/parallel_cases.py --help, AI_OPERATOR.md.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
# The launcher itself never touches the GPU; the cases get the caller's JAX_PLATFORMS back.
export _PARALLEL_CASES_JAX_PLATFORMS="${JAX_PLATFORMS-<unset>}"
JAX_PLATFORMS=cpu exec "${PYTHON:-python3}" "$repo/scripts/parallel_cases.py" "$@"
