#!/usr/bin/env bash
# v0.15 S1 — nsys launch/idle instrument for the host-removal variants.
#
# Usage: run_nsys_capture.sh <tag> [extra env as VAR=VAL ...]
# Profiles probe_nsys_steady.py (3 x 50 production steps) and emits:
#   proofs/perf/v015/nsys_<tag>_{cuda_api_sum,cuda_gpu_kern_sum,nvtx_gpu_proj_sum}.csv
#   plus a per-step launch/graph/idle summary on stdout.
#
# For command-buffer variants the caller MUST include
# --xla_enable_command_buffers_during_profiling=true in XLA_FLAGS (XLA falls
# back to op-by-op under an active profiler otherwise -- measured fact, see
# the conversion-pass strings).  Keep that flag in the non-nsys A/B runs too
# so the compile-cache key (and therefore the measured executable) is shared.
set -euo pipefail
cd "$(dirname "$0")/../../.."

TAG="$1"; shift
REP="/tmp/v015_nsys_${TAG}"
rm -f "${REP}.nsys-rep" "${REP}.sqlite"

for kv in "$@"; do export "${kv?}"; done

scripts/with_gpu_lock.sh --label "s1-nsys-${TAG}" -- \
  taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
    JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
    nsys profile -t cuda,nvtx --cuda-graph-trace=node -o "$REP" --force-overwrite true \
      python proofs/perf/v015/probe_nsys_steady.py

for report in cuda_api_sum cuda_gpu_kern_sum nvtx_gpu_proj_sum; do
  nsys stats --report "$report" --format csv --force-export=true "$REP.nsys-rep" \
    > "proofs/perf/v015/nsys_${TAG}_${report}.csv" 2>/dev/null || true
done

python - "$TAG" <<'EOF'
import csv, json, sys
from pathlib import Path
tag = sys.argv[1]
here = Path("proofs/perf/v015")
steps = 150.0

def rows(p):
    if not p.exists():
        return []
    with open(p) as f:
        rs = [r for r in csv.reader(f) if r]
    hdr = None
    out = []
    for r in rs:
        if hdr is None:
            if any("Time" in c or "Name" in c for c in r):
                hdr = r
            continue
        out.append(dict(zip(hdr, r)))
    return out

def f(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0

api = rows(here / f"nsys_{tag}_cuda_api_sum.csv")
launch = sum(f(r.get("Num Calls")) for r in api if "LaunchKernel" in r.get("Name", ""))
graphl = sum(f(r.get("Num Calls")) for r in api if "GraphLaunch" in r.get("Name", ""))
launch_ms = sum(f(r.get("Total Time (ns)")) for r in api if "LaunchKernel" in r.get("Name", "")) / 1e6
graph_ms = sum(f(r.get("Total Time (ns)")) for r in api if "GraphLaunch" in r.get("Name", "")) / 1e6
kern = rows(here / f"nsys_{tag}_cuda_gpu_kern_sum.csv")
kern_ms = sum(f(r.get("Total Time (ns)")) for r in kern) / 1e6
print(json.dumps({
    "tag": tag, "steps_profiled": steps,
    "cuLaunchKernel_per_step": round(launch / steps, 1),
    "cuGraphLaunch_per_step": round(graphl / steps, 1),
    "launch_api_ms_per_step": round(launch_ms / steps, 2),
    "graph_api_ms_per_step": round(graph_ms / steps, 2),
    "device_kernel_ms_per_step": round(kern_ms / steps, 2),
}, indent=2))
EOF
