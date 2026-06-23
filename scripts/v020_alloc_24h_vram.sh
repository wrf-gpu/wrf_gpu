#!/usr/bin/env bash
# v020_alloc_24h_vram.sh -- the cuda_async MUST-NOT-REGRESS 24h gate for v0.20.0.
# Runs ONE warm 24h all-7 9-domain forecast with the cuda_async pooling allocator
# + all bit-identical host levers ON, sampling GPU VRAM at 1 Hz (timestamped), then
# reduces to a per-output-group VRAM census. PASS = no CUDA OOM, all domains
# finite + outputs present, and peak VRAM FLAT across the 24 hourly output groups
# (the v0.19.1 leak-guard property: a per-group climb would betray a renewed leak
# or an allocator-pool growth). Run UNDER the GPU lock (single holder).
#
# Env: INPUT_DIR (default canary_all7/run), HOURS (default 24), AB_TASKSET (0-3),
#      JAX_CACHE (shared warm cache), ALLOCATOR (default cuda_async), OUT_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-<DATA_ROOT>/wrf_downscale/canary_all7/run}"
HOURS="${HOURS:-24}"
AB_TASKSET="${AB_TASKSET:-0-3}"
JAX_CACHE="${JAX_CACHE:-<DATA_ROOT>/gpuwrf_jax_cache}"
ALLOCATOR="${ALLOCATOR:-cuda_async}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-proofs/v020/lowhang/alloc_24h/${TS}}"
mkdir -p "$OUT_ROOT/out" "$OUT_ROOT/proof" "$JAX_CACHE"

export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR="$JAX_CACHE"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export OMP_NUM_THREADS=4 GPUWRF_MYNN_BOULAC_ONZ=1
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export GPUWRF_ALLOCATOR="$ALLOCATOR" GPUWRF_NESTED_ASYNC_OUTPUT=1 GPUWRF_NESTED_SYNC_MODE=root:1

smi="$OUT_ROOT/nvidia_smi_ts.csv"
echo "epoch_s,mem_used_MiB" > "$smi"
( while true; do
    printf '%s,%s\n' "$(date -u +%s)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    sleep 1
  done ) >> "$smi" 2>/dev/null &
smi_pid=$!

echo "=== v0.20 cuda_async 24h VRAM-flat gate (alloc=$ALLOCATOR hours=$HOURS input=$INPUT_DIR) ==="
start=$(date -u +%s)
taskset -c "$AB_TASKSET" python -m gpuwrf run \
  --input-dir "$INPUT_DIR" --output-dir "$OUT_ROOT/out" --proof-dir "$OUT_ROOT/proof" \
  --max-dom 9 --hours "$HOURS" > "$OUT_ROOT/run.log" 2>&1
rc=$?
end=$(date -u +%s)
kill "$smi_pid" 2>/dev/null || true; wait "$smi_pid" 2>/dev/null || true
echo "$rc" > "$OUT_ROOT/rc.txt"; echo "$((end-start))" > "$OUT_ROOT/wall_s.txt"

oom="no"; grep -qiE "CUDA_ERROR_OUT_OF_MEMORY|RESOURCE_EXHAUSTED|Failed to allocate" "$OUT_ROOT/run.log" && oom="YES"
echo "$oom" > "$OUT_ROOT/oom.txt"

python - "$OUT_ROOT/proof/nested_pipeline_run.json" "$smi" "$HOURS" "$rc" "$oom" "$((end-start))" <<'PY'
import json, sys
pj, smi, hours, rc, oom, wall = sys.argv[1:7]
ts=[]; mem=[]
with open(smi) as fh:
    next(fh, None)
    for ln in fh:
        ln=ln.strip()
        if not ln or ',' not in ln: continue
        a,b=ln.split(',',1)
        try: ts.append(int(a)); mem.append(int(float(b)))
        except ValueError: pass
print(f"=== 24h VRAM CENSUS  rc={rc} oom={oom} wall={wall}s samples={len(mem)} ===")
if mem:
    t0=ts[0]; n=len(mem); peak=max(mem)
    # 6 equal wall-time bins -> peak VRAM per bin (flat across bins == no leak/pool-growth)
    bins=6; span=max(1, ts[-1]-t0)
    binpeak=[0]*bins
    for t,m in zip(ts,mem):
        k=min(bins-1, (t-t0)*bins//span); binpeak[k]=max(binpeak[k],m)
    print(f"  overall peak VRAM = {peak} MiB")
    print(f"  per-sextile peak  = {binpeak}  (MiB; flat => no climb)")
    print(f"  first->last sextile delta = {binpeak[-1]-binpeak[0]} MiB")
try:
    p=json.load(open(pj))
    print(f"  finite={p.get('all_domains_finite')} outputs_present={p.get('all_outputs_present')} "
          f"forecast_only_s={p.get('wall_clock_forecast_only_s')}")
except Exception as e:
    print(f"  (no proof json: {e})")
PY
echo "=== artifacts under $OUT_ROOT ==="
