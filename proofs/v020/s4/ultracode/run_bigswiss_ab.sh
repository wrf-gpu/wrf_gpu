#!/usr/bin/env bash
# run_bigswiss_ab.sh — DRAM-scale fp32 A/B on the v0.17 bigswiss single grid
# (461x461x45, 3km, ~212k cols, full physics: Thompson/MYNN/RRTMG). DECOUPLED
# from the broken nested full-unroll: single domain, default scan kernel, NO
# GPUWRF_THOMAS_UNROLL. Three precision arms, two horizons each (1h,2h) so
# ms/step = (wall_2h - wall_1h)/(steps_2h - steps_1h) cancels compile+load+warmup.
#
#   arm fp64        : production baseline (force_fp64=True, fp64_default).
#   arm mixed       : GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020 (S4; 4 perts fp32).
#   arm aggressive  : GPUWRF_FORCE_FP64=0 (ADR-007 DEFAULT_DTYPES: u/v/theta/qv+moisture fp32).
#
# x64 stays ON for every arm (the fp64-locked dynamics/totals need it); we change
# the surgical MODE, never the old global-x64-off proxy that fooled G0.
#
# Measures per (arm,horizon): wall_s, peak_vram_mib (nvidia-smi sampler, platform
# allocator => true peak incl RRTMG transient), did_oom, step count. GPU-locked.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
INPUT="${V020_BIG_INPUT:-<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init}"
JAX_CACHE="${V020_JAX_CACHE:-<DATA_ROOT>/gpuwrf_jax_cache}"
RR="$HERE/bigswiss_ab_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RR"
echo "[ab] input=$INPUT  rundir=$RR"

run_one() {  # arm horizon extra_env...
  local arm="$1"; local hours="$2"; shift 2
  local tag="${arm}_${hours}h"
  local out="$RR/$tag/out"; local proofs="$RR/$tag/proofs"; local scratch="$RR/$tag/scratch"
  local log="$RR/$tag.log"; local vram="$RR/$tag.vram.csv"
  mkdir -p "$out" "$proofs" "$scratch"

  # background VRAM sampler
  ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader 2>/dev/null >> "$vram"; sleep 1; done ) &
  local vpid=$!

  local start end wall rc
  start=$(date -u +%s.%N)
  env PYTHONPATH="$ROOT/src" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      JAX_ENABLE_COMPILATION_CACHE=true \
      JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" \
      OMP_NUM_THREADS=4 \
      GPUWRF_MYNN_BOULAC_ONZ=1 \
      "$@" \
      taskset -c 0-3 python -m gpuwrf run \
        --input-dir "$INPUT" \
        --namelist "$INPUT/namelist.input" \
        --output-dir "$out" --proof-dir "$proofs" --scratch-dir "$scratch" \
        --max-dom 1 --domain d01 --hours "$hours" \
      > "$log" 2>&1
  rc=$?
  end=$(date -u +%s.%N)
  kill "$vpid" 2>/dev/null || true
  wall=$(echo "$end - $start" | bc -l)

  local peak
  peak=$(grep -oE "[0-9]+ MiB" "$vram" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  local oom="false"
  grep -qiE "RESOURCE_EXHAUSTED|out of memory|CUDA_ERROR_OUT_OF_MEMORY|RESOURCE EXHAUSTED" "$log" && oom="true"
  [[ "$rc" != "0" ]] && { grep -qiE "RESOURCE_EXHAUSTED|out of memory" "$log" && oom="true"; }

  printf '{"arm":"%s","hours":%s,"rc":%s,"wall_s":%.2f,"peak_vram_mib":%s,"oom":%s}\n' \
    "$arm" "$hours" "$rc" "$wall" "${peak:-null}" "$oom" | tee "$RR/$tag.metric.json"
  echo "[ab] $tag rc=$rc wall=${wall}s peak=${peak:-?}MiB oom=$oom"
}

# Order: fp64 (warm cache) first to validate the harness, then mixed, then aggressive.
# Each arm: 1h then 2h.
run_one fp64       1
run_one fp64       2
run_one mixed      1   GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020
run_one mixed      2   GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020
run_one aggressive 1   GPUWRF_FORCE_FP64=0
run_one aggressive 2   GPUWRF_FORCE_FP64=0

# reduce
python3 - "$RR" <<'PY'
import json, glob, os, sys
rr = sys.argv[1]
m = {}
for f in glob.glob(f"{rr}/*.metric.json"):
    d = json.load(open(f))
    m[(d["arm"], d["hours"])] = d
DT = 18.0
def steps(h): return int(h*3600/DT)
arms = ["fp64","mixed","aggressive"]
out = {"grid":"v017_bigswiss_461x461x45_3km_212kcol","dt_s":DT,"steps_per_h":steps(1),"arms":{}}
base_ms = None
for a in arms:
    r1 = m.get((a,1)); r2 = m.get((a,2))
    if not r1 or not r2: continue
    dwall = r2["wall_s"] - r1["wall_s"]; dsteps = steps(2)-steps(1)
    ms = 1000.0*dwall/dsteps if dsteps else None
    peak = max(x for x in [r1.get("peak_vram_mib"), r2.get("peak_vram_mib")] if x)
    out["arms"][a] = {
        "ms_per_step": round(ms,2) if ms else None,
        "wall_1h_s": round(r1["wall_s"],1), "wall_2h_s": round(r2["wall_s"],1),
        "peak_vram_mib": peak, "oom_1h": r1["oom"], "oom_2h": r2["oom"],
        "rc_1h": r1["rc"], "rc_2h": r2["rc"],
    }
    if a=="fp64": base_ms = ms; base_v = peak
for a in arms:
    if a in out["arms"] and base_ms:
        ar=out["arms"][a]
        if ar["ms_per_step"]:
            ar["speedup_vs_fp64"] = round(base_ms/ar["ms_per_step"],3)
        if ar["peak_vram_mib"]:
            ar["vram_reduction_vs_fp64_pct"] = round(100*(1-ar["peak_vram_mib"]/base_v),1)
json.dump(out, open(f"{rr}/SUMMARY.json","w"), indent=2)
print(json.dumps(out, indent=2))
PY
echo "[ab] DONE -> $RR/SUMMARY.json"
