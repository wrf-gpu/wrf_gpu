#!/usr/bin/env bash
# v020_g0_decision.sh — G0 fp32-vs-fp64 DECISION probe at DRAM-bound scale.
#
# THE decisive ~1-GPU-session probe of FINAL_FP32_SPRINT_PLAN §2 (hardened by
# V0200-ROADMAP §8.4). On the largest DRAM-bound single grid that fp64 FITS (the
# v0.17 bigswiss 460x460x44 ~211k cols; working set >> 96 MiB L2, so fp32's
# bandwidth/ALU edge can physically appear), it runs a WARM fp64-vs-fp32 A/B and emits
# a GO / KILL verdict.
#
# HOW THE TWO ARMS DIFFER (no production source edit — collision-safe):
#   * fp64 arm: JAX_ENABLE_X64=true  (the production default; the baseline).
#   * fp32 arm: JAX_ENABLE_X64=false + GPUWRF_THOMPSON_FP32=1  == the §2 "AGGRESSIVE"
#     variant: with x64 OFF, JAX canonicalises ALL arrays (p'/ph'/mu'/w storage AND the
#     PCR tridiagonal solve) to fp32 end-to-end. This is the cheapest faithful proxy for
#     "downcast p'/ph'/mu'/w + drop the fp64 PCR solve to fp32" WITHOUT touching dycore
#     source (which another worker owns). It is the AGGRESSIVE arm by design; a GO here
#     justifies the surgical per-field rewrite, a KILL ends it.
#
# MEASURES per arm: warm ms/step (wall / advance count), peak VRAM incl. transient
# (nvidia-smi sampler), dominant-kernel wall (nsys top kernel), did-it-fit. The
# transient fp32-able fraction + bottleneck-moved flag are filled by the operator from
# the dtype/liveness audit (the driver leaves a clearly-marked hook + a conservative
# default).
#
# Then scripts/v020_g0_verdict.py reduces the manifest to G0-SPEED-GO / G0-CAPABILITY-GO
# / G0-KILL with the exact action.
#
# GPU SAFETY: each arm's forecast is wrapped in with_gpu_lock.sh via v020_run_gpu and
# SKIPPED under V020_DRYRUN=1 (CPU dry-run validates arg-parse + the template + the
# verdict reducer on a synthetic manifest).
#
# Usage:
#   scripts/v020_g0_decision.sh                  # FULL A/B (holds GPU lock, ~1 session)
#   V020_DRYRUN=1 scripts/v020_g0_decision.sh    # CPU dry-run (no GPU)
# Env: V020_BIG_INPUT (case), V020_G0_HOURS (default 1; warm window), V020_G0_DURATION.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v020_probe_common.sh"

: "${V020_G0_HOURS:=1}"        # warm forecast hours per arm (short; this is a probe)
: "${V020_G0_DURATION:=1200}"  # nsys bounded capture per arm (s)
INPUT="$V020_BIG_INPUT"

v020_assert_case "$INPUT" || { v020_log "G0 abort: missing DRAM-scale case $INPUT"; exit 1; }

RR="$(v020_mk_rundir g0_decision)"
mkdir -p "$RR"
v020_log "G0 run dir: $RR  (dryrun=$V020_DRYRUN)  grid input=$INPUT"

cat > "$RR/runinfo.txt" <<EOF
probe=G0_fp32_vs_fp64_decision
grid=DRAM_bound_single_large (v017 bigswiss 460x460x44 ~211k cols, working set >> 96MiB L2)
input=$INPUT
fp64_arm=JAX_ENABLE_X64=true (production baseline)
fp32_arm=JAX_ENABLE_X64=false + GPUWRF_THOMPSON_FP32=1 (aggressive global-fp32 proxy; NO source edit)
hours_per_arm=$V020_G0_HOURS
criteria=FINAL_FP32_SPRINT_PLAN.md §2 + V0200-ROADMAP.md §8.4 (three-outcome split)
EOF

# ----------------------------------------------------------------------------------
# run_arm NAME X64FLAG EXTRA_ENV...
#   runs one warm forecast under nsys + a VRAM sampler, then reduces to an arm JSON.
# ----------------------------------------------------------------------------------
run_arm() {
  local name="$1"; local x64="$2"; shift 2
  local extra_env=("$@")
  local out="$RR/${name}_out"; local proofs="$RR/${name}_proofs"
  local scratch="$RR/${name}_scratch"; local log="$RR/${name}.log"
  local vram="$RR/${name}_vram.csv"; local nrep="$RR/${name}_nsys"
  local sprefix="$RR/${name}_stats"
  mkdir -p "$out" "$proofs" "$scratch"

  local GPU_CMD=(
    env
    PYTHONPATH="$V020_ROOT/src"
    JAX_ENABLE_X64="$x64"
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_ENABLE_COMPILATION_CACHE=true
    JAX_COMPILATION_CACHE_DIR="$V020_JAX_CACHE"
    OMP_NUM_THREADS="$V020_OMP"
    GPUWRF_MYNN_BOULAC_ONZ=1
    XLA_PYTHON_CLIENT_ALLOCATOR=platform
    GPUWRF_SCRATCH="$scratch"
    "${extra_env[@]}"
    nsys profile
      --trace=cuda,nvtx,osrt
      --sample=none --cpuctxsw=none
      --cuda-memory-usage=false
      --duration="$V020_G0_DURATION"
      --force-overwrite=true
      --output="$nrep"
    taskset -c "$V020_TASKSET"
    python -m gpuwrf run
      --input-dir "$INPUT"
      --namelist "$INPUT/namelist.input"
      --output-dir "$out"
      --proof-dir "$proofs"
      --scratch-dir "$scratch"
      --max-dom 1
      --domain "${V020_G0_DOMAIN:-d01}"
      --hours "$V020_G0_HOURS"
  )

  # background VRAM sampler ONLY in real mode
  local vpid=""
  if [[ "$V020_DRYRUN" != "1" ]]; then
    ( while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
          --format=csv,noheader 2>/dev/null >> "$vram"; sleep 3
      done ) & vpid=$!
  fi

  local start end wall
  start=$(date -u +%s)
  v020_run_gpu "opus-g0-$name" "$((V020_G0_DURATION + 3600))" -- "${GPU_CMD[@]}" \
    > "$log" 2>&1
  local rc=$?
  end=$(date -u +%s); wall=$((end - start))
  [[ -n "$vpid" ]] && kill "$vpid" 2>/dev/null || true
  echo "$rc"   > "$RR/${name}.rc"
  echo "$wall" > "$RR/${name}.wall_s"
  v020_log "G0 arm '$name' rc=$rc wall=${wall}s (0 in dry-run = skipped)"

  # nsys stats export (real mode + capture present)
  if [[ "$V020_DRYRUN" != "1" && -f "$nrep.nsys-rep" ]]; then
    nsys stats --force-overwrite=true --force-export=true \
      --report cuda_gpu_kern_sum --report cuda_api_sum \
      --format csv --output "$sprefix" "$nrep.nsys-rep" \
      > "$RR/${name}_stats.log" 2>&1 || true
  fi

  # 'fit' = the run exited 0 AND no OOM token in the log (real mode); template-true in dry-run
  local fit="true"
  if [[ "$V020_DRYRUN" != "1" ]]; then
    if [[ "$rc" != "0" ]] || grep -qiE "RESOURCE_EXHAUSTED|out of memory|OOM|CUDA_ERROR_OUT_OF_MEMORY" "$log"; then
      fit="false"
    fi
  fi

  # transient fp32-able fraction: REAL value comes from the operator's dtype/liveness
  # audit (the verdict treats unknown conservatively). The fp32 arm carries the hook.
  local frac_args=()
  if [[ "$name" == "fp32" ]]; then
    # default -1 (= unknown -> NaN -> conservative). The operator overrides with the
    # measured share via V020_G0_FP32_TRANSIENT_FRAC after reading the HLO/liveness audit.
    frac_args=(--transient-fp32-frac "${V020_G0_FP32_TRANSIENT_FRAC:--1}")
    [[ "${V020_G0_BOTTLENECK_MOVED:-false}" == "true" ]] && frac_args+=(--bottleneck-moved true)
  fi

  taskset -c "$V020_TASKSET" python "$HERE/v020_run_metrics.py" \
    --wall-s "$wall" --proof-dir "$proofs" \
    --vram-csv "$vram" --stats-prefix "$sprefix" \
    --fit "$fit" "${frac_args[@]}" \
    --out "$RR/${name}_arm.json" \
    > "$RR/${name}_metrics.log" 2>&1 || v020_log "metrics reduce for '$name' had no inputs (dry-run ok)"
}

# --- fp64 baseline arm, then fp32 aggressive arm ----------------------------------
run_arm fp64 true
run_arm fp32 false GPUWRF_THOMPSON_FP32=1

# --- assemble the manifest --------------------------------------------------------
# In dry-run there are no measured arm JSONs -> use the template so the verdict reducer
# can be exercised end-to-end; print a clear DRY banner.
MANIFEST="$RR/g0_manifest.json"
if [[ -s "$RR/fp64_arm.json" && -s "$RR/fp32_arm.json" ]]; then
  taskset -c "$V020_TASKSET" python - "$RR" <<'PY'
import json, sys
rr = sys.argv[1]
fp64 = json.load(open(f"{rr}/fp64_arm.json"))
fp32 = json.load(open(f"{rr}/fp32_arm.json"))
manifest = {
  "grid": {"name": "v017_bigswiss_460x460x44", "cols": 211600,
           "fp64_oom_target": True,
           "_note": "is there a useful 1km/large grid fp64 OOMs on but fp32 fits? "
                    "set from the capability sub-run; default True (a 1km nest exists)."},
  "fp64": fp64, "fp32": fp32,
  "notes": {"fp32_arm": "JAX_ENABLE_X64=false + GPUWRF_THOMPSON_FP32=1 (aggressive proxy)",
            "transient_fp32_fraction_source": "operator dtype/liveness audit "
            "(V020_G0_FP32_TRANSIENT_FRAC); -1/NaN treated conservatively by the verdict"},
}
json.dump(manifest, open(f"{rr}/g0_manifest.json", "w"), indent=2)
print(f"wrote {rr}/g0_manifest.json")
PY
else
  v020_log "no measured arm JSONs -> emitting TEMPLATE manifest (dry-run / pre-capture)"
  taskset -c "$V020_TASKSET" python "$HERE/v020_g0_verdict.py" --emit-template "$MANIFEST"
fi

# --- emit the verdict --------------------------------------------------------------
taskset -c "$V020_TASKSET" python "$HERE/v020_g0_verdict.py" \
  --manifest "$MANIFEST" --out "$RR/G0_VERDICT.json" \
  | tee "$RR/G0_VERDICT.txt" || true

{
  echo "G0 decision artifacts ($RR):"
  echo "  manifest : $MANIFEST"
  echo "  verdict  : $RR/G0_VERDICT.json"
  echo "  fp64 arm : $RR/fp64_arm.json   log $RR/fp64.log"
  echo "  fp32 arm : $RR/fp32_arm.json   log $RR/fp32.log"
  if [[ "$V020_DRYRUN" == "1" ]]; then
    echo "  NOTE: DRY-RUN verdict is from the TEMPLATE (null metrics) — NOT a real decision."
  fi
} | tee "$RR/G0_SUMMARY.txt"
v020_log "G0 DONE -> $RR/G0_SUMMARY.txt"
exit 0
