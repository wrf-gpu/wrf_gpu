#!/usr/bin/env bash
# proofs/v017/rainnc_fix/test_rainnc_fix.sh
#
# MANAGER-RUN-ONLY (under the GPU lock). Do NOT run unattended; this launches a
# GPU Switzerland d01 forecast. The investigating agent did NOT run this.
#
# Purpose: validate whether a candidate Thompson-microphysics RAINNC fix
#   (1) brings Switzerland d01 RAINNC inside the strict identity tolerance, AND
#   (2) does NOT regress any of the other 9 already-green gate fields.
#
# It runs a GPU cpu_wrf_replay forecast against the SAME WRF V4.7.1 CPU truth
# used by the v0.17 identity gate, builds the Grid-Delta Atlas with the FROZEN
# tolerance manifest, and prints a PASS/FAIL adjudication.
#
# IMPORTANT: there is currently NO harmless fix committed (see report.md). This
# script is the acceptance harness for any FUTURE candidate. Run the BASELINE
# config first (HOURS=24 is enough to expose the snow/graupel partition deficit
# and confirm the 9 fields' margins) to confirm the harness reproduces the
# known RAINNC FAIL, then run again with the candidate branch checked out.
#
# Usage:
#   bash proofs/v017/rainnc_fix/test_rainnc_fix.sh           # default 24h probe
#   HOURS=72 bash proofs/v017/rainnc_fix/test_rainnc_fix.sh  # full 72h gate
#
set -euo pipefail

# ---- pin to cores 0-3 (12-rank CPU job owns 4-15) ----
CORES="${CORES:-0-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# ---- repo / worktree root (run from a checkout of the candidate branch) ----
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# ---- fixed identity-gate config (matches the v0.17 fast-compile identity run) ----
export GPUWRF_REPLAY_SEGMENTED="${GPUWRF_REPLAY_SEGMENTED:-1}"   # bit-identical fast compile
export GPUWRF_MYNN_BOULAC_ONZ="${GPUWRF_MYNN_BOULAC_ONZ:-1}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
# Leave Thompson levers at DEFAULT (riming ON, cold-collection ON). A candidate
# fix should be the new default, or set its own env gate here explicitly.

HOURS="${HOURS:-24}"
DOMAIN="d01"
CPU_TRUTH="${CPU_TRUTH:-<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu}"
TOL_MANIFEST="${TOL_MANIFEST:-$REPO/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${RUN_ROOT:-<DATA_ROOT>/wrf_gpu_validation/v017_rainnc_fix_test_${STAMP}}"
GPU_OUT="$RUN_ROOT/gpu_output"
ATLAS="$RUN_ROOT/grid_delta_atlas"
SCRATCH="$RUN_ROOT/scratch"
mkdir -p "$GPU_OUT" "$ATLAS" "$SCRATCH"

echo "== RAINNC fix acceptance test =="
echo "repo            = $REPO"
echo "branch          = $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "hours           = $HOURS"
echo "cpu truth       = $CPU_TRUTH   (WRF V4.7.1, mp_physics=8)"
echo "tolerance       = $TOL_MANIFEST  (FROZEN manifest)"
echo "run root        = $RUN_ROOT"
echo

# ---- 1. GPU forecast (cpu_wrf_replay; IC + LBC from the CPU truth history) ----
echo "== [1/3] GPU forecast (cores $CORES) =="
taskset -c "$CORES" python -m gpuwrf.cli run \
  --init-mode cpu_wrf_replay \
  --cpu-wrf-dir "$CPU_TRUTH" \
  --domain "$DOMAIN" \
  --hours "$HOURS" \
  --output-dir "$GPU_OUT" \
  --scratch-dir "$SCRATCH" \
  2>&1 | tee "$RUN_ROOT/gpu.log"
# NOTE: confirm the exact `gpuwrf.cli run` flag names against this branch's
# `python -m gpuwrf.cli run --help`; the v0.17 identity run used the
# cpu_wrf_replay path (see daily_pipeline.execute_daily_pipeline).

# ---- 2. Grid-Delta Atlas vs the CPU truth, FROZEN tolerance manifest ----
echo "== [2/3] Grid-Delta Atlas =="
taskset -c "$CORES" python scripts/build_grid_delta_atlas.py \
  --cpu-dir "$CPU_TRUTH" \
  --gpu-dir "$GPU_OUT" \
  --case-id "switzerland_d01_rainnc_fix_test" \
  --domain "$DOMAIN" \
  --tolerance-json "$TOL_MANIFEST" \
  --proof-dir "$ATLAS" \
  --asset-dir "$ATLAS/assets" \
  2>&1 | tee "$RUN_ROOT/atlas.log"

# ---- 3. Adjudication: RAINNC strict-green AND no regression on the other 9 ----
echo "== [3/3] Adjudication =="
taskset -c "$CORES" python - "$ATLAS/grid_delta_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
verdict = summary.get("verdict")
fails = summary.get("tolerance_failures", [])
print(f"atlas verdict        : {verdict}")
print(f"tolerance failures   : {summary.get('tolerance_failure_count')}")

# The 10 strict gate fields (RAINNC + the 9 already-green).
GATE = ["RAINNC","T2","U10","V10","PSFC","Q2","QVAPOR","T","U","V","W"]
fail_fields = {f["field"]: f["failures"] for f in fails}

rainnc_green = "RAINNC" not in fail_fields
# regression = any OTHER gate field now failing
regressed = sorted(f for f in fail_fields if f in GATE and f != "RAINNC")

print()
for f in fail_fields:
    for d in fail_fields[f]:
        print(f"  FAIL {f:10s} {d.get('metric')}={d.get('value'):.4f} > {d.get('limit')}")

print()
if rainnc_green and not regressed:
    print("RESULT: PASS  -- RAINNC strict-green AND no regression on the other 9. MERGE candidate.")
    sys.exit(0)
elif not rainnc_green and not regressed:
    print("RESULT: FAIL  -- RAINNC still outside tolerance (no fix / insufficient fix). DO NOT MERGE.")
    sys.exit(1)
else:
    print(f"RESULT: REJECT -- regressed gate field(s): {regressed}. NOT harmless. DO NOT MERGE.")
    sys.exit(2)
PY
echo "artifacts under: $RUN_ROOT"
