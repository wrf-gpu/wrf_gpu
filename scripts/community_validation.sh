#!/usr/bin/env bash
#
# community_validation.sh -- outsider-facing CPU community-standard validation suite.
#
# Assembles the community-standard validation evidence an external reviewer
# expects from a WRF reimplementation, all reproducible on a CPU-only machine
# with NOTHING beyond this repository:
#
#   1. Idealized dycore gates -- Straka 1993 density current + Skamarock /
#      Bryan-Fritsch warm bubble, re-run on CPU via the existing idealized
#      runner and checked against the published WRF benchmark spec.
#   2. Closed-domain conservation budgets -- dry-mass / total-water /
#      moist-static-energy budget closure (relative residual ~0 in fp64).
#   3. Bitwise restart -- full state+carry+stochastic-seed NetCDF wrfrst
#      write->read->compare bit-identity round-trip.
#
# It REUSES the existing proof generators / tests (it does not reimplement or
# destructively modify them) and emits a single aggregator proof object
# (proofs/v013/community_validation.json). The honest CPU-vs-GPU/data gap list
# (what an outsider runs on CPU vs what needs a GPU or the purged CPU-WRF
# corpus) is printed at the end and documented in docs/VALIDATION.md.
#
# CPU-only by construction: JAX_PLATFORMS=cpu is forced; no GPU context is
# created (the GPU is owned by other lanes). This is the community-validation
# companion to scripts/verify_reproducibility.sh.
#
# Usage:   bash scripts/community_validation.sh
# Exit 0 = PASS (every CPU community gate green); non-zero = FAIL.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export JAX_PLATFORMS="cpu"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PY="${PYTHON:-python3}"

# Pin Python/JAX to cores 0-3 if taskset is available (workstation convention;
# cores 4-31 are reserved for the nightly CPU-WRF runs and must not be touched).
RUN=("$PY")
if command -v taskset >/dev/null 2>&1; then
  RUN=(taskset -c 0-3 "$PY")
fi

log="$(mktemp -t wrf_gpu_community_validation.XXXXXX.log)"
out="${ROOT}/proofs/v013/community_validation.json"
echo "community_validation: log=$log  root=$ROOT"
echo "JAX_PLATFORMS=$JAX_PLATFORMS  (CPU only; no GPU context is created)"
echo "Aggregator proof object -> $out"
echo

echo "[1/1] running idealized + closed-domain conservation + bitwise-restart gates (CPU)"
echo "      (the two idealized dycore integrations take ~3 min on 4 CPU cores)"
{ echo "===== community_validation aggregator ====="; \
  "${RUN[@]}" proofs/v013/community_validation.py --output "$out"; } 2>&1 | tee -a "$log"
rc="${PIPESTATUS[0]}"
echo

# ----------------------------------------------------------------------------
# Per-gate summary parsed from the emitted proof object.
# ----------------------------------------------------------------------------
echo "=================== SUMMARY ==================="
if [ -f "$out" ]; then
  "${RUN[@]}" - "$out" <<'PYEOF'
import json, sys
proof = json.loads(open(sys.argv[1]).read())
def fmt(b): return "PASS" if b else "FAIL"
for name, gate in proof["gates"].items():
    print(f"  {fmt(gate['pass']):4s}  {name}")
    if name == "idealized_dycore":
        for ck, cv in gate["cases"].items():
            print(f"          {fmt(cv['pass']):4s}  {ck}  (verdict={cv['verdict']}, {cv['wall_s']}s)")
    if name == "closed_domain_conservation":
        r = gate.get("closed_domain_residuals", {})
        print(f"          dry-mass rel-residual   = {r.get('dry_mass_relative_residual')}")
        print(f"          total-water rel-residual= {r.get('total_water_relative_residual')}")
        print(f"          moist-static-energy res = {r.get('moist_static_energy_residual_j')} J")
    if name == "bitwise_restart":
        print(f"          full-carry bit-identical      = {gate.get('bit_identical_full_carry')}")
        print(f"          stochastic-seed bit-identical = {gate.get('bit_identical_stochastic_seeds')}")
print()
print(f"  OVERALL: {fmt(proof['pass'])}")
PYEOF
else
  echo "  (no proof object produced -- see $log)"
fi
echo

# ----------------------------------------------------------------------------
# Honest CPU-vs-GPU/data gap list.
# ----------------------------------------------------------------------------
cat <<'GAPS'
=============== CPU-vs-GPU/DATA GAP (see docs/VALIDATION.md) ===============
CPU-reproducible from this repo alone (run above + scripts/verify_reproducibility.sh):
  * Idealized dycore gates (Straka density current, Skamarock/Bryan-Fritsch warm bubble).
  * Closed-domain dry-mass / total-water / moist-static-energy budget closure (fp64).
  * Bitwise restart: full state+carry+stochastic-seed wrfrst write->read bit-identity.
  * CPU physics savepoint-parity proofs.

Needs an NVIDIA GPU (out of scope here):
  * Speedup / throughput / per-watt and multi-GPU (DGX) claims (+ profiler artifacts).
  * 1km nested live-forecast stability gates (d03) and GWD-nested gates.
  * Multi-hour restart forecast-CONTINUITY acceptance (restart trajectory == uninterrupted);
    the structural bit-identity above is CPU, the trajectory match needs GPU + corpus.

Needs the purged CPU-WRF corpus (not redistributable; /mnt/data):
  * TOST operational equivalence vs 28-rank CPU-WRF (proofs/m20/*).
  * Multi-day operational skill-vs-obs gates and station scoring.
GAPS
echo

if [ "$rc" -eq 0 ]; then
  echo "PASS community_validation (all CPU community gates green) log=$log proof=$out"
else
  echo "FAIL community_validation (one or more CPU gates failed) log=$log proof=$out"
fi
exit "$rc"
