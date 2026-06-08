#!/usr/bin/env bash
#
# verify_reproducibility.sh -- outsider-facing CPU reproducibility gate.
#
# Runs the part of the historical proof collection that an external reviewer can
# reproduce on a CPU-only machine with NOTHING beyond this repository:
#   1. large binary assets present + sha256-matching the pinned manifest
#      (manifest/reproducibility_assets.json) -- notably the Thompson lookup
#      tables, which were the asset flagged as missing in the v0.11 critique;
#   2. asset-exercising unit tests (Thompson/RRTMG table loaders + manifest pin);
#   3. CPU-runnable physics savepoint-parity gates (JAX vs vendored WRF-oracle
#      savepoints) -- these need NO GPU, NO purged corpus, and NO WRF source tree.
#
# What this script DELIBERATELY does NOT run (and why) is printed at the end and
# documented in docs/REPRODUCIBILITY.md: GPU-only proofs (speedup, multi-GPU,
# 1km nested), and operational/TOST proofs that need purged CPU-WRF corpus.
#
# CPU-only by construction: JAX_PLATFORMS=cpu is forced; no GPU context is created.
# Honors WRF_PRISTINE_ROOT if you have a pristine WRF checkout (enables provenance
# hashing), but does not require it.
#
# Usage:   bash scripts/verify_reproducibility.sh
# Exit 0 = PASS (every CPU-reproducible gate green); non-zero = FAIL.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export JAX_PLATFORMS="cpu"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PY="${PYTHON:-python3}"

# Pin Python/JAX to cores 0-3 if taskset is available (workstation convention).
RUN=("$PY")
if command -v taskset >/dev/null 2>&1; then
  RUN=(taskset -c 0-3 "$PY")
fi

log="$(mktemp -t wrf_gpu_verify_reproducibility.XXXXXX.log)"
echo "verify_reproducibility: log=$log  root=$ROOT"
echo "JAX_PLATFORMS=$JAX_PLATFORMS  (CPU only; no GPU context is created)"
echo

declare -a NAMES=()
declare -a RESULTS=()
overall_rc=0

record() {  # record <name> <rc>
  NAMES+=("$1")
  if [ "$2" -eq 0 ]; then
    RESULTS+=("PASS")
    printf '  [PASS] %s\n' "$1"
  else
    RESULTS+=("FAIL")
    overall_rc=1
    printf '  [FAIL] %s (see %s)\n' "$1" "$log"
  fi
}

run_proof() {  # run_proof <relpath.py>
  local rel="$1"
  { echo "===== PROOF $rel ====="; "${RUN[@]}" "$rel"; } >>"$log" 2>&1
  record "proof: $rel" "$?"
}

run_pytest() {  # run_pytest <name> <test files...>
  local name="$1"; shift
  { echo "===== PYTEST $name ====="; "${RUN[@]}" -m pytest -q "$@"; } >>"$log" 2>&1
  record "pytest: $name" "$?"
}

# ----------------------------------------------------------------------------
# Tier 0 -- large binary assets present + sha256 == pinned manifest.
# This is the gate that the v0.11 critique flagged: the Thompson .npz must be
# bundled (it was missing from the v0.0.1 public tree) or the full collection
# cannot run. We verify EVERY pin_verified asset in the manifest.
# ----------------------------------------------------------------------------
echo "[Tier 0] binary asset integrity (manifest/reproducibility_assets.json)"
{ echo "===== TIER0 asset integrity ====="; "${RUN[@]}" - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
man = json.loads(Path("manifest/reproducibility_assets.json").read_text())
bad = 0
for a in man["assets"]:
    p = Path(a["path"])
    if not a.get("pin_verified", False):
        print(f"SKIP (pin not verified upstream): {a['name']} -- {a.get('pin_discrepancy','')[:80]}")
        continue
    if not p.is_file():
        print(f"MISSING: {a['path']}"); bad += 1; continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != a["checksum_sha256"]:
        print(f"SHA MISMATCH: {a['path']} got {h} want {a['checksum_sha256']}"); bad += 1
    else:
        print(f"OK {a['name']} ({a['bytes']} bytes) sha256 matches pin")
sys.exit(1 if bad else 0)
PYEOF
} >>"$log" 2>&1
record "tier0: pinned binary assets present + sha256-match" "$?"
echo

# ----------------------------------------------------------------------------
# Tier 1 -- asset-exercising unit tests (load the .npz tables, check the pin,
# check WRF-source constant values + table shapes). Pure CPU.
# ----------------------------------------------------------------------------
echo "[Tier 1] asset-exercising unit tests"
run_pytest "thompson tables + manifest pin" \
  tests/test_m5_thompson_constants.py tests/test_m5_thompson_tier1.py
run_pytest "rrtmg tables loader" \
  tests/test_m5_rrtmg_tables.py
echo

# ----------------------------------------------------------------------------
# Tier 2 -- CPU physics savepoint-parity proofs. Each validates the JAX port
# against vendored UNMODIFIED-WRF Fortran savepoints (proofs/.../savepoints/).
# No GPU, no corpus, no WRF source tree required.
# ----------------------------------------------------------------------------
echo "[Tier 2] CPU physics savepoint-parity proofs (JAX vs vendored WRF savepoints)"
run_proof proofs/b1/coupled_moist_smoke.py
run_proof proofs/v060/run_kessler_parity.py
run_proof proofs/v060/run_boulac_parity.py
run_proof proofs/v060/run_dudhia_parity.py
run_proof proofs/v060/run_rrtm_lw_parity.py
run_proof proofs/v060/run_wsm_sm_parity.py
run_proof proofs/v060/run_grellfreitas_parity.py
run_proof proofs/v060/run_tiedtke_gpubatch_parity.py
echo

# ----------------------------------------------------------------------------
# Summary + honest gap list.
# ----------------------------------------------------------------------------
echo "=================== SUMMARY ==================="
for i in "${!NAMES[@]}"; do
  printf '  %-6s %s\n' "${RESULTS[$i]}" "${NAMES[$i]}"
done
echo

cat <<'GAPS'
=================== NOT RUN HERE (needs GPU or purged data) ===================
The following parts of the proof collection are NOT CPU-reproducible from this
repo alone and are therefore out of scope for this gate (see docs/REPRODUCIBILITY.md):

  * GPU-only       proofs/perf/*, proofs/multigpu_dgx/*, proofs/v0120/* nested 1km,
                   and ANY speedup / throughput claim -> require an NVIDIA GPU.
  * Purged corpus  proofs/m20/* (TOST equivalence), multi-day operational gates,
                   and proofs/v090/*_savepoint_parity oracle dirs -> require real
                   CPU-WRF wrfout + AIFS forcing (not redistributable; /mnt/data).
  * Oracle rebuild Fortran-linked savepoint regeneration + provenance sha256 over
                   the WRF source -> set WRF_PRISTINE_ROOT to a pristine WRF v4
                   checkout. Vendored savepoints make this OPTIONAL for the gates
                   above; without it, provenance hashes record "missing".
GAPS
echo

if [ "$overall_rc" -eq 0 ]; then
  echo "PASS verify_reproducibility (all CPU-reproducible gates green) log=$log"
else
  echo "FAIL verify_reproducibility (one or more CPU gates failed) log=$log"
fi
exit "$overall_rc"
