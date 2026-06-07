#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_switzerland_cpu_reference.sh — produce the CPU-WRF reference forecast for
# the Switzerland (Gotthard) equivalence case.  *** MANAGER / maintainer step.***
#
# A normal user does NOT run this: they use the published compact reference (see
# scripts/equivalence_switzerland.sh + docs/equivalence-switzerland.md). This
# script is how the published reference is produced ONCE.
#
# It runs the SERIAL pristine gfortran WRF (wrf.exe) on the same wrfinput_d01 +
# wrfbdy_d01 + namelist.input that the GPU port consumes, in ${CASE_ROOT}/run_cpu,
# times the wall clock, and validates the wrfout history.
#
# CPU budget: serial 3 km / 42x42 / 45 lvl / 24 h ≈ tens of minutes (single
# core). Drop FCST_HOURS to 12 by re-running build_switzerland_case.sh with
# FCST_HOURS=12 if you want it shorter.
#
# Pin to cores 0-3 to leave the rest of the machine free:
#     taskset -c 0-3 bash scripts/run_switzerland_cpu_reference.sh
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

# Maintainer step. CASE_ROOT/WRF default to overridable paths; set them for your
# machine. CASE_ROOT defaults to a repo-relative writable dir (no /mnt required);
# WRF must point at YOUR serial gfortran WRF build (real.exe/wrf.exe).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_ROOT="${CASE_ROOT:-${REPO}/runs/switzerland}"
CPU_DIR="${CPU_DIR:-${CASE_ROOT}/run_cpu}"
WRF="${WRF:-${WRF_ROOT:-/path/to/your/WRF}}"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
die(){ echo "[$(ts)] ERROR: $*" >&2; exit 1; }

[[ -s "${CPU_DIR}/wrfinput_d01" ]] || die "missing ${CPU_DIR}/wrfinput_d01 — run scripts/build_switzerland_case.sh first"
[[ -s "${CPU_DIR}/wrfbdy_d01"   ]] || die "missing ${CPU_DIR}/wrfbdy_d01"
[[ -s "${CPU_DIR}/namelist.input" ]] || die "missing ${CPU_DIR}/namelist.input"
[[ -x "${WRF}/main/wrf.exe" ]] || die "missing ${WRF}/main/wrf.exe"

log "── CPU-WRF reference forecast (serial gfortran) ──"
log "  run dir: ${CPU_DIR}"

# Clean any prior wrfout so the run is fresh and the comparator sees only this run.
rm -f "${CPU_DIR}"/wrfout_d01_* "${CPU_DIR}"/rsl.* "${CPU_DIR}"/wrf.log

T0=$(date +%s.%N)
( cd "$CPU_DIR" && "${WRF}/main/wrf.exe" > wrf.log 2>&1 )
RC=$?
T1=$(date +%s.%N)
WALL=$(python3 -c "print(${T1}-${T0})")
echo "$WALL" > "${CPU_DIR}/cpu_wall_seconds.txt"

N=$(find "${CPU_DIR}" -maxdepth 1 -name 'wrfout_d01_*' | wc -l)
log "  wrf.exe rc=${RC}  wrfout files=${N}  wall=${WALL}s"
if [[ $N -lt 2 ]]; then
  log "  tail of wrf.log:"; tail -25 "${CPU_DIR}/wrf.log" 2>/dev/null
  tail -10 "${CPU_DIR}"/rsl.error.0000 2>/dev/null
  die "CPU-WRF produced <2 wrfout files (SUCCESS marker not reached?)"
fi
grep -q "SUCCESS COMPLETE WRF" "${CPU_DIR}/wrf.log" 2>/dev/null \
  && log "  WRF reported SUCCESS COMPLETE" \
  || log "  NOTE: 'SUCCESS COMPLETE WRF' not found in wrf.log (check tail)"

# Quick finiteness check on the last frame.
python3 - "$CPU_DIR" << 'PY'
import sys, glob, numpy as np
from netCDF4 import Dataset
d=sys.argv[1]
f=sorted(glob.glob(f"{d}/wrfout_d01_*"))[-1]
ds=Dataset(f)
ok=True
for v in ("T2","U10","V10","PSFC","T","U","V","QVAPOR"):
    if v in ds.variables:
        a=np.asarray(ds.variables[v][:],dtype=np.float64)
        fin=np.isfinite(a).all()
        print(f"  {v:8s} {f.split('/')[-1]} finite={fin} min={a.min():.3g} max={a.max():.3g}")
        ok = ok and fin
print("  LAST-FRAME FINITE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 3)
PY
log "═══ CPU reference complete: ${CPU_DIR} (wall ${WALL}s in cpu_wall_seconds.txt) ═══"
