#!/usr/bin/env bash
# Build the v0.13 Tier-3 GSFC/Goddard NUWRF LONGWAVE (ra_lw_physics=5)
# single-column fp64 ORACLE against the WRF phys/module_ra_goddard.F
# Chou-Suarez IR kernel ``lwrad``, and emit gold savepoints as JSON.
#
# lwrad is PRIVATE in the pristine module (only init_goddardrad/goddardrad
# are public). To call it from a standalone driver we apply a SINGLE-LINE,
# VISIBILITY-ONLY shim that adds ``public :: lwrad`` after the module's
# existing public line -- NO physics is changed; the kernel bytes are
# identical. Both the pristine checksum and the post-shim checksum are
# recorded in the manifest so the shim is auditable.
#
# The Goddard module's internal working precision (fp_kind) is already
# double, so this is the canonical fp64 reference for the (fp64) JAX
# port -- there is NO JAX-vs-JAX self-compare.
#
# NOTE: the GOCART/Mie aerosol-LUT path (read_lut_nc, `use netcdf`) lives
# in init_goddardrad / aero_opt, which the LW-no-aerosol oracle never
# calls (aerosol optics are passed zero). We compile ONLY the module +
# driver; the netcdf-dependent subroutines are present in the object but
# never invoked, so no NetCDF link is required. If a future build pulls
# them in, add `-lnetcdff` and a LUT file.
#
# CPU-only, cores 0-3. Requires the conda env `wrfbuild`.
# Usage:  goddard_lw_build_and_run.sh     # fp64 (module is already double)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
WRF_GODDARD="${WRF_PHYS}/module_ra_goddard.F"
WRF_CHECKERR="${WRF_PHYS}/module_checkerror.F"
OUT_SAVE="${HERE}/../../savepoints/radiation_lw"
SRC="${HERE}/module_ra_goddard.F"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2

mkdir -p "${OUT_SAVE}"

# --- copy pristine source + record its checksum -----------------------
cp "${WRF_GODDARD}" "${SRC}"
PRISTINE_SHA="$(sha256sum "${SRC}" | awk '{print $1}')"

# --- apply the VISIBILITY-ONLY shim (public :: lwrad) -----------------
# Insert immediately after the existing module public line. This is the
# ONLY modification; it changes a visibility attribute, not a single
# physics statement.
python3 - "${SRC}" <<'PYEOF'
import sys, io
path = sys.argv[1]
with io.open(path, "r", encoding="latin-1") as fh:
    lines = fh.readlines()
marker = "public :: init_goddardrad, goddardrad"
out, done = [], False
for ln in lines:
    out.append(ln)
    if (not done) and marker in ln:
        out.append("  public :: lwrad  !ORACLE-SHIM v0.13 (visibility only; no physics change)\n")
        done = True
if not done:
    sys.stderr.write("ERROR: could not find Goddard module public line for shim\n")
    sys.exit(2)
with io.open(path, "w", encoding="latin-1") as fh:
    fh.writelines(out)
PYEOF
SHIM_SHA="$(sha256sum "${SRC}" | awk '{print $1}')"

# --- copy the self-contained pristine module_checkerror.F -------------
cp "${WRF_CHECKERR}" "${HERE}/module_checkerror.F"

# --- build (fp64; WRF default REAL -> double via -fdefault-real-8) ----
cd "${HERE}"
# -fdefault-real-8/-fdefault-double-8 = the canonical WRF fp64 build: every
# default REAL becomes REAL(8), matching fp_kind=SELECTED_REAL_KIND(15,307),
# so the module's internal `real` work arrays and the `real(fp_kind)` lwrad
# interface are precision-consistent. The driver is compiled the same way,
# so its REAL outputs (flx_out/acflx*) are REAL(8) too.
# NO_IEEE_MODULE avoids the optional ieee_arithmetic dep. WRF_CHEM is left
# UNDEFINED -> the GOCART aerosol-coupling `use` and code are preprocessed
# out (no module_gocart_coupling / netcdf dependency).
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -fdefault-real-8 -fdefault-double-8 -DRWORDSIZE=8 -DNO_IEEE_MODULE -ffpe-summary=none -fallow-argument-mismatch -std=legacy"
# Dependency order: wrf_error stub + checkerror, then Goddard, then driver.
taskset -c 0-3 gfortran ${FFLAGS} -c goddard_lw_stubs.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_checkerror.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_goddard.F
taskset -c 0-3 gfortran ${FFLAGS} -c goddard_lw_oracle_driver.f90
taskset -c 0-3 gfortran -O2 -o goddard_lw_oracle \
    goddard_lw_stubs.o module_checkerror.o module_ra_goddard.o goddard_lw_oracle_driver.o

# --- manifest ---------------------------------------------------------
{
  echo "build_mode=fp64"
  echo "fflags=${FFLAGS}"
  echo "gfortran=$(gfortran --version | head -1)"
  echo "wrf_source=${WRF_GODDARD}"
  echo "pristine_sha256=${PRISTINE_SHA}"
  echo "shim_sha256=${SHIM_SHA}"
  echo "shim=single-line visibility-only (public :: lwrad), no physics change"
} > "${OUT_SAVE}/goddard_lw_build_manifest.txt"

# --- run regimes ------------------------------------------------------
for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./goddard_lw_oracle "$c" > "goddard_lw_case_${c}.txt"
  python3 "${HERE}/goddard_lw_dump_to_json.py" \
      "goddard_lw_case_${c}.txt" "${OUT_SAVE}/goddard_lw_case_${c}.json"
done

echo "OK: goddard LW oracle built + 6 savepoints written to ${OUT_SAVE}"
echo "    pristine_sha256=${PRISTINE_SHA}"
echo "    shim_sha256=${SHIM_SHA}"
