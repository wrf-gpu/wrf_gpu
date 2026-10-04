#!/usr/bin/env bash
# Build the v0.23 F2 NSSL 2-moment (mp_physics=18) single-column oracle
# against the UNMODIFIED WRF source (phys/module_mp_nssl_2mom.F) and emit
# gold savepoints as JSON.
#
# CPU-only, pinned to cores 4-31 (cores 0-3 + the GPU are reserved for
# another agent's lane). Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: module_mp_nssl_2mom.F is copied VERBATIM (unmodified) from
# <USER_HOME>/src/wrf_pristine/WRF. The module has NO `USE` dependencies;
# the only project-authored Fortran is:
#   - nssl2mom_oracle_driver.f90 (column builder + dump; never touches physics)
#   - nssl2mom_stub_modules.f90  (no-op wrf_dm_on_monitor logging shim +
#                                 wrf_error_fatal abort shim; NO physics)
# nssl_2mom_init does an internal `open(...,status='old')` +
# `read(NML=nssl_mp_params)` of ./namelist.input, so an EMPTY
# namelist.input is provided in the build dir (read gets iostat/=0 and all
# module defaults are kept -- same as a WRF run whose namelist.input has
# no &nssl_mp_params group).
set -o pipefail   # NOT -e/-u: conda activate hooks reference unset vars.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
OUT_SAVE="${HERE}/../../../v022/f2_oracles/nssl_2mom"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=4

# MODE: "fp32" (default; WRF default real kind = single precision) or
# "fp64" (-fdefault-real-8 -fdefault-double-8: promotes all default REALs
# and literals to 64-bit while keeping the module's explicit DOUBLE
# PRECISION internals at 64-bit; scheme source otherwise unmodified).
MODE="${1:-fp32}"
PROMOTE=""
PREFIX="nssl"
if [ "${MODE}" = "fp64" ]; then
  PROMOTE="-fdefault-real-8 -fdefault-double-8"
  PREFIX="nssl_fp64"
fi
BLDDIR="${HERE}/build_${MODE}"

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}" "${OUT_SAVE}"

# copy pristine WRF source verbatim
cp "${WRF_PHYS}/module_mp_nssl_2mom.F"     "${BLDDIR}/module_mp_nssl_2mom.F"
cp "${HERE}/nssl2mom_stub_modules.f90"     "${BLDDIR}/nssl2mom_stub_modules.f90"
cp "${HERE}/nssl2mom_oracle_driver.f90"    "${BLDDIR}/nssl2mom_oracle_driver.f90"

# record exact source checksums (provenance) of the UNMODIFIED scheme source
( cd "${BLDDIR}" && sha256sum module_mp_nssl_2mom.F nssl2mom_stub_modules.f90 \
    nssl2mom_oracle_driver.f90 > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
# -ffree-line-length-none for the long .F lines; -cpp for the cpp guards:
# WRF_CHEM undefined -> wrfchem_flag=0 (chem branches excluded),
# NMM_CORE undefined -> invertccn=.false. (EM core), WRF_ELEC undefined ->
# electrification path excluded. All match the default WRF EM build.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp ${PROMOTE}"
taskset -c 4-31 gfortran ${FFLAGS} -c nssl2mom_stub_modules.f90   || exit 11
taskset -c 4-31 gfortran ${FFLAGS} -c module_mp_nssl_2mom.F       || exit 12
taskset -c 4-31 gfortran ${FFLAGS} -c nssl2mom_oracle_driver.f90  || exit 13
taskset -c 4-31 gfortran ${FFLAGS} -o nssl2mom_oracle \
    nssl2mom_stub_modules.o module_mp_nssl_2mom.o \
    nssl2mom_oracle_driver.o                                      || exit 14

# empty namelist.input: nssl_2mom_init's internal namelist read finds no
# nssl_mp_params group and keeps all module defaults (WRF-default path).
: > namelist.input

for c in 1 2 3 4 5 6; do
  taskset -c 4-31 ./nssl2mom_oracle "$c" > "case_${c}.txt" || { echo "RUN FAIL case $c"; exit 20; }
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/${PREFIX}_case_${c}.json" || exit 21
done

# build manifest (provenance for this mode)
{
  echo "oracle: nssl2mom single-column (WRF mp_physics=18 default config)"
  echo "mode: ${MODE}"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gfortran: $(gfortran --version | head -1)"
  echo "fflags: ${FFLAGS}"
  echo "pinning: taskset -c 4-31, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "wrf_source: ${WRF_PHYS}/module_mp_nssl_2mom.F (copied verbatim)"
  echo "namelist.input: empty (module defaults; no nssl_mp_params group)"
  echo "outputs: ${PREFIX}_case_{1..6}.json"
  echo "checksums:"
  cat "${OUT_SAVE}/wrf_source_checksums.txt"
} > "${OUT_SAVE}/build_manifest_${MODE}.txt"

echo "OK: NSSL 2-mom oracle (mode=${MODE}) built and 6 savepoints written to ${OUT_SAVE}"
