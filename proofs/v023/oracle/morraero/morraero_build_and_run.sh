#!/usr/bin/env bash
# Build the v0.23 F2 AEROSOL-AWARE Morrison 2-moment (WRF mp_physics=40)
# single-column oracle against the UNMODIFIED WRF source
# (phys/module_mp_morr_two_moment_aero.F) + its real dependency
# phys/module_mp_radar.F + the real share/module_model_constants.F, and emit
# gold savepoints as JSON.
#
# CPU-only, pinned to cores 4-31 (cores 0-3 + the GPU are reserved for
# another agent's lane). Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: the aero-Morrison scheme source, the radar dependency, and the
# model constants are copied VERBATIM (unmodified) from
# <USER_HOME>/src/wrf_pristine/WRF. The only project-authored Fortran is:
#   - morraero_oracle_driver.f90  (column builder + dump; never touches physics)
#   - morraero_stub_modules.f90   (empty module_wrf_error + no-op global
#                                  wrf_debug; identical to the v0.6.0 base
#                                  Morrison oracle stub)
#
# Modeled on proofs/v060/oracle/morrison_build_and_run.sh.
set -o pipefail   # NOT -e/-u: conda activate hooks reference unset vars.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
# fp32 and fp64 savepoints share one directory; the JSON prefix differs.
OUT_SAVE="${HERE}/../../../v022/f2_oracles/morrison_aero"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=4
BLDDIR="${HERE}/build"

# MODE: "fp32" (default, canonical WRF default real kind = single precision) or
# "fp64" (-fdefault-real-8 promotes all default REALs to double; scheme source
# otherwise unmodified) to cross-check that any trace-cell diagnostic floor
# flips are fp32 detection-threshold dust in the reference, not port error.
MODE="${1:-fp32}"
PROMOTE=""
PREFIX="morr_aero_case"
if [ "${MODE}" = "fp64" ]; then
  PREFIX="morr_aero_fp64_case"
  PROMOTE="-fdefault-real-8 -fdefault-double-8"
fi

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}" "${OUT_SAVE}"

# copy pristine WRF sources verbatim
cp "${WRF_SHARE}/module_model_constants.F"           "${BLDDIR}/module_model_constants.F"
cp "${WRF_PHYS}/module_mp_radar.F"                   "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_morr_two_moment_aero.F"    "${BLDDIR}/module_mp_morr_two_moment_aero.F"
cp "${HERE}/morraero_stub_modules.f90"               "${BLDDIR}/morraero_stub_modules.f90"
cp "${HERE}/morraero_oracle_driver.f90"              "${BLDDIR}/morraero_oracle_driver.f90"

# record exact source checksums (provenance) of the UNMODIFIED scheme sources
( cd "${BLDDIR}" && sha256sum module_model_constants.F module_mp_radar.F \
    module_mp_morr_two_moment_aero.F morraero_stub_modules.f90 morraero_oracle_driver.f90 \
    > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
# -ffree-line-length-none for the long .F lines; -cpp for the #if (WRF_CHEM==1)
# guards (WRF_CHEM undefined -> chem branches excluded, matching the WRF
# default non-chem build, exactly as for the base Morrison oracle).
FFLAGS="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp ${PROMOTE}"
taskset -c 4-31 gfortran ${FFLAGS} -c morraero_stub_modules.f90            || exit 11
taskset -c 4-31 gfortran ${FFLAGS} -c module_model_constants.F             || exit 12
taskset -c 4-31 gfortran ${FFLAGS} -c module_mp_radar.F                    || exit 13
taskset -c 4-31 gfortran ${FFLAGS} -c module_mp_morr_two_moment_aero.F     || exit 14
taskset -c 4-31 gfortran ${FFLAGS} -c morraero_oracle_driver.f90           || exit 15
taskset -c 4-31 gfortran ${FFLAGS} -o morraero_oracle \
    morraero_stub_modules.o module_model_constants.o module_mp_radar.o \
    module_mp_morr_two_moment_aero.o morraero_oracle_driver.o              || exit 16

for c in 1 2 3 4 5 6; do
  taskset -c 4-31 ./morraero_oracle "$c" > "case_${c}.txt" || { echo "RUN FAIL case $c"; exit 20; }
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/${PREFIX}_${c}.json" || exit 21
done

# build manifest: fp32 run truncates, fp64 run appends its stanza
MANIFEST="${OUT_SAVE}/build_manifest.txt"
if [ "${MODE}" = "fp32" ]; then : > "${MANIFEST}"; fi
{
  echo "== morraero oracle build: mode=${MODE} $(date -Is) host=$(hostname) =="
  echo "compiler : $(gfortran --version | head -1)"
  echo "FFLAGS   : ${FFLAGS}"
  echo "WRF tree : ${WRF} (sources copied verbatim; sha256 in wrf_source_checksums.txt)"
  echo "driver   : proofs/v023/oracle/morraero/morraero_oracle_driver.f90"
  echo "entry    : MORR_TWO_MOMENT_INIT_AERO(0); MP_MORR_TWO_MOMENT_AERO (aercu_opt=2)"
  echo "cases    : 1-6 -> ${PREFIX}_{1..6}.json (schema wrf-v023-f2-morraero-column-savepoint-v1)"
  echo ""
} >> "${MANIFEST}"

echo "OK: Morrison-aero oracle (mode=${MODE}) built and 6 savepoints written to ${OUT_SAVE}"
