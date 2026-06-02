#!/usr/bin/env bash
# Build the v0.6.0 Morrison 2-moment single-column oracle against the UNMODIFIED
# WRF Morrison source (phys/module_mp_morr_two_moment.F) + its real dependency
# phys/module_mp_radar.F + the real share/module_model_constants.F, and emit gold
# savepoints as JSON.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved).
# Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: the Morrison scheme source, the radar dependency, and the model
# constants are copied VERBATIM (unmodified) from /home/enric/src/wrf_pristine/WRF.
# The only project-authored Fortran is:
#   - morrison_oracle_driver.f90  (column builder + dump; never touches physics)
#   - morrison_stub_modules.f90   (empty module_wrf_error + no-op global wrf_debug
#                                  so the scheme links without ESMF/full WRF; the
#                                  scheme references NO module-scope name from
#                                  module_wrf_error, only the global wrf_debug).
set -o pipefail   # NOT -e/-u: conda activate hooks reference unset vars.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
OUT_SAVE="${HERE}/../savepoints"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
BLDDIR="${HERE}/build"

# MODE: "fp32" (default, canonical WRF default real kind = single precision) or
# "fp64" (-fdefault-real-8 promotes all default REALs to double; scheme source
# otherwise unmodified) to cross-check that any trace-cell diagnostic floor
# flips are fp32 detection-threshold dust in the reference, not port error.
MODE="${1:-fp32}"
PROMOTE=""
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_fp64"
  PROMOTE="-fdefault-real-8 -fdefault-double-8"
fi

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}" "${OUT_SAVE}"

# copy pristine WRF sources verbatim
cp "${WRF_SHARE}/module_model_constants.F"        "${BLDDIR}/module_model_constants.F"
cp "${WRF_PHYS}/module_mp_radar.F"                "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_morr_two_moment.F"      "${BLDDIR}/module_mp_morr_two_moment.F"
cp "${HERE}/morrison_stub_modules.f90"            "${BLDDIR}/morrison_stub_modules.f90"
cp "${HERE}/morrison_oracle_driver.f90"           "${BLDDIR}/morrison_oracle_driver.f90"

# record exact source checksums (provenance) of the UNMODIFIED scheme sources
( cd "${BLDDIR}" && sha256sum module_model_constants.F module_mp_radar.F \
    module_mp_morr_two_moment.F morrison_stub_modules.f90 morrison_oracle_driver.f90 \
    > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
# -ffree-line-length-none for the long .F lines; -cpp for the #if (WRF_CHEM==1)
# guards (WRF_CHEM undefined -> chem branches excluded, matching the WRF default).
FFLAGS="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp ${PROMOTE}"
taskset -c 0-3 gfortran ${FFLAGS} -c morrison_stub_modules.f90    || exit 11
taskset -c 0-3 gfortran ${FFLAGS} -c module_model_constants.F     || exit 12
taskset -c 0-3 gfortran ${FFLAGS} -c module_mp_radar.F            || exit 13
taskset -c 0-3 gfortran ${FFLAGS} -c module_mp_morr_two_moment.F  || exit 14
taskset -c 0-3 gfortran ${FFLAGS} -c morrison_oracle_driver.f90   || exit 15
taskset -c 0-3 gfortran ${FFLAGS} -o morrison_oracle \
    morrison_stub_modules.o module_model_constants.o module_mp_radar.o \
    module_mp_morr_two_moment.o morrison_oracle_driver.o            || exit 16

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./morrison_oracle "$c" > "case_${c}.txt" || { echo "RUN FAIL case $c"; exit 20; }
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/morrison_case_${c}.json" || exit 21
done
echo "OK: Morrison oracle (mode=${MODE}) built and 6 savepoints written to ${OUT_SAVE}"
