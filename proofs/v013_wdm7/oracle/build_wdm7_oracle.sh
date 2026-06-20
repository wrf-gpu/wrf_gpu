#!/usr/bin/env bash
# Build the WDM7 (mp_physics=26) single-column oracle against the UNMODIFIED WRF
# phys/module_mp_wdm7.F (+ the real phys/module_mp_radar.F and frame/libmassv.F
# for vsrec/vssqrt) and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved).
# Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Usage:
#   bash proofs/v013_wdm7/oracle/build_wdm7_oracle.sh fp32
#   bash proofs/v013_wdm7/oracle/build_wdm7_oracle.sh fp64
#
# fp32 = canonical WRF single precision (binding PROGNOSTIC reference).
# fp64 = -fdefault-real-8 override (scheme source otherwise UNMODIFIED); used
#        for the categorical effective-radius diagnostics. CRUCIAL: also pass
#        -DDOUBLE_PRECISION so module_mp_wdm7.F's VREC/VSQRT macros select the
#        real*8 libmassv variants (vrec/vsqrt) instead of the real*4 ones
#        (vsrec/vssqrt) -- otherwise promoted real*8 denfac is passed to real*4
#        subroutine args and corrupts the fall-speed/precip (RAIN denormal).
#
# Provenance: module_mp_wdm7.F, phys/module_mp_radar.F, and frame/libmassv.F are
# copied VERBATIM (unmodified) from <USER_HOME>/src/wrf_pristine/WRF. The only
# project-authored Fortran is:
#   * wdm7_oracle_driver.f90          -- the column builder + dump (never touches
#                                        scheme physics)
#   * stub_module_wrf_error.F90       -- no-op wrf_debug/message (module_mp_radar
#                                        uses only wrf_debug; serial oracle)
#   * stub_module_model_constants.F90 -- the 3 RE_*_BG background radii, exact
#                                        pristine values.
# The microphysics mass + Nc/Nr/Nn tendencies come from the UNMODIFIED scheme
# (module_mp_wdm7.F), and the radar reflectivity helper compiles against the
# UNMODIFIED phys/module_mp_radar.F. The reflectivity DIAGNOSTIC itself is not
# enabled in the savepoints (diagflag is never passed; wdm72D has no reflectivity
# path) and is out of scope.
set -o pipefail   # NOT -e/-u: conda activate/deactivate hooks reference unset
                  # vars and emit benign warnings that would otherwise abort.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_FRAME="${WRF}/frame"
OUT_SAVE="${HERE}/../savepoints_wdm7"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
BLDDIR="${HERE}/build"

MODE="${1:-fp32}"
FP64FLAG=""
DRVDEF=""
WDM7DEF=""
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_wdm7_fp64"
  FP64FLAG="-fdefault-real-8 -fdefault-double-8"
  DRVDEF="-DWDM7_FP64"
  WDM7DEF="-DDOUBLE_PRECISION"
  BLDDIR="${HERE}/build_fp64"
fi

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}"
# copy pristine WRF sources verbatim
cp "${WRF_PHYS}/module_mp_wdm7.F"            "${BLDDIR}/module_mp_wdm7.F"
cp "${WRF_PHYS}/module_mp_radar.F"           "${BLDDIR}/module_mp_radar.F"
cp "${WRF_FRAME}/libmassv.F"                 "${BLDDIR}/libmassv.F"
cp "${HERE}/stub_module_wrf_error.F90"       "${BLDDIR}/stub_module_wrf_error.F90"
cp "${HERE}/stub_module_model_constants.F90" "${BLDDIR}/stub_module_model_constants.F90"
cp "${HERE}/wdm7_oracle_driver.f90"          "${BLDDIR}/wdm7_oracle_driver.F90"

# record exact source checksums (provenance). The UNMODIFIED scheme + radar are
# the WRF files; the stubs/driver are project-authored.
mkdir -p "${OUT_SAVE}"
( cd "${BLDDIR}" && sha256sum module_mp_wdm7.F module_mp_radar.F libmassv.F \
     stub_module_wrf_error.F90 stub_module_model_constants.F90 \
     wdm7_oracle_driver.F90 > "${OUT_SAVE}/wdm7_source_checksums.txt" )

cd "${BLDDIR}"
# -O2 matches WRF default opt. module_mp_wdm7.F and module_mp_radar.F are
# FREE-form (despite .F) and module_mp_wdm7.F uses cpp (#ifndef) -> -cpp.
FFLAGS="-O2 -ffpe-summary=none ${FP64FLAG}"

taskset -c 0-3 gfortran ${FFLAGS} -ffixed-form -c libmassv.F -o libmassv.o
taskset -c 0-3 gfortran ${FFLAGS} -c stub_module_wrf_error.F90
taskset -c 0-3 gfortran ${FFLAGS} -c stub_module_model_constants.F90
taskset -c 0-3 gfortran ${FFLAGS} -cpp -ffree-form -ffree-line-length-none \
    -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS} ${WDM7DEF} -cpp -ffree-form -ffree-line-length-none \
    -c module_mp_wdm7.F
taskset -c 0-3 gfortran ${FFLAGS} ${DRVDEF} -cpp -ffree-form -ffree-line-length-none \
    -c wdm7_oracle_driver.F90
taskset -c 0-3 gfortran ${FFLAGS} -o wdm7_oracle \
    libmassv.o stub_module_wrf_error.o stub_module_model_constants.o \
    module_mp_radar.o module_mp_wdm7.o wdm7_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./wdm7_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/wdm7_dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/wdm7_case_${c}.json"
done
echo "OK: WDM7 oracle (mode=${MODE}) built and 6 savepoints written to ${OUT_SAVE}"
