#!/usr/bin/env bash
# Build the WDM5 (mp_physics=14) savepoint oracle against the UNMODIFIED
# pristine WRF source phys/module_mp_wdm5.F (+ its real dependency
# module_mp_radar.F, the real share/module_model_constants.F, and the
# VREC/VSQRT library frame/libmassv.F).
#
# Usage:
#   bash proofs/v013/oracle/build_wdm5_oracle.sh fp32
#   bash proofs/v013/oracle/build_wdm5_oracle.sh fp64
#
# fp32 = canonical WRF single precision (binding PROGNOSTIC reference).
# fp64 = -fdefault-real-8 -DDOUBLE_PRECISION override (scheme source otherwise
#        UNMODIFIED; the libmassv VREC/VSQRT macros pick the real*8 variants);
#        used for the categorical effective-radius diagnostics and as the
#        machine-precision cross-check that the oracle itself is consistent.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved). Requires
# the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: module_mp_wdm5.F, module_mp_radar.F, module_model_constants.F and
# libmassv.F are copied VERBATIM (unmodified) from /home/user/src/wrf_pristine.
# The only project-authored Fortran is wdm5_oracle_driver.f90 (column builder +
# dump; never touches scheme physics) and morrison_stub_modules.f90 (a no-op
# module_wrf_error + global wrf_debug/wrf_error_fatal, reused from the v0.6.0
# oracle). The mass + Nc/Nr/Nn tendencies come from the UNMODIFIED scheme.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/user/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
WRF_FRAME="${WRF}/frame"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -e
export OMP_NUM_THREADS=2

MODE="${1:-fp32}"
OUT_SAVE="${HERE}/../savepoints_wdm5"
FFLAGS_FREE="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none -I${WRF_PHYS}"
FFLAGS_FIXED="-O2 -cpp -ffixed-form -ffixed-line-length-none -ffpe-summary=none -I${WRF_PHYS}"
DRVDEF=""
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_wdm5_fp64"
  FFLAGS_FREE="${FFLAGS_FREE} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  FFLAGS_FIXED="${FFLAGS_FIXED} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  DRVDEF="-DWDM5_FP64"
fi

BLDDIR="${HERE}/build_wdm5_${MODE}"
rm -rf "${BLDDIR}"
mkdir -p "${BLDDIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BLDDIR}/module_model_constants.F"
cp "${WRF_FRAME}/libmassv.F"               "${BLDDIR}/libmassv.F"
cp "${WRF_PHYS}/module_mp_radar.F"         "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_wdm5.F"          "${BLDDIR}/module_mp_wdm5.F"
cp "${HERE}/../../v060/oracle/morrison_stub_modules.f90" "${BLDDIR}/morrison_stub_modules.f90"
cp "${HERE}/wdm5_oracle_driver.f90"        "${BLDDIR}/wdm5_oracle_driver.f90"

( cd "${BLDDIR}" && sha256sum \
    module_model_constants.F libmassv.F module_mp_radar.F module_mp_wdm5.F \
    morrison_stub_modules.f90 wdm5_oracle_driver.f90 \
    > "${OUT_SAVE}/wdm5_source_checksums.txt" )

cd "${BLDDIR}"
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c morrison_stub_modules.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS_FIXED} -c libmassv.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_wdm5.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} ${DRVDEF} -c wdm5_oracle_driver.f90

taskset -c 0-3 gfortran ${FFLAGS_FREE} -o wdm5_oracle \
  morrison_stub_modules.o module_model_constants.o libmassv.o \
  module_mp_radar.o module_mp_wdm5.o wdm5_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./wdm5_oracle "$c" > "wdm5_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/wdm5_dump_to_json.py" "wdm5_case_${c}.txt" \
      "${OUT_SAVE}/wdm5_case_${c}.json" "WDM5 (mp_physics=14)"
done

echo "OK: WDM5 oracle (mode=${MODE}) wrote savepoints to ${OUT_SAVE}"
