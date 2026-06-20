#!/usr/bin/env bash
# Build the WSM7 (mp_physics=24) savepoint oracle against the UNMODIFIED
# pristine WRF source phys/module_mp_wsm7.F (+ its real dependency
# module_mp_radar.F and the VREC/VSQRT library frame/libmassv.F).
#
# Usage:
#   bash proofs/v013/oracle/build_wsm7_oracle.sh fp32
#   bash proofs/v013/oracle/build_wsm7_oracle.sh fp64
#
# fp32 = canonical WRF single precision (binding PROGNOSTIC reference).
# fp64 = -fdefault-real-8 override (scheme source otherwise UNMODIFIED); used
#        for the categorical effective-radius diagnostics only.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
WRF_FRAME="${WRF}/frame"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -e
export OMP_NUM_THREADS=2

MODE="${1:-fp32}"
OUT_SAVE="${HERE}/../savepoints_wsm7"
FFLAGS_FREE="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none -I${WRF_PHYS}"
FFLAGS_FIXED="-O2 -cpp -ffixed-form -ffixed-line-length-none -ffpe-summary=none -I${WRF_PHYS}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_wsm7_fp64"
  FFLAGS_FREE="${FFLAGS_FREE} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  FFLAGS_FIXED="${FFLAGS_FIXED} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
fi

BLDDIR="${HERE}/build_wsm7_${MODE}"
rm -rf "${BLDDIR}"
mkdir -p "${BLDDIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BLDDIR}/module_model_constants.F"
cp "${WRF_FRAME}/libmassv.F"               "${BLDDIR}/libmassv.F"
cp "${WRF_PHYS}/module_mp_radar.F"         "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_wsm7.F"          "${BLDDIR}/module_mp_wsm7.F"
cp "${HERE}/../../v060/oracle/morrison_stub_modules.f90" "${BLDDIR}/morrison_stub_modules.f90"
cp "${HERE}/wsm7_oracle_driver.f90"        "${BLDDIR}/wsm7_oracle_driver.f90"

( cd "${BLDDIR}" && sha256sum \
    module_model_constants.F libmassv.F module_mp_radar.F module_mp_wsm7.F \
    morrison_stub_modules.f90 wsm7_oracle_driver.f90 \
    > "${OUT_SAVE}/wsm7_source_checksums.txt" )

cd "${BLDDIR}"
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c morrison_stub_modules.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS_FIXED} -c libmassv.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_wsm7.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c wsm7_oracle_driver.f90

taskset -c 0-3 gfortran ${FFLAGS_FREE} -o wsm7_oracle \
  morrison_stub_modules.o module_model_constants.o libmassv.o \
  module_mp_radar.o module_mp_wsm7.o wsm7_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./wsm7_oracle "$c" > "wsm7_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/wsm7_dump_to_json.py" "wsm7_case_${c}.txt" \
      "${OUT_SAVE}/wsm7_case_${c}.json" "WSM7 (mp_physics=24)"
done

echo "OK: WSM7 oracle (mode=${MODE}) wrote savepoints to ${OUT_SAVE}"
