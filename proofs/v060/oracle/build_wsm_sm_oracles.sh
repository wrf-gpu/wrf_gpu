#!/usr/bin/env bash
# Build WSM5/WSM3 savepoint oracles against UNMODIFIED pristine WRF classic
# sources (phys/module_mp_wsm5.F and phys/module_mp_wsm3.F).
#
# Usage:
#   bash proofs/v060/oracle/build_wsm_sm_oracles.sh fp32
#   bash proofs/v060/oracle/build_wsm_sm_oracles.sh fp64
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
WRF_FRAME="${WRF}/frame"
OUT_SAVE="${HERE}/../savepoints"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -e
export OMP_NUM_THREADS=2

MODE="${1:-fp32}"
FFLAGS_FREE="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none -I${WRF_PHYS}"
FFLAGS_FIXED="-O2 -cpp -ffixed-form -ffixed-line-length-none -ffpe-summary=none -I${WRF_PHYS}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_fp64"
  FFLAGS_FREE="${FFLAGS_FREE} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  FFLAGS_FIXED="${FFLAGS_FIXED} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
fi

BLDDIR="${HERE}/build_wsm_sm_${MODE}"
rm -rf "${BLDDIR}"
mkdir -p "${BLDDIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BLDDIR}/module_model_constants.F"
cp "${WRF_FRAME}/libmassv.F" "${BLDDIR}/libmassv.F"
cp "${WRF_PHYS}/module_mp_radar.F" "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_wsm5.F" "${BLDDIR}/module_mp_wsm5.F"
cp "${WRF_PHYS}/module_mp_wsm3.F" "${BLDDIR}/module_mp_wsm3.F"
cp "${WRF_PHYS}/mic-wsm5-3-5-code.h" "${BLDDIR}/mic-wsm5-3-5-code.h"
cp "${WRF_PHYS}/mic-wsm5-3-5-locvar.h" "${BLDDIR}/mic-wsm5-3-5-locvar.h"
cp "${WRF_PHYS}/mic-wsm5-3-5-callsite.h" "${BLDDIR}/mic-wsm5-3-5-callsite.h"
cp "${HERE}/morrison_stub_modules.f90" "${BLDDIR}/morrison_stub_modules.f90"
cp "${HERE}/wsm5_oracle_driver.f90" "${BLDDIR}/wsm5_oracle_driver.f90"
cp "${HERE}/wsm3_oracle_driver.f90" "${BLDDIR}/wsm3_oracle_driver.f90"

( cd "${BLDDIR}" && sha256sum \
    module_model_constants.F libmassv.F module_mp_radar.F module_mp_wsm5.F module_mp_wsm3.F \
    mic-wsm5-3-5-code.h mic-wsm5-3-5-locvar.h mic-wsm5-3-5-callsite.h \
    morrison_stub_modules.f90 wsm5_oracle_driver.f90 wsm3_oracle_driver.f90 \
    > "${OUT_SAVE}/wsm_sm_source_checksums.txt" )

cd "${BLDDIR}"
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c morrison_stub_modules.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS_FIXED} -c libmassv.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_wsm5.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_wsm3.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c wsm5_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c wsm3_oracle_driver.f90

taskset -c 0-3 gfortran ${FFLAGS_FREE} -o wsm5_oracle \
  morrison_stub_modules.o module_model_constants.o libmassv.o module_mp_radar.o \
  module_mp_wsm5.o wsm5_oracle_driver.o
taskset -c 0-3 gfortran ${FFLAGS_FREE} -o wsm3_oracle \
  module_model_constants.o libmassv.o module_mp_wsm3.o wsm3_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./wsm5_oracle "$c" > "wsm5_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/wsm_sm_dump_to_json.py" "wsm5_case_${c}.txt" "${OUT_SAVE}/wsm5_case_${c}.json" "WSM5 (mp_physics=4)"
  taskset -c 0-3 ./wsm3_oracle "$c" > "wsm3_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/wsm_sm_dump_to_json.py" "wsm3_case_${c}.txt" "${OUT_SAVE}/wsm3_case_${c}.json" "WSM3 (mp_physics=3)"
done

echo "OK: WSM3/WSM5 oracles (mode=${MODE}) wrote savepoints to ${OUT_SAVE}"
