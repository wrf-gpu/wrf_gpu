#!/usr/bin/env bash
# Build the v0.18 QNSE-EDMF PBL reference oracle (bl_pbl_physics=4)
# against the UNMODIFIED pristine WRF module_bl_qnsepbl.F and emit fp64
# single-column savepoints as JSON.
#
# REFERENCE-ONLY endpoint: this proves the pristine-WRF source boundary and
# stages real oracle data for a later GPU/JAX port. It is not an operational
# scan implementation and not a self-compare.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/user/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
SAVE="${HERE}/../../savepoints_fp64/qnse"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

BUILD="${HERE}/build_fp64"
rm -rf "${BUILD}"
mkdir -p "${BUILD}" "${SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BUILD}/module_model_constants.F"
cp "${WRF_PHYS}/module_bl_qnsepbl.F" "${BUILD}/module_bl_qnsepbl.F"
cp "${HERE}/qnse_oracle_driver.f90" "${BUILD}/qnse_oracle_driver.f90"

( cd "${BUILD}" && sha256sum module_model_constants.F module_bl_qnsepbl.F > "${SAVE}/qnse_wrf_source_checksums.txt" )

cd "${BUILD}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0 -ffpe-summary=none -fdefault-real-8 -fdefault-double-8"
taskset -c 0-3 gfortran ${FFLAGS} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_qnsepbl.F
taskset -c 0-3 gfortran ${FFLAGS} -c qnse_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o qnse_oracle \
    module_model_constants.o module_bl_qnsepbl.o qnse_oracle_driver.o

{
  echo "mode=fp64"
  echo "full_wrf_exe=false"
  echo "wrf_sources=${WRF_SHARE}/module_model_constants.F,${WRF_PHYS}/module_bl_qnsepbl.F"
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags=${FFLAGS}"
} > "${SAVE}/qnse_build_manifest.txt"

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./qnse_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/../dump_to_json.py" "case_${c}.txt" "${SAVE}/qnse_case_${c}.json"
done

echo "OK: QNSE oracle built; fp64 -> ${SAVE}"
