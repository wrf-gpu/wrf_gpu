#!/usr/bin/env bash
# Build the v0.18 EEPS PBL reference oracle (bl_pbl_physics=16)
# against the UNMODIFIED pristine WRF module_bl_eepsilon.F and emit fp64
# single-column savepoints as JSON.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/user/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
SAVE="${HERE}/../../savepoints_fp64/eeps"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

BUILD="${HERE}/build_fp64"
rm -rf "${BUILD}"
mkdir -p "${BUILD}" "${SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BUILD}/module_model_constants.F"
cp "${WRF_PHYS}/module_bl_eepsilon.F" "${BUILD}/module_bl_eepsilon.F"
cp "${HERE}/eeps_oracle_driver.f90" "${BUILD}/eeps_oracle_driver.f90"

( cd "${BUILD}" && sha256sum module_model_constants.F module_bl_eepsilon.F > "${SAVE}/eeps_wrf_source_checksums.txt" )

cd "${BUILD}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0 -ffpe-summary=none -fdefault-real-8 -fdefault-double-8"
taskset -c 0-3 gfortran ${FFLAGS} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_eepsilon.F
taskset -c 0-3 gfortran ${FFLAGS} -c eeps_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o eeps_oracle module_model_constants.o module_bl_eepsilon.o eeps_oracle_driver.o

{
  echo "mode=fp64"
  echo "full_wrf_exe=false"
  echo "wrf_sources=${WRF_SHARE}/module_model_constants.F,${WRF_PHYS}/module_bl_eepsilon.F"
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags=${FFLAGS}"
} > "${SAVE}/eeps_build_manifest.txt"

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./eeps_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/../dump_to_json.py" "case_${c}.txt" "${SAVE}/eeps_case_${c}.json"
done

echo "OK: EEPS oracle built; fp64 -> ${SAVE}"
