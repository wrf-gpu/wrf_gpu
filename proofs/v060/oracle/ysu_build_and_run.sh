#!/usr/bin/env bash
# Build the v0.6.0 YSU single-column oracle against UNMODIFIED WRF
# module_bl_ysu.F + physics_mmm/bl_ysu.F90 and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
MMM="${WRF_PHYS}/physics_mmm"
OUT_SAVE="${HERE}/../savepoints"
BUILD_DIR="${HERE}/build"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/ccpp_kind_types.f90" "${BUILD_DIR}/ccpp_kind_types.f90"
cp "${MMM}/bl_ysu.F90" "${BUILD_DIR}/bl_ysu.F90"
cp "${WRF_PHYS}/module_bl_ysu.F" "${BUILD_DIR}/module_bl_ysu.F"
cp "${HERE}/ysu_oracle_driver.f90" "${BUILD_DIR}/ysu_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum ccpp_kind_types.f90 bl_ysu.F90 module_bl_ysu.F \
    > "${OUT_SAVE}/ysu_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c ccpp_kind_types.f90
taskset -c 0-3 gfortran ${FFLAGS} -c bl_ysu.F90
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_ysu.F
taskset -c 0-3 gfortran ${FFLAGS} -c ysu_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o ysu_oracle \
    ccpp_kind_types.o bl_ysu.o module_bl_ysu.o ysu_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./ysu_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/ysu_case_${c}.json"
done
echo "OK: YSU oracle built and 6 savepoints written to ${OUT_SAVE}"
