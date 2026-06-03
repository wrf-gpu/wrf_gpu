#!/usr/bin/env bash
# Build the v0.6.0 revised-MM5 surface-layer oracle against UNMODIFIED WRF
# module_sf_sfclayrev.F + physics_mmm/sf_sfclayrev.F90 and emit JSON savepoints.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
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
cp "${MMM}/sf_sfclayrev.F90" "${BUILD_DIR}/sf_sfclayrev.F90"
cp "${WRF_PHYS}/module_sf_sfclayrev.F" "${BUILD_DIR}/module_sf_sfclayrev.F"
cp "${HERE}/sfclayrev_oracle_driver.f90" "${BUILD_DIR}/sfclayrev_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum ccpp_kind_types.f90 sf_sfclayrev.F90 module_sf_sfclayrev.F \
    > "${OUT_SAVE}/sfclayrev1_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c ccpp_kind_types.f90
taskset -c 0-3 gfortran ${FFLAGS} -c sf_sfclayrev.F90
taskset -c 0-3 gfortran ${FFLAGS} -c module_sf_sfclayrev.F
taskset -c 0-3 gfortran ${FFLAGS} -c sfclayrev_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o sfclayrev_oracle \
    ccpp_kind_types.o sf_sfclayrev.o module_sf_sfclayrev.o sfclayrev_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./sfclayrev_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/sfclayrev1_case_${c}.json"
done
echo "OK: revised-MM5 surface-layer oracle built and 6 savepoints written to ${OUT_SAVE}"
