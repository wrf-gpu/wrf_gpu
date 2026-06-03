#!/usr/bin/env bash
# Build the v0.6.0 Janjic (MYJ) surface-layer single-column oracle against the
# UNMODIFIED WRF module_sf_myjsfc.F + share/module_model_constants.F and emit
# gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
OUT_SAVE="${HERE}/../savepoints"
BUILD_DIR="${HERE}/build_myjsfc"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BUILD_DIR}/module_model_constants.F"
cp "${WRF_PHYS}/module_sf_myjsfc.F" "${BUILD_DIR}/module_sf_myjsfc.F"
cp "${HERE}/myjsfc_oracle_driver.f90" "${BUILD_DIR}/myjsfc_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_model_constants.F module_sf_myjsfc.F \
    > "${OUT_SAVE}/myjsfc_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
# No DM_PARALLEL / NMM_CORE macros: serial single-column ARW-core path.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_sf_myjsfc.F
taskset -c 0-3 gfortran ${FFLAGS} -c myjsfc_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o myjsfc_oracle \
    module_model_constants.o module_sf_myjsfc.o myjsfc_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./myjsfc_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/myjsfc_dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/myjsfc_case_${c}.json"
done
echo "OK: MYJ surface-layer oracle built and 6 savepoints written to ${OUT_SAVE}"
