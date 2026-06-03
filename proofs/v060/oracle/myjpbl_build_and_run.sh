#!/usr/bin/env bash
# Build the v0.6.0 MYJ PBL single-column oracle against the UNMODIFIED WRF
# module_bl_myjpbl.F + module_sf_myjsfc.F + share/module_model_constants.F and
# emit gold savepoints as JSON. The driver runs the PAIRED Janjic surface layer
# (sf_sfclay=2) before MYJ PBL (bl_pbl=2), as WRF requires.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
OUT_SAVE="${HERE}/../savepoints"
BUILD_DIR="${HERE}/build_myjpbl"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BUILD_DIR}/module_model_constants.F"
cp "${WRF_PHYS}/module_sf_myjsfc.F" "${BUILD_DIR}/module_sf_myjsfc.F"
cp "${WRF_PHYS}/module_bl_myjpbl.F" "${BUILD_DIR}/module_bl_myjpbl.F"
cp "${HERE}/myjpbl_oracle_driver.f90" "${BUILD_DIR}/myjpbl_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_model_constants.F module_sf_myjsfc.F module_bl_myjpbl.F \
    > "${OUT_SAVE}/myjpbl_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_sf_myjsfc.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_myjpbl.F
taskset -c 0-3 gfortran ${FFLAGS} -c myjpbl_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o myjpbl_oracle \
    module_model_constants.o module_sf_myjsfc.o module_bl_myjpbl.o myjpbl_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./myjpbl_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/myjpbl_dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/myjpbl_case_${c}.json"
done
echo "OK: MYJ PBL oracle built and 6 savepoints written to ${OUT_SAVE}"
