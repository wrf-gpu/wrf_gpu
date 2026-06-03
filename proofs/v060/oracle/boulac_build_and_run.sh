#!/usr/bin/env bash
# Build the v0.6.0 BouLac single-column oracle against UNMODIFIED WRF
# phys/module_bl_boulac.F and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
OUT_SAVE="${HERE}/../savepoints"
BUILD_DIR="${HERE}/build_boulac"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_bl_boulac.F" "${BUILD_DIR}/module_bl_boulac.F"
cp "${HERE}/boulac_oracle_driver.f90" "${BUILD_DIR}/boulac_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_bl_boulac.F > "${OUT_SAVE}/boulac_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -fdefault-real-8 -fdefault-double-8 -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_boulac.F
taskset -c 0-3 gfortran ${FFLAGS} -c boulac_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o boulac_oracle module_bl_boulac.o boulac_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./boulac_oracle "$c" > "boulac_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/boulac_dump_to_json.py" "boulac_case_${c}.txt" "${OUT_SAVE}/boulac_case_${c}.json"
done
echo "OK: BouLac oracle built and 6 savepoints written to ${OUT_SAVE}"
