#!/usr/bin/env bash
# Build the v0.6.0 ACM2 single-column oracle against UNMODIFIED WRF
# module_bl_acm.F and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
OUT_SAVE="${HERE}/../savepoints"
BUILD_DIR="${HERE}/build"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_bl_acm.F" "${BUILD_DIR}/module_bl_acm.F"
cp "${HERE}/acm2_oracle_driver.f90" "${BUILD_DIR}/acm2_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_bl_acm.F > "${OUT_SAVE}/acm2_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_acm.F
taskset -c 0-3 gfortran ${FFLAGS} -c acm2_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o acm2_oracle module_bl_acm.o acm2_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./acm2_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/acm2_case_${c}.json"
done
echo "OK: ACM2 oracle built and 6 savepoints written to ${OUT_SAVE}"
