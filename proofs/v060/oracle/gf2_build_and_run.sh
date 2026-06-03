#!/usr/bin/env bash
# Build the v0.6.0 Grell-Freitas single-column oracle against the UNMODIFIED
# pristine WRF GF sources and emit JSON savepoints.
#
# CPU-only, pinned to cores 0-3. Build products stay under oracle/build and are
# intentionally not proof objects; the JSON savepoints and parity report record
# source checksums.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
OUT_SAVE="${ROOT}/proofs/v060/savepoints"
BUILD_DIR="${HERE}/build"

set +u
if [ -f /home/enric/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/enric/miniconda3/etc/profile.d/conda.sh
  conda activate wrfbuild || true
fi
set -u

export OMP_NUM_THREADS=2
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"
rm -f "${BUILD_DIR}"/*

cp "${WRF_PHYS}/module_gfs_machine.F" "${BUILD_DIR}/module_gfs_machine.F"
cp "${WRF_PHYS}/module_gfs_physcons.F" "${BUILD_DIR}/module_gfs_physcons.F"
cp "${WRF_PHYS}/module_cu_gf_deep.F" "${BUILD_DIR}/module_cu_gf_deep.F"
cp "${WRF_PHYS}/module_cu_gf_sh.F" "${BUILD_DIR}/module_cu_gf_sh.F"
cp "${WRF_PHYS}/module_cu_gf_wrfdrv.F" "${BUILD_DIR}/module_cu_gf_wrfdrv.F"
cp "${HERE}/gf_oracle_driver.f90" "${BUILD_DIR}/gf_oracle_driver.f90"

cd "${BUILD_DIR}"
FC=${FC:-gfortran}
FFLAGS=(-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -DWRF_CHEM=0 -DWRF_DFI_RADAR=0 -DNMM_CORE=0)

taskset -c 0-3 "${FC}" "${FFLAGS[@]}" -c module_gfs_machine.F
taskset -c 0-3 "${FC}" "${FFLAGS[@]}" -c module_gfs_physcons.F
taskset -c 0-3 "${FC}" "${FFLAGS[@]}" -c module_cu_gf_deep.F
taskset -c 0-3 "${FC}" "${FFLAGS[@]}" -c module_cu_gf_sh.F
taskset -c 0-3 "${FC}" "${FFLAGS[@]}" -c module_cu_gf_wrfdrv.F
taskset -c 0-3 "${FC}" -O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -c gf_oracle_driver.f90
taskset -c 0-3 "${FC}" -O2 -o gf_oracle \
  module_gfs_machine.o module_gfs_physcons.o module_cu_gf_deep.o \
  module_cu_gf_sh.o module_cu_gf_wrfdrv.o gf_oracle_driver.o

for c in 1 2 3 4 5; do
  taskset -c 0-3 ./gf_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/gf_case_${c}.json"
done

echo "OK: Grell-Freitas oracle built and 5 savepoints written to ${OUT_SAVE}"
