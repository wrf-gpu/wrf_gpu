#!/usr/bin/env bash
# Build v0.18 Grell-family standalone pristine-WRF oracles and emit JSON savepoints.
# The harness calls pristine WRF Fortran sources, not GPU/JAX candidates.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
WRF_ROOT="${WRF_PRISTINE_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}"
WRF_PHYS="${WRF_ROOT}/phys"
OUT_SAVE="${ROOT}/proofs/v018/savepoints/cumulus_grell"

set +u
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2

mkdir -p "${OUT_SAVE}"
cd "${HERE}"

cp "${WRF_PHYS}/module_cu_g3.F" .
cp "${WRF_PHYS}/module_cu_gd.F" .

FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -fdefault-real-8 -fdefault-double-8 -ffpe-summary=none -fallow-argument-mismatch -std=legacy"

taskset -c 0-3 gfortran ${FFLAGS} -c module_cu_g3.F
taskset -c 0-3 gfortran ${FFLAGS} -DSCHEME_G3 -c grell_oracle_driver.F90 -o g3_oracle_driver.o
taskset -c 0-3 gfortran -O2 -o g3_oracle module_cu_g3.o g3_oracle_driver.o
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./g3_oracle "$c" > "g3_case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "g3_case_${c}.txt" "${OUT_SAVE}/g3_case_${c}.json"
done

taskset -c 0-3 gfortran ${FFLAGS} -c module_cu_gd.F
taskset -c 0-3 gfortran ${FFLAGS} -DSCHEME_GD -c grell_oracle_driver.F90 -o gd_oracle_driver.o
taskset -c 0-3 gfortran -O2 -o gd_oracle module_cu_gd.o gd_oracle_driver.o
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./gd_oracle "$c" > "gd_case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "gd_case_${c}.txt" "${OUT_SAVE}/gd_case_${c}.json"
done

echo "OK: Grell-family oracle savepoints written to ${OUT_SAVE}"
