#!/usr/bin/env bash
# Build the v0.6.0 modified-Tiedtke single-column oracle against the
# UNMODIFIED WRF module_cu_tiedtke.F plus its WRF module_model_constants.F
# dependency, then emit regime savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_TIEDTKE="/home/enric/src/wrf_pristine/WRF/phys/module_cu_tiedtke.F"
WRF_CONSTANTS="/home/enric/src/wrf_pristine/WRF/share/module_model_constants.F"
OUT_SAVE="${HERE}/../savepoints"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2

cp "${WRF_TIEDTKE}" "${HERE}/module_cu_tiedtke.F"
cp "${WRF_CONSTANTS}" "${HERE}/module_model_constants.F"

cd "${HERE}"
taskset -c 0-3 gfortran -O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -c module_model_constants.F
taskset -c 0-3 gfortran -O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -c module_cu_tiedtke.F
taskset -c 0-3 gfortran -O2 -ffree-line-length-none -ffpe-summary=none -c tiedtke_oracle_driver.f90
taskset -c 0-3 gfortran -O2 -o tiedtke_oracle module_model_constants.o module_cu_tiedtke.o tiedtke_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./tiedtke_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/tiedtke_case_${c}.json"
done

echo "OK: oracle built and 5 savepoints written to ${OUT_SAVE}"
