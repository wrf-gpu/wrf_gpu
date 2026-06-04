#!/usr/bin/env bash
# Build the v0.6.0 BMJ single-column oracle against UNMODIFIED WRF
# module_cu_bmj.F and emit JSON savepoints.
#
# CPU-only, pinned to cores 0-3. Requires conda env `wrfbuild`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_BMJ="/home/enric/src/wrf_pristine/WRF/phys/module_cu_bmj.F"
OUT_SAVE="${HERE}/../savepoints"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -euo pipefail
export OMP_NUM_THREADS=2

cp "${WRF_BMJ}" "${HERE}/module_cu_bmj.F"

cd "${HERE}"
taskset -c 0-3 gfortran -O2 -ffree-line-length-none -ffpe-summary=none -c bmj_module_model_constants.f90
taskset -c 0-3 gfortran -O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -c module_cu_bmj.F
taskset -c 0-3 gfortran -O2 -ffree-line-length-none -ffpe-summary=none -c bmj_oracle_driver.f90
taskset -c 0-3 gfortran -O2 -o bmj_oracle bmj_module_model_constants.o module_cu_bmj.o bmj_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./bmj_oracle "$c" > "bmj_case_${c}.txt"
  python3 "${HERE}/bmj_dump_to_json.py" "bmj_case_${c}.txt" "${OUT_SAVE}/bmj_case_${c}.json"
done
sha256sum "${WRF_BMJ}" > "${OUT_SAVE}/bmj_wrf_source_checksums.txt"
echo "OK: BMJ oracle built and 5 savepoints written to ${OUT_SAVE}"
