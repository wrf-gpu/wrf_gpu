#!/usr/bin/env bash
# Build the v0.6.0 KF-eta single-column oracle against the UNMODIFIED WRF
# module_cu_kfeta.F and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved tonight).
# Requires the conda env `wrfbuild` (gfortran 14.x).
set -uo pipefail   # NOT -e: the conda activate/deactivate hooks emit benign
                   # warnings that would otherwise abort the script.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_KF="/home/enric/src/wrf_pristine/WRF/phys/module_cu_kfeta.F"
OUT_SAVE="${HERE}/../savepoints"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2

# copy the pristine WRF source verbatim (provenance: wrf_pristine, unmodified)
cp "${WRF_KF}" "${HERE}/module_cu_kfeta.F"

cd "${HERE}"
# module_cu_kfeta.F is FREE-FORM despite the .F extension -> force -ffree-form
taskset -c 0-3 gfortran -O2 -ffree-line-length-none -ffpe-summary=none -c module_wrf_error.f90
taskset -c 0-3 gfortran -O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -c module_cu_kfeta.F
taskset -c 0-3 gfortran -O2 -ffree-line-length-none -ffpe-summary=none -c kf_oracle_driver.f90
taskset -c 0-3 gfortran -O2 -o kf_oracle module_wrf_error.o module_cu_kfeta.o kf_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./kf_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/kf_case_${c}.json"
done
echo "OK: oracle built and 5 savepoints written to ${OUT_SAVE}"
