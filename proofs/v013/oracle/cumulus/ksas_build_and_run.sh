#!/usr/bin/env bash
# Build the v0.13 KSAS (cu_physics=14) single-column oracle against the
# UNMODIFIED WRF module_cu_ksas.F (self-contained, no module deps).
#
# Compiled with -fdefault-real-8 -fdefault-double-8 so WRF's default REAL
# becomes double -> a true fp64 reference for the (fp64) JAX kernel.
# CPU-only, cores 0-3. Requires the conda env `wrfbuild`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_KSAS="/home/enric/src/wrf_pristine/WRF/phys/module_cu_ksas.F"
OUT_SAVE="${HERE}/../../savepoints/cumulus"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2

cp "${WRF_KSAS}" "${HERE}/module_cu_ksas.F"

cd "${HERE}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -fdefault-real-8 -fdefault-double-8 -ffpe-summary=none -fallow-argument-mismatch -std=legacy"
taskset -c 0-3 gfortran ${FFLAGS} -c module_cu_ksas.F
taskset -c 0-3 gfortran ${FFLAGS} -c ksas_oracle_driver.f90
taskset -c 0-3 gfortran -O2 -o ksas_oracle module_cu_ksas.o ksas_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./ksas_oracle "$c" > "ksas_case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "ksas_case_${c}.txt" "${OUT_SAVE}/ksas_case_${c}.json"
done

echo "OK: ksas oracle built and 5 savepoints written to ${OUT_SAVE}"
