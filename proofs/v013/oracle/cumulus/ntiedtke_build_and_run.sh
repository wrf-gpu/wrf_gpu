#!/usr/bin/env bash
# Build the v0.13 New-Tiedtke (cu_physics=16) single-column oracle against the
# UNMODIFIED WRF module_cu_ntiedtke.F + its CCPP core physics_mmm/cu_ntiedtke.F90
# + ccpp_kind_types.F, then emit regime savepoints as JSON.
#
# Compiled with -DDOUBLE_PRECISION so kind_phys = selected_real_kind(12) (fp64);
# the JAX kernel is fp64, so the oracle is a true fp64 reference (NOT fp32).
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
WRF_KIND="${WRF_PHYS}/ccpp_kind_types.F"
WRF_CORE="${WRF_PHYS}/physics_mmm/cu_ntiedtke.F90"
WRF_DRV="${WRF_PHYS}/module_cu_ntiedtke.F"
OUT_SAVE="${HERE}/../../savepoints/cumulus"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2

cp "${WRF_KIND}" "${HERE}/ccpp_kind_types.F"
cp "${WRF_CORE}" "${HERE}/cu_ntiedtke.F90"
cp "${WRF_DRV}"  "${HERE}/module_cu_ntiedtke.F"

cd "${HERE}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DDOUBLE_PRECISION -ffpe-summary=none -fallow-argument-mismatch"
taskset -c 0-3 gfortran ${FFLAGS} -c ccpp_kind_types.F
taskset -c 0-3 gfortran ${FFLAGS} -c cu_ntiedtke.F90
taskset -c 0-3 gfortran ${FFLAGS} -c module_cu_ntiedtke.F
taskset -c 0-3 gfortran ${FFLAGS} -c ntiedtke_oracle_driver.f90
taskset -c 0-3 gfortran -O2 -o ntiedtke_oracle \
    ccpp_kind_types.o cu_ntiedtke.o module_cu_ntiedtke.o ntiedtke_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./ntiedtke_oracle "$c" > "ntiedtke_case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "ntiedtke_case_${c}.txt" "${OUT_SAVE}/ntiedtke_case_${c}.json"
done

echo "OK: ntiedtke oracle built and 5 savepoints written to ${OUT_SAVE}"
