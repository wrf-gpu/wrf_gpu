#!/usr/bin/env bash
# Build the v0.18 KEPS PBL reference oracle (bl_pbl_physics=17)
# against the UNMODIFIED pristine WRF module_bl_keps.F and emit fp64
# single-column savepoints as JSON.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/user/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
SAVE="${HERE}/../../savepoints_fp64/keps"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

BUILD="${HERE}/build_fp64"
rm -rf "${BUILD}"
mkdir -p "${BUILD}" "${SAVE}"

cp "${WRF_PHYS}/module_bl_keps.F" "${BUILD}/module_bl_keps.F"
cp "${HERE}/keps_oracle_driver.f90" "${BUILD}/keps_oracle_driver.f90"

( cd "${BUILD}" && sha256sum module_bl_keps.F > "${SAVE}/keps_wrf_source_checksums.txt" )

cd "${BUILD}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0 -ffpe-summary=none -fdefault-real-8 -fdefault-double-8"
taskset -c 0-3 gfortran ${FFLAGS} -c module_bl_keps.F
taskset -c 0-3 gfortran ${FFLAGS} -c keps_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o keps_oracle module_bl_keps.o keps_oracle_driver.o

{
  echo "mode=fp64"
  echo "full_wrf_exe=false"
  echo "wrf_sources=${WRF_PHYS}/module_bl_keps.F"
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags=${FFLAGS}"
} > "${SAVE}/keps_build_manifest.txt"

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./keps_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/../dump_to_json.py" "case_${c}.txt" "${SAVE}/keps_case_${c}.json"
done

echo "OK: KEPS oracle built; fp64 -> ${SAVE}"
