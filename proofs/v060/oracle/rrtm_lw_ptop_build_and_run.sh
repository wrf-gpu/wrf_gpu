#!/usr/bin/env bash
# Build the v0.13 NON-5000-PTOP classic RRTM longwave oracle against the SAME
# UNMODIFIED WRF phys/module_ra_rrtm.F + module_ra_clWRF_support.F and emit gold
# savepoints at p_top != 5000 Pa (low-top 100 mb, high-top 20 mb).  This proves
# the grid-aware buffer sizing (Finding F1 fix) matches WRF at a non-default top.
#
# CPU-only, pinned to cores 0-3, conda env `wrfbuild`.  fp64 only (canonical).
# Usage:  rrtm_lw_ptop_build_and_run.sh
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
WRF_RUN="/home/enric/src/wrf_pristine/WRF/run"
OUT_SAVE="${HERE}/../savepoints_fp64"
PREC_FLAGS="-fdefault-real-8 -fdefault-double-8 -DRWORDSIZE=8 -DIWORDSIZE=4"
BUILD_DIR="${HERE}/build_rrtmlw_ptop_fp64"
DATA_FILE="${WRF_RUN}/RRTM_DATA_DBL"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_ra_rrtm.F"            "${BUILD_DIR}/module_ra_rrtm.F"
cp "${WRF_PHYS}/module_ra_clWRF_support.F"   "${BUILD_DIR}/module_ra_clWRF_support.F"
cp "${HERE}/module_wrf_error.f90"            "${BUILD_DIR}/module_wrf_error.f90"
cp "${HERE}/rrtm_dm_stubs.f90"               "${BUILD_DIR}/rrtm_dm_stubs.f90"
cp "${HERE}/rrtm_lw_ptop_oracle_driver.f90"  "${BUILD_DIR}/rrtm_lw_ptop_oracle_driver.f90"
cp "${DATA_FILE}"                            "${BUILD_DIR}/RRTM_DATA"

( cd "${BUILD_DIR}" && sha256sum module_ra_rrtm.F module_ra_clWRF_support.F RRTM_DATA \
    > "${OUT_SAVE}/rrtm_lw_ptop_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fallow-argument-mismatch -fconvert=big-endian -frecord-marker=4 ${PREC_FLAGS}"
taskset -c 0-3 gfortran ${FFLAGS} -c module_wrf_error.f90
taskset -c 0-3 gfortran ${FFLAGS} -c rrtm_dm_stubs.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_clWRF_support.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_rrtm.F
taskset -c 0-3 gfortran ${FFLAGS} -c rrtm_lw_ptop_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o rrtm_lw_ptop_oracle \
    module_wrf_error.o rrtm_dm_stubs.o module_ra_clWRF_support.o \
    module_ra_rrtm.o rrtm_lw_ptop_oracle_driver.o

{
  echo "build_mode=fp64"
  echo "fflags=${FFLAGS}"
  echo "gfortran=$(gfortran --version | head -1)"
  echo "rrtm_data=${DATA_FILE}"
  echo "wrf_source=${WRF_PHYS}/module_ra_rrtm.F"
  echo "note=NON-5000-ptop savepoints (Finding F1 grid-aware buffer fix)"
} > "${OUT_SAVE}/rrtm_lw_ptop_build_manifest.txt"

for c in 1 2; do
  taskset -c 0-3 ./rrtm_lw_ptop_oracle "$c" > "rrtm_lw_ptop_case_${c}.txt"
  python3 "${HERE}/rrtm_lw_dump_to_json.py" "rrtm_lw_ptop_case_${c}.txt" \
      "${OUT_SAVE}/rrtm_lw_ptop_case_${c}.json"
done
echo "OK (fp64): non-5000-ptop classic-RRTM LW oracle built; 2 savepoints in ${OUT_SAVE}"
