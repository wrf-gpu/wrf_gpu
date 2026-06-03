#!/usr/bin/env bash
# Build the v0.6.0 classic RRTM longwave (ra_lw_physics=1) single-column oracle
# against UNMODIFIED WRF phys/module_ra_rrtm.F + module_ra_clWRF_support.F and
# emit gold savepoints as JSON. The 16-band k-distribution lookup tables are
# read from the unmodified RRTM_DATA / RRTM_DATA_DBL assets.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
# Usage:  rrtm_lw_build_and_run.sh            # default REAL*4 build (RRTM_DATA)
#         rrtm_lw_build_and_run.sh fp64       # -fdefault-real-8 (RRTM_DATA_DBL)
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
WRF_RUN="/home/enric/src/wrf_pristine/WRF/run"
MODE="${1:-fp32}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_fp64"
  PREC_FLAGS="-fdefault-real-8 -fdefault-double-8 -DRWORDSIZE=8 -DIWORDSIZE=4"
  BUILD_DIR="${HERE}/build_rrtmlw_fp64"
  DATA_FILE="${WRF_RUN}/RRTM_DATA_DBL"
else
  OUT_SAVE="${HERE}/../savepoints"
  PREC_FLAGS="-DRWORDSIZE=4 -DIWORDSIZE=4"
  BUILD_DIR="${HERE}/build_rrtmlw"
  DATA_FILE="${WRF_RUN}/RRTM_DATA"
fi

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_ra_rrtm.F"          "${BUILD_DIR}/module_ra_rrtm.F"
cp "${WRF_PHYS}/module_ra_clWRF_support.F" "${BUILD_DIR}/module_ra_clWRF_support.F"
cp "${HERE}/module_wrf_error.f90"          "${BUILD_DIR}/module_wrf_error.f90"
cp "${HERE}/rrtm_dm_stubs.f90"             "${BUILD_DIR}/rrtm_dm_stubs.f90"
cp "${HERE}/rrtm_lw_oracle_driver.f90"     "${BUILD_DIR}/rrtm_lw_oracle_driver.f90"
cp "${DATA_FILE}"                          "${BUILD_DIR}/RRTM_DATA"

( cd "${BUILD_DIR}" && sha256sum module_ra_rrtm.F module_ra_clWRF_support.F RRTM_DATA \
    > "${OUT_SAVE}/rrtm_lw_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
# WRF ships RRTM_DATA / RRTM_DATA_DBL as big-endian unformatted with 4-byte
# record markers (configure.wrf BYTESWAPIO = -fconvert=big-endian
# -frecord-marker=4); match that or the lookup-table read fails.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fallow-argument-mismatch -fconvert=big-endian -frecord-marker=4 ${PREC_FLAGS}"
taskset -c 0-3 gfortran ${FFLAGS} -c module_wrf_error.f90
taskset -c 0-3 gfortran ${FFLAGS} -c rrtm_dm_stubs.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_clWRF_support.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_rrtm.F
taskset -c 0-3 gfortran ${FFLAGS} -c rrtm_lw_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o rrtm_lw_oracle \
    module_wrf_error.o rrtm_dm_stubs.o module_ra_clWRF_support.o \
    module_ra_rrtm.o rrtm_lw_oracle_driver.o

{
  echo "build_mode=${MODE}"
  echo "fflags=${FFLAGS}"
  echo "gfortran=$(gfortran --version | head -1)"
  echo "rrtm_data=${DATA_FILE}"
  echo "wrf_source=${WRF_PHYS}/module_ra_rrtm.F"
} > "${OUT_SAVE}/rrtm_lw_build_manifest.txt"

for c in 1 2 3 4 5 6 7; do
  taskset -c 0-3 ./rrtm_lw_oracle "$c" > "rrtm_lw_case_${c}.txt"
  python3 "${HERE}/rrtm_lw_dump_to_json.py" "rrtm_lw_case_${c}.txt" "${OUT_SAVE}/rrtm_lw_case_${c}.json"
done
echo "OK (${MODE}): classic-RRTM LW oracle built and 7 savepoints written to ${OUT_SAVE}"
