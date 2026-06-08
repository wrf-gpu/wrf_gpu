#!/usr/bin/env bash
# Build the GSFC (Chou-Suarez) shortwave (ra_sw_physics=2) single-column
# oracle against UNMODIFIED WRF phys/module_ra_gsfcsw.F and emit gold
# savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
# Usage:  gsfcsw_build_and_run.sh            # default REAL*4 build  -> savepoints_gsfcsw
#         gsfcsw_build_and_run.sh fp64       # -fdefault-real-8      -> savepoints_gsfcsw_fp64
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
MODE="${1:-fp32}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_gsfcsw_fp64"
  PREC_FLAGS="-fdefault-real-8 -fdefault-double-8"
  BUILD_DIR="${HERE}/build_gsfcsw_fp64"
else
  OUT_SAVE="${HERE}/../savepoints_gsfcsw"
  PREC_FLAGS=""
  BUILD_DIR="${HERE}/build_gsfcsw"
fi

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_ra_gsfcsw.F" "${BUILD_DIR}/module_ra_gsfcsw.F"
cp "${HERE}/gsfcsw_stubs.f90"        "${BUILD_DIR}/gsfcsw_stubs.f90"
cp "${HERE}/gsfcsw_oracle_driver.f90" "${BUILD_DIR}/gsfcsw_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_ra_gsfcsw.F \
    > "${OUT_SAVE}/gsfcsw_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
# WRF_CHEM undefined -> aerosol-feedback code is preprocessed out; the optional
# aerosol args are simply absent at the call site. -fallow-argument-mismatch is
# not needed (clean interface), but harmless if present in newer gfortran.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none ${PREC_FLAGS}"
taskset -c 0-3 gfortran ${FFLAGS} -c gsfcsw_stubs.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_gsfcsw.F
taskset -c 0-3 gfortran ${FFLAGS} -c gsfcsw_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o gsfcsw_oracle \
    module_ra_gsfcsw.o gsfcsw_oracle_driver.o gsfcsw_stubs.o

{
  echo "build_mode=${MODE}"
  echo "fflags=${FFLAGS}"
  echo "gfortran=$(gfortran --version | head -1)"
  echo "wrf_source=${WRF_PHYS}/module_ra_gsfcsw.F"
} > "${OUT_SAVE}/gsfcsw_build_manifest.txt"

for c in 1 2 3 4 5 6 7; do
  taskset -c 0-3 ./gsfcsw_oracle "$c" > "gsfcsw_case_${c}.txt"
  python3 "${HERE}/gsfcsw_dump_to_json.py" "gsfcsw_case_${c}.txt" "${OUT_SAVE}/gsfcsw_case_${c}.json"
done
echo "OK (${MODE}): GSFC SW oracle built and 7 savepoints written to ${OUT_SAVE}"
