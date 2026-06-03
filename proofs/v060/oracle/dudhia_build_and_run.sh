#!/usr/bin/env bash
# Build the v0.6.0 Dudhia shortwave (ra_sw_physics=1) single-column oracle
# against UNMODIFIED WRF phys/module_ra_sw.F and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
# Usage:  dudhia_build_and_run.sh            # default REAL*4 build -> savepoints
#         dudhia_build_and_run.sh fp64       # -fdefault-real-8 -> savepoints_fp64
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
MODE="${1:-fp32}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_fp64"
  PREC_FLAGS="-fdefault-real-8 -fdefault-double-8"
  BUILD_DIR="${HERE}/build_dudhia_fp64"
else
  OUT_SAVE="${HERE}/../savepoints"
  PREC_FLAGS=""
  BUILD_DIR="${HERE}/build_dudhia"
fi

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_ra_sw.F" "${BUILD_DIR}/module_ra_sw.F"
cp "${HERE}/dudhia_oracle_driver.f90" "${BUILD_DIR}/dudhia_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_ra_sw.F \
    > "${OUT_SAVE}/dudhia_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none ${PREC_FLAGS}"
taskset -c 0-3 gfortran ${FFLAGS} -c module_ra_sw.F
taskset -c 0-3 gfortran ${FFLAGS} -c dudhia_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o dudhia_oracle \
    module_ra_sw.o dudhia_oracle_driver.o

{
  echo "build_mode=${MODE}"
  echo "fflags=${FFLAGS}"
  echo "gfortran=$(gfortran --version | head -1)"
  echo "wrf_source=${WRF_PHYS}/module_ra_sw.F"
} > "${OUT_SAVE}/dudhia_build_manifest.txt"

for c in 1 2 3 4 5 6 7; do
  taskset -c 0-3 ./dudhia_oracle "$c" > "dudhia_case_${c}.txt"
  python3 "${HERE}/dudhia_dump_to_json.py" "dudhia_case_${c}.txt" "${OUT_SAVE}/dudhia_case_${c}.json"
done
echo "OK (${MODE}): Dudhia SW oracle built and 7 savepoints written to ${OUT_SAVE}"
