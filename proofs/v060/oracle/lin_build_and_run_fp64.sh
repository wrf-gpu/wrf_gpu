#!/usr/bin/env bash
# Build the v0.6.0 Purdue-Lin oracle in DOUBLE precision (gfortran
# -fdefault-real-8) against the same UNMODIFIED WRF module_mp_lin.F +
# module_mp_radar.F, and emit fp64 gold savepoints as JSON.
#
# Purpose: cross-check that the trace-cell condensate flips (a single cold cell
# with ~1e-5 kg/kg ice where the fp32 reference's qpz<qsat saturation decision
# differs from fp64) are an fp32 detection-threshold artifact of the REFERENCE,
# not a port error. Against this fp64 oracle the fp64 JAX port agrees to ~fp64.
#
# Scheme source is byte-identical to the fp32 build (same checksums); only the
# compiler's default REAL kind changes. CPU-only, pinned to cores 0-3.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
OUT_SAVE="${HERE}/../savepoints_lin_fp64"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
BLDDIR="${HERE}/build_lin_fp64"

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}"
cp "${WRF_PHYS}/module_mp_radar.F"        "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_lin.F"          "${BLDDIR}/module_mp_lin.F"
cp "${HERE}/module_wrf_error_lin.f90"     "${BLDDIR}/module_wrf_error.f90"
cp "${HERE}/lin_oracle_driver.f90"        "${BLDDIR}/lin_oracle_driver.f90"

mkdir -p "${OUT_SAVE}"
( cd "${BLDDIR}" && sha256sum module_mp_lin.F module_mp_radar.F \
     > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fno-range-check -fdefault-real-8 -fdefault-double-8"
taskset -c 0-3 gfortran ${FFLAGS} -c module_wrf_error.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_mp_lin.F
taskset -c 0-3 gfortran ${FFLAGS} -c lin_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o lin_oracle_fp64 \
    module_wrf_error.o module_mp_radar.o module_mp_lin.o lin_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./lin_oracle_fp64 "$c" > "case_${c}.txt"
  python3 "${HERE}/lin_dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/lin_case_${c}.json"
done
echo "OK: Lin fp64 oracle built and 6 savepoints written to ${OUT_SAVE}"
