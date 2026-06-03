#!/usr/bin/env bash
# Build the v0.6.0 Purdue-Lin (mp_physics=2) single-column oracle against the
# UNMODIFIED WRF phys/module_mp_lin.F scheme (+ its module_mp_radar dep) and
# emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved).
# Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: module_mp_lin.F and module_mp_radar.F are copied VERBATIM
# (unmodified) from /home/enric/src/wrf_pristine/WRF/phys. The only
# project-authored Fortran is lin_oracle_driver.f90 (the column builder +
# dump) and module_wrf_error.f90 (a minimal logging stub so the unresolved
# externals link), neither of which touches the scheme physics.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
OUT_SAVE="${HERE}/../savepoints_lin"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
BLDDIR="${HERE}/build_lin"

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}"
# copy pristine WRF sources verbatim
cp "${WRF_PHYS}/module_mp_radar.F"        "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_lin.F"          "${BLDDIR}/module_mp_lin.F"
cp "${HERE}/module_wrf_error_lin.f90"     "${BLDDIR}/module_wrf_error.f90"
cp "${HERE}/lin_oracle_driver.f90"        "${BLDDIR}/lin_oracle_driver.f90"

mkdir -p "${OUT_SAVE}"
# record exact source checksums (provenance) -- the scheme .F files only
( cd "${BLDDIR}" && sha256sum module_mp_lin.F module_mp_radar.F \
     > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
# Default WRF REAL = single precision (kind_phys not used by this scheme).
# WRF .F sources are free-form + C-preprocessed; force both for gfortran.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none -fno-range-check"
taskset -c 0-3 gfortran ${FFLAGS} -c module_wrf_error.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_mp_lin.F
taskset -c 0-3 gfortran ${FFLAGS} -c lin_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o lin_oracle \
    module_wrf_error.o module_mp_radar.o module_mp_lin.o lin_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./lin_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/lin_dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/lin_case_${c}.json"
done
echo "OK: Lin oracle built and 6 savepoints written to ${OUT_SAVE}"
