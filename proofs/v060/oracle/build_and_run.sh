#!/usr/bin/env bash
# Build the v0.6.0 WSM6 single-column oracle against the UNMODIFIED WRF
# physics_mmm WSM6 source (mp_wsm6.F90 + mp_wsm6_effectRad.F90 + deps) and
# emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved).
# Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: every WRF source file is copied VERBATIM (unmodified) from
# /home/enric/src/wrf_pristine/WRF/phys{,/physics_mmm}. The only project-authored
# Fortran is wsm6_oracle_driver.f90 (the column builder + dump), which never
# touches the scheme physics.
set -o pipefail   # NOT -e/-u: conda activate/deactivate hooks reference unset
                  # vars and emit benign warnings that would otherwise abort.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
MMM="${WRF_PHYS}/physics_mmm"
OUT_SAVE="${HERE}/../savepoints"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
# Define AFTER conda activation: the conda toolchain env exports its own BUILD
# (=x86_64-conda-linux-gnu), so we use a non-colliding name set post-activate.
BLDDIR="${HERE}/build"

# MODE: "fp32" (default, canonical WRF kind_phys=selected_real_kind(6)) or
# "fp64" (kind_phys override to selected_real_kind(15)) to cross-check that
# trace-cell effective-radius floor flips are fp32 detection-threshold dust in
# the reference, not port error. Pass as $1.
MODE="${1:-fp32}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_fp64"
fi

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}"
# copy pristine WRF sources verbatim
if [ "${MODE}" = "fp64" ]; then
  cp "${HERE}/ccpp_kind_types_fp64.f90"    "${BLDDIR}/ccpp_kind_types.f90"
else
  cp "${WRF_PHYS}/ccpp_kind_types.f90"     "${BLDDIR}/ccpp_kind_types.f90"
fi
cp "${MMM}/module_libmassv.F90"            "${BLDDIR}/module_libmassv.F90"
cp "${MMM}/mp_radar.F90"                   "${BLDDIR}/mp_radar.F90"
cp "${MMM}/mp_wsm6.F90"                    "${BLDDIR}/mp_wsm6.F90"
cp "${MMM}/mp_wsm6_effectRad.F90"          "${BLDDIR}/mp_wsm6_effectRad.F90"
cp "${HERE}/wsm6_oracle_driver.f90"        "${BLDDIR}/wsm6_oracle_driver.f90"

# record exact source checksums (provenance)
( cd "${BLDDIR}" && sha256sum ccpp_kind_types.f90 module_libmassv.F90 mp_radar.F90 \
     mp_wsm6.F90 mp_wsm6_effectRad.F90 > "${OUT_SAVE}/wrf_source_checksums.txt" 2>/dev/null ) || \
  mkdir -p "${OUT_SAVE}" && ( cd "${BLDDIR}" && sha256sum ccpp_kind_types.f90 \
     module_libmassv.F90 mp_radar.F90 mp_wsm6.F90 mp_wsm6_effectRad.F90 \
     > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
FFLAGS="-O2 -ffree-line-length-none -ffpe-summary=none"
taskset -c 0-3 gfortran ${FFLAGS} -c ccpp_kind_types.f90
taskset -c 0-3 gfortran ${FFLAGS} -c module_libmassv.F90
taskset -c 0-3 gfortran ${FFLAGS} -c mp_radar.F90
taskset -c 0-3 gfortran ${FFLAGS} -c mp_wsm6.F90
taskset -c 0-3 gfortran ${FFLAGS} -c mp_wsm6_effectRad.F90
taskset -c 0-3 gfortran ${FFLAGS} -c wsm6_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o wsm6_oracle \
    ccpp_kind_types.o module_libmassv.o mp_radar.o mp_wsm6.o \
    mp_wsm6_effectRad.o wsm6_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./wsm6_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/wsm6_case_${c}.json"
done
echo "OK: WSM6 oracle (mode=${MODE}) built and 6 savepoints written to ${OUT_SAVE}"
