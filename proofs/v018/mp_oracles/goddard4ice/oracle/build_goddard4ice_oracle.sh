#!/usr/bin/env bash
# Build the Goddard GCE 4-ice (WRF mp_physics=7, GSFCGCE_4ICE_NUWRF)
# single-column savepoint oracle against the UNMODIFIED pristine WRF source
# phys/module_mp_gsfcgce_4ice_nuwrf.F (+ its real dependency
# phys/module_mp_radar.F and share/module_model_constants.F).
#
# The 4-ice scheme carries HAIL as its OWN prognostic category (qh), distinct
# from graupel (qg) -- the distinction vs the 3-ice gsfcgce. The public entry
# gsfcgce_4ice_nuwrf internally calls consat_s (constants) every call and
# saticel_s (the saturation-adjustment + 4-ice core), and runs radar_init on
# the first timestep, so NO separate init entry is needed.
#
# Usage:
#   bash proofs/v017_goddard4ice/oracle/build_goddard4ice_oracle.sh fp32
#   bash proofs/v017_goddard4ice/oracle/build_goddard4ice_oracle.sh fp64
#
# fp32 = canonical WRF single precision (binding PROGNOSTIC reference).
# fp64 = -fdefault-real-8 override (scheme source otherwise UNMODIFIED); used
#        for the categorical effective-radius diagnostics only.
#
# Provenance: module_mp_gsfcgce_4ice_nuwrf.F, module_mp_radar.F, and
# module_model_constants.F are copied VERBATIM (unmodified) from
# <USER_HOME>/src/wrf_pristine/WRF. The only project-authored Fortran is:
#   * goddard4ice_oracle_driver.f90 -- the single-column builder + dump (never
#                                      touches scheme physics).
#   * goddard4ice_stub_modules.f90  -- module_wrf_error + the global wrf_debug /
#                                      wrf_message / wrf_error_fatal /
#                                      wrf_dm_on_monitor (=.TRUE., serial monitor
#                                      identity). NO physics.
#
# WRF_CHEM is NOT defined, so `use module_gocart_coupling` and the inline-Gocart
# coupling args are compiled OUT (the #if (WRF_CHEM==1) guards). The scheme uses
# no VREC/VSQRT, so frame/libmassv.F is not required.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -e
export OMP_NUM_THREADS=2

MODE="${1:-fp32}"
OUT_SAVE="${HERE}/../savepoints_goddard4ice"
FFLAGS_FREE="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none -fallow-argument-mismatch -I${WRF_PHYS}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_goddard4ice_fp64"
  FFLAGS_FREE="${FFLAGS_FREE} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
fi

BLDDIR="${HERE}/build_goddard4ice_${MODE}"
rm -rf "${BLDDIR}"
mkdir -p "${BLDDIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F"        "${BLDDIR}/module_model_constants.F"
cp "${WRF_PHYS}/module_mp_radar.F"                "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_gsfcgce_4ice_nuwrf.F"   "${BLDDIR}/module_mp_gsfcgce_4ice_nuwrf.F"
cp "${HERE}/goddard4ice_stub_modules.f90"         "${BLDDIR}/goddard4ice_stub_modules.f90"
cp "${HERE}/goddard4ice_oracle_driver.f90"        "${BLDDIR}/goddard4ice_oracle_driver.f90"

( cd "${BLDDIR}" && sha256sum \
    module_model_constants.F module_mp_radar.F module_mp_gsfcgce_4ice_nuwrf.F \
    goddard4ice_stub_modules.f90 goddard4ice_oracle_driver.f90 \
    > "${OUT_SAVE}/goddard4ice_source_checksums.txt" )

cd "${BLDDIR}"
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c goddard4ice_stub_modules.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_gsfcgce_4ice_nuwrf.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c goddard4ice_oracle_driver.f90

taskset -c 0-3 gfortran ${FFLAGS_FREE} -o goddard4ice_oracle \
  goddard4ice_stub_modules.o module_model_constants.o \
  module_mp_radar.o module_mp_gsfcgce_4ice_nuwrf.o goddard4ice_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./goddard4ice_oracle "$c" > "goddard4ice_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/goddard4ice_dump_to_json.py" \
      "goddard4ice_case_${c}.txt" \
      "${OUT_SAVE}/goddard4ice_case_${c}.json" \
      "Goddard GCE 4-ice (mp_physics=7)"
done

echo "OK: Goddard 4-ice oracle (mode=${MODE}) wrote savepoints to ${OUT_SAVE}"
