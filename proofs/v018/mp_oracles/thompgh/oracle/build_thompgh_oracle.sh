#!/usr/bin/env bash
# Build the Thompson graupel-hail (mp_physics=38, THOMPSONGH) single-column
# savepoint oracle against the UNMODIFIED pristine WRF source
# phys/module_mp_thompson.F (+ its real deps module_mp_radar.F, the VREC/VSQRT
# library frame/libmassv.F, and share/module_model_constants.F).
#
# The is_hail_aware (mp=38) variable-density-graupel path is exercised: the
# driver calls thompson_init WITH `ng` PRESENT, which builds the 9-density-plane
# (NRHG=9) collision lookup tables (qr_acr_qg_mp38V1) IN-MEMORY. This table
# generation is the dominant cost (a 6-deep quadrature nest over
# ntb_r*ntb_r1=1369 x NRHG=9 x ntb_g*ntb_g1=1369 x inner bins); expect MINUTES.
#
# Usage:
#   bash proofs/v017_thompgh/oracle/build_thompgh_oracle.sh fp32
#   bash proofs/v017_thompgh/oracle/build_thompgh_oracle.sh fp64
#
# fp32 = canonical WRF single precision (binding PROGNOSTIC reference).
# fp64 = -fdefault-real-8 override (scheme source otherwise UNMODIFIED); used
#        for the categorical effective-radius diagnostics only.
#
# Provenance: module_mp_thompson.F, module_mp_radar.F, libmassv.F, and
# module_model_constants.F are copied VERBATIM (unmodified) from
# /home/user/src/wrf_pristine/WRF. The only project-authored Fortran is:
#   * thompgh_oracle_driver.f90  -- the single-column builder + dump (never
#                                   touches scheme physics).
#   * thompgh_stub_modules.f90   -- module_wrf_error / module_dm(wrf_dm_max_real)
#                                   / module_timing + the global wrf_debug /
#                                   wrf_message / wrf_error_fatal /
#                                   wrf_dm_on_monitor / wrf_dm_decomp1d (serial
#                                   identity) / wrf_dm_gatherv (serial no-op) /
#                                   nl_get_{force_read_thompson,
#                                   write_thompson_tables,write_thompson_mp38table}.
#                                   All serial reductions / production control
#                                   values -- NO physics.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/user/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
WRF_FRAME="${WRF}/frame"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -e
export OMP_NUM_THREADS=2

MODE="${1:-fp32}"
OUT_SAVE="${HERE}/../savepoints_thompgh"
FFLAGS_FREE="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none -fallow-argument-mismatch -I${WRF_PHYS}"
FFLAGS_FIXED="-O2 -cpp -ffixed-form -ffixed-line-length-none -ffpe-summary=none -fallow-argument-mismatch -I${WRF_PHYS}"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_thompgh_fp64"
  FFLAGS_FREE="${FFLAGS_FREE} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  FFLAGS_FIXED="${FFLAGS_FIXED} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
fi

BLDDIR="${HERE}/build_thompgh_${MODE}"
rm -rf "${BLDDIR}"
mkdir -p "${BLDDIR}" "${OUT_SAVE}"

cp "${WRF_SHARE}/module_model_constants.F" "${BLDDIR}/module_model_constants.F"
cp "${WRF_FRAME}/libmassv.F"               "${BLDDIR}/libmassv.F"
cp "${WRF_PHYS}/module_mp_radar.F"         "${BLDDIR}/module_mp_radar.F"
cp "${WRF_PHYS}/module_mp_thompson.F"      "${BLDDIR}/module_mp_thompson.F"
cp "${HERE}/thompgh_stub_modules.f90"      "${BLDDIR}/thompgh_stub_modules.f90"
cp "${HERE}/thompgh_oracle_driver.f90"     "${BLDDIR}/thompgh_oracle_driver.f90"

( cd "${BLDDIR}" && sha256sum \
    module_model_constants.F libmassv.F module_mp_radar.F module_mp_thompson.F \
    thompgh_stub_modules.f90 thompgh_oracle_driver.f90 \
    > "${OUT_SAVE}/thompgh_source_checksums.txt" )

cd "${BLDDIR}"
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c thompgh_stub_modules.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS_FIXED} -c libmassv.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_thompson.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c thompgh_oracle_driver.f90

taskset -c 0-3 gfortran ${FFLAGS_FREE} -o thompgh_oracle \
  thompgh_stub_modules.o module_model_constants.o libmassv.o \
  module_mp_radar.o module_mp_thompson.o thompgh_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./thompgh_oracle "$c" > "thompgh_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/thompgh_dump_to_json.py" "thompgh_case_${c}.txt" \
      "${OUT_SAVE}/thompgh_case_${c}.json" "Thompson graupel-hail (mp_physics=38)"
done

echo "OK: Thompson-GH oracle (mode=${MODE}) wrote savepoints to ${OUT_SAVE}"
