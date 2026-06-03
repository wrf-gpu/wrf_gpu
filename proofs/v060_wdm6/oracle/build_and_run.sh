#!/usr/bin/env bash
# Build the v0.6.0 WDM6 single-column oracle against the UNMODIFIED WRF
# module_mp_wdm6.F (+ frame/libmassv.F for vsrec/vssqrt) and emit gold
# savepoints as JSON.
#
# CPU-only, pinned to cores 0-3 (the GPU + cores 4-31 are reserved).
# Requires the conda env `wrfbuild` (gfortran 14.x).
#
# Provenance: module_mp_wdm6.F, phys/module_mp_radar.F, and frame/libmassv.F
# are copied VERBATIM (unmodified) from /home/enric/src/wrf_pristine/WRF. The
# only project-authored Fortran is:
#   * wdm6_oracle_driver.f90          -- the column builder + dump (never
#                                        touches scheme physics)
#   * stub_module_wrf_error.F90       -- no-op wrf_debug/message (module_mp_radar
#                                        uses only wrf_debug; serial oracle)
#   * stub_module_model_constants.F90 -- the 3 RE_*_BG background radii, exact
#                                        pristine values.
# The microphysics mass + Nc/Nr/Nn tendencies come from the UNMODIFIED scheme
# (module_mp_wdm6.F), and the radar reflectivity helper compiles against the
# UNMODIFIED phys/module_mp_radar.F. The reflectivity DIAGNOSTIC itself is not
# enabled in the savepoints (diagflag is never passed) and is out of scope.
set -o pipefail   # NOT -e/-u: conda activate/deactivate hooks reference unset
                  # vars and emit benign warnings that would otherwise abort.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_FRAME="${WRF}/frame"
OUT_SAVE="${HERE}/../savepoints"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
BLDDIR="${HERE}/build"

# MODE: "fp32" (default, canonical classic-WRF single precision) or "fp64"
# (-fdefault-real-8, scheme source otherwise UNMODIFIED) to cross-check that
# trace-cell floor flips are fp32 detection-threshold dust in the reference,
# not port error. Pass as $1.
MODE="${1:-fp32}"
FP64FLAG=""
DRVDEF=""
WDM6DEF=""
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_fp64"
  # Promote the scheme's bare `real` to 8 bytes. CRUCIAL: also pass
  # -DDOUBLE_PRECISION so module_mp_wdm6.F's VREC/VSQRT macros select the
  # real*8 libmassv variants (vrec/vsqrt) instead of the real*4 ones
  # (vsrec/vssqrt). Without this, the promoted real*8 denfac arrays are passed
  # to real*4 subroutine args and corrupt the fall-speed/precip (RAIN denormal).
  FP64FLAG="-fdefault-real-8 -fdefault-double-8"
  DRVDEF="-DWDM6_FP64"
  WDM6DEF="-DDOUBLE_PRECISION"
fi

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}"
# copy pristine WRF sources verbatim
cp "${WRF_PHYS}/module_mp_wdm6.F"            "${BLDDIR}/module_mp_wdm6.F"
cp "${WRF_PHYS}/module_mp_radar.F"           "${BLDDIR}/module_mp_radar.F"
cp "${WRF_FRAME}/libmassv.F"                 "${BLDDIR}/libmassv.F"
cp "${HERE}/stub_module_wrf_error.F90"       "${BLDDIR}/stub_module_wrf_error.F90"
cp "${HERE}/stub_module_model_constants.F90" "${BLDDIR}/stub_module_model_constants.F90"
cp "${HERE}/wdm6_oracle_driver.f90"          "${BLDDIR}/wdm6_oracle_driver.F90"

# record exact source checksums (provenance). The UNMODIFIED scheme + radar are
# the WRF files; the stubs/driver are project-authored.
mkdir -p "${OUT_SAVE}"
( cd "${BLDDIR}" && sha256sum module_mp_wdm6.F module_mp_radar.F libmassv.F \
     stub_module_wrf_error.F90 stub_module_model_constants.F90 \
     wdm6_oracle_driver.F90 > "${OUT_SAVE}/wrf_source_checksums.txt" )

cd "${BLDDIR}"
# -O2 matches WRF default opt. module_mp_wdm6.F and module_mp_radar.F are
# FREE-form (despite .F) and module_mp_wdm6.F uses cpp (#ifndef) -> -cpp.
FFLAGS="-O2 -ffpe-summary=none ${FP64FLAG}"

taskset -c 0-3 gfortran ${FFLAGS} -ffixed-form -c libmassv.F -o libmassv.o
taskset -c 0-3 gfortran ${FFLAGS} -c stub_module_wrf_error.F90
taskset -c 0-3 gfortran ${FFLAGS} -c stub_module_model_constants.F90
taskset -c 0-3 gfortran ${FFLAGS} -cpp -ffree-form -ffree-line-length-none \
    -c module_mp_radar.F
taskset -c 0-3 gfortran ${FFLAGS} ${WDM6DEF} -cpp -ffree-form -ffree-line-length-none \
    -c module_mp_wdm6.F
taskset -c 0-3 gfortran ${FFLAGS} ${DRVDEF} -cpp -ffree-form -ffree-line-length-none \
    -c wdm6_oracle_driver.F90
taskset -c 0-3 gfortran ${FFLAGS} -o wdm6_oracle \
    libmassv.o stub_module_wrf_error.o stub_module_model_constants.o \
    module_mp_radar.o module_mp_wdm6.o wdm6_oracle_driver.o

for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./wdm6_oracle "$c" > "case_${c}.txt"
  python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/wdm6_case_${c}.json"
done
echo "OK: WDM6 oracle (mode=${MODE}) built and 6 savepoints written to ${OUT_SAVE}"
