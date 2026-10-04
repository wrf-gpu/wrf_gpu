#!/usr/bin/env bash
# Build the v0.17 RUC land-surface model oracle against UNMODIFIED WRF
# phys/module_sf_ruclsm.F (subroutine LSMRUC + internal SOILVEGIN/SFCTMP and
# the RUCLSM_SOILVEGPARM table reader) and emit fp64 JSON savepoints.
# CPU-only, pinned to cores 0-3.  Requires conda env `wrfbuild`.
#
# LSMRUC `use module_model_constants` and `use module_wrf_error`; both are
# compiled from pristine source with the SAME fp64 flags.  RUCLSM_SOILVEGPARM
# reads VEGPARM.TBL / SOILPARM.TBL / GENPARM.TBL (copied here from pristine
# WRF/run) at runtime; the per-rank distributed-memory broadcast routines
# (wrf_dm_on_monitor / wrf_dm_bcast_*) and wrf_debug / wrf_abort are satisfied
# by tiny serial stubs.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"
WRF_FRAME="${WRF}/frame"
WRF_RUN="${WRF}/run"
OUT_SAVE="${HERE}/../../savepoints/ruclsm/fp64"

# Temporarily relax -u for conda hooks that reference unset vars.
set +u
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
export OMP_NUM_THREADS=2
set -e

BUILD_DIR="${HERE}/build_fp64"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_SAVE}"

# --- pristine WRF sources (UNMODIFIED) ---
cp "${WRF_SHARE}/module_model_constants.F" "${BUILD_DIR}/module_model_constants.F"
cp "${WRF_FRAME}/module_wrf_error.F"       "${BUILD_DIR}/module_wrf_error.F"
cp "${WRF_PHYS}/module_sf_ruclsm.F"        "${BUILD_DIR}/module_sf_ruclsm.F"
cp "${HERE}/ruclsm_oracle_driver.f90"      "${BUILD_DIR}/ruclsm_oracle_driver.f90"

# --- pristine RUC parameter tables (read at runtime) ---
cp "${WRF_RUN}/VEGPARM.TBL"  "${BUILD_DIR}/VEGPARM.TBL"
cp "${WRF_RUN}/SOILPARM.TBL" "${BUILD_DIR}/SOILPARM.TBL"
cp "${WRF_RUN}/GENPARM.TBL"  "${BUILD_DIR}/GENPARM.TBL"

( cd "${BUILD_DIR}" && sha256sum \
    module_sf_ruclsm.F module_wrf_error.F module_model_constants.F \
    VEGPARM.TBL SOILPARM.TBL GENPARM.TBL \
    > "${OUT_SAVE}/ruclsm_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"

# WRF distributed-memory + debug stubs (serial single-rank).
#  - wrf_dm_on_monitor() -> .TRUE. so RUCLSM_SOILVEGPARM reads the tables.
#  - wrf_dm_bcast_*       -> no-op (single rank).
#  - wrf_debug / wrf_abort -> referenced by module_sf_ruclsm + module_wrf_error.
cat > _wrfstubs.f90 <<'EOF'
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_string(buf, n)
  character(len=*), intent(inout) :: buf
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_string

subroutine wrf_dm_bcast_integer(buf, n)
  integer, intent(in) :: n
  integer, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_real(buf, n)
  integer, intent(in) :: n
  real, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_real

subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

subroutine wrf_abort()
  stop 1
end subroutine wrf_abort
EOF

# fp64 throughout (-fdefault-real-8 -fdefault-double-8).  EM_CORE=1 so the
# LSMRUC signature includes the SPP / lake / precip-constituent arguments
# matched by the driver.
# -fallow-argument-mismatch: WRF calls wrf_dm_bcast_real with both array and
# scalar actual args (rank mismatch across call sites); WRF builds with this
# relaxation on gfortran 10+.  This does NOT modify WRF source.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none -fdefault-real-8 -fdefault-double-8 -fallow-argument-mismatch"

taskset -c 0-3 gfortran ${FFLAGS} -c module_model_constants.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_wrf_error.F
taskset -c 0-3 gfortran ${FFLAGS} -c module_sf_ruclsm.F
taskset -c 0-3 gfortran ${FFLAGS} -c _wrfstubs.f90
taskset -c 0-3 gfortran ${FFLAGS} -c ruclsm_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o ruclsm_oracle \
    ruclsm_oracle_driver.o module_sf_ruclsm.o module_wrf_error.o \
    module_model_constants.o _wrfstubs.o

{
  echo "mode=fp64"
  echo "full_wrf_exe=false"
  echo "wrf_sources=${WRF_PHYS}/module_sf_ruclsm.F ${WRF_FRAME}/module_wrf_error.F ${WRF_SHARE}/module_model_constants.F"
  echo "wrf_tables=${WRF_RUN}/VEGPARM.TBL ${WRF_RUN}/SOILPARM.TBL ${WRF_RUN}/GENPARM.TBL"
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags=${FFLAGS}"
} > "${OUT_SAVE}/ruclsm_build_manifest.txt"

# Single invocation drives all regimes (RUC tables are loaded once internally).
taskset -c 0-3 ./ruclsm_oracle fp64 > "ruclsm_fp64.txt"
python3 "${HERE}/ruclsm_dump_to_json.py" "ruclsm_fp64.txt" "${OUT_SAVE}/ruclsm_fp64.json"

echo "OK: RUC LSM fp64 oracle built and savepoints written under ${OUT_SAVE}"
