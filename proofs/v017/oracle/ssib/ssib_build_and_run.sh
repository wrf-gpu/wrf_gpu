#!/usr/bin/env bash
# Build the v0.17 SSiB land-surface model oracle against UNMODIFIED WRF
# phys/module_sf_ssib.F (subroutine SSIB at line 885, sf_surface_physics=8) and
# emit fp64 JSON savepoints.  CPU-only, pinned to cores 0-3.  Requires conda env
# `wrfbuild`.
#
# SSiB is self-contained: VEGOUT reads ALL vegetation/soil parameters from
# module-level DATA arrays (no SSIBPARM.TBL / LANDUSE.TBL / SSIB_INIT), there are
# NO preprocessor directives, and the module does NOT `USE module_model_constants`
# (its own CPAIR/GASR/TF/GRAV/STEFAN/HLAT parameters).  The only external symbols
# are wrf_message / wrf_error_fatal / wrf_debug; tiny stubs satisfy the linker.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
OUT_SAVE="${HERE}/../../savepoints/ssib/fp64"

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

cp "${WRF_PHYS}/module_sf_ssib.F"   "${BUILD_DIR}/module_sf_ssib.F"
cp "${HERE}/ssib_oracle_driver.f90" "${BUILD_DIR}/ssib_oracle_driver.f90"

( cd "${BUILD_DIR}" && sha256sum module_sf_ssib.F \
    > "${OUT_SAVE}/ssib_wrf_source_checksums.txt" )

cd "${BUILD_DIR}"

# WRF runtime stubs (SSIB references wrf_message/wrf_error_fatal/wrf_debug).
cat > _wrfstubs.f90 <<'EOF'
subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(*,'(A)') trim(msg)
end subroutine wrf_message
subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'FATAL: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal
subroutine wrf_error_fatal3(fname, line, msg)
  character(len=*), intent(in) :: fname, msg
  integer, intent(in) :: line
  write(0,'(A)') 'FATAL: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal3
subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
EOF

# fp64 throughout (-fdefault-real-8 -fdefault-double-8).  No preprocessor blocks
# in module_sf_ssib.F, but -cpp is harmless and matches the PX template.
# -fallow-argument-mismatch: WRF's OWN standard build flag (configure.wrf for
# gfortran>=10).  Required here because module_sf_ssib.F has a latent source bug:
# in SUBROUTINE SNRESULT the local `message` is undeclared, so implicit typing
# makes it INTEGER (names m..n), while in SSIB it is CHARACTER*256; both are
# passed to the external wrf_message, which gfortran 14 rejects as a hard error
# without this flag.  We do NOT modify the pristine source.
FFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none -fdefault-real-8 -fdefault-double-8 -fallow-argument-mismatch"

taskset -c 0-3 gfortran ${FFLAGS} -c module_sf_ssib.F
taskset -c 0-3 gfortran ${FFLAGS} -c _wrfstubs.f90
taskset -c 0-3 gfortran ${FFLAGS} -c ssib_oracle_driver.f90
taskset -c 0-3 gfortran ${FFLAGS} -o ssib_oracle \
    ssib_oracle_driver.o module_sf_ssib.o _wrfstubs.o

{
  echo "mode=fp64"
  echo "full_wrf_exe=false"
  echo "wrf_sources=${WRF_PHYS}/module_sf_ssib.F"
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags=${FFLAGS}"
} > "${OUT_SAVE}/ssib_build_manifest.txt"

for c in 1 2 3 4 5; do
  taskset -c 0-3 ./ssib_oracle "$c" fp64 > "case_${c}.txt"
  python3 "${HERE}/ssib_dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/ssib_case_${c}.json"
done

echo "OK: SSiB LSM fp64 oracle built and savepoints written under ${OUT_SAVE}"
