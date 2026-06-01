#!/usr/bin/env bash
# Build the standalone WRF Noah-MP offline savepoint driver (Sprint 0b oracle).
# Links the COMPILED pristine WRF Noah-MP objects (no WRF rebuild). CPU-only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYS=/home/enric/src/wrf_pristine/WRF/phys

# WRF gfortran flags (configure.wrf: single precision, free form, big-endian I/O).
FFLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2 -I${PHYS}"

# Activate the conda env that built pristine WRF (matching gfortran ABI/.mod).
# Temporarily relax -e: some conda (de)activate hooks reference unset vars.
set +eu
# shellcheck disable=SC1091
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -eu

cd "${HERE}"

# Tiny stubs for the WRF runtime symbols the Noah-MP objects reference but that
# are never executed on the land path (urban/BEP/GFDL-radiation, wrf_message).
cat > _wrfstubs.F90 <<'EOF'
subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(*,'(A)') trim(msg)
end subroutine wrf_message
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

cat > _modstubs.F90 <<'EOF'
module module_sf_urban
contains
  subroutine urban(); end subroutine urban
  subroutine iri_scheme(); end subroutine iri_scheme
end module module_sf_urban
module module_sf_bep
contains
  subroutine bep(); end subroutine bep
end module module_sf_bep
module module_sf_bep_bem
contains
  subroutine bep_bem(); end subroutine bep_bem
end module module_sf_bep_bem
module module_ra_gfdleta
contains
  subroutine cal_mon_day(); end subroutine cal_mon_day
end module module_ra_gfdleta
EOF

gfortran ${FFLAGS} -c _wrfstubs.F90 -o _wrfstubs.o
gfortran ${FFLAGS} -c _modstubs.F90 -o _modstubs.o
gfortran ${FFLAGS} -c noahmp_offline_driver.F90 -o noahmp_offline_driver.o

gfortran noahmp_offline_driver.o _wrfstubs.o _modstubs.o \
  "${PHYS}/module_sf_noahmplsm.o" "${PHYS}/module_sf_noahmpdrv.o" \
  "${PHYS}/module_sf_gecros.o" "${PHYS}/module_sf_noahmp_glacier.o" \
  "${PHYS}/module_sf_noahmp_groundwater.o" \
  -o noahmp_offline_driver.exe

echo "BUILT noahmp_offline_driver.exe"
