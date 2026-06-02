#!/usr/bin/env bash
# Build the standalone WRF Noah-classic (sf_surface_physics=2) offline savepoint
# driver (v0.6.0 lane 14 oracle). Links the COMPILED pristine WRF Noah-classic
# objects (module_sf_noahlsm.o / module_sf_noahdrv.o) — no WRF rebuild. CPU-only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYS=/home/enric/src/wrf_pristine/WRF/phys

# WRF gfortran flags (configure.wrf: single precision, free form, big-endian I/O).
FFLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2 -I${PHYS}"

# Activate the conda env that built pristine WRF (matching gfortran ABI/.mod).
set +eu
# shellcheck disable=SC1091
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -eu

cd "${HERE}"

# WRF runtime symbol stubs referenced by the Noah-classic objects but never
# executed on the SF_URBAN_PHYSICS=0 / serial land path.
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
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor
subroutine wrf_dm_bcast_integer(buf, n)
  integer :: n, buf(*)
end subroutine wrf_dm_bcast_integer
subroutine wrf_dm_bcast_real(buf, n)
  integer :: n
  real :: buf(*)
end subroutine wrf_dm_bcast_real
subroutine wrf_dm_bcast_string(buf, n)
  integer :: n
  character(len=*) :: buf
end subroutine wrf_dm_bcast_string
EOF

# Module stubs for the urban/BEP/GFDL-radiation modules referenced by the driver
# (only entered when SF_URBAN_PHYSICS>0, which we never set).
cat > _modstubs.F90 <<'EOF'
module module_sf_urban
  real :: oasis = 1.0
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
gfortran ${FFLAGS} -c noahclassic_offline_driver.F90 -o noahclassic_offline_driver.o

gfortran noahclassic_offline_driver.o _wrfstubs.o _modstubs.o \
  "${PHYS}/module_sf_noahlsm.o" "${PHYS}/module_sf_noahdrv.o" \
  "${PHYS}/module_sf_noahlsm_glacial_only.o" \
  -o noahclassic_offline_driver.exe

echo "BUILT noahclassic_offline_driver.exe"
