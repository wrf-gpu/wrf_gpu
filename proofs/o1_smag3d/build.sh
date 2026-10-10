#!/bin/bash
# o1-smag3d: build the pristine-WRF diff_opt=2/km_opt=3 oracle driver against libwrflib.a (RE01 recipe).
#   OUT (default <USER_HOME>/wrf_gpu2_lanes/o1-smag3d/oracle) receives smag3d_driver.exe. Cores 28,29. QUIET-guarded.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
D=${OUT:-<USER_HOME>/wrf_gpu2_lanes/o1-smag3d/oracle}; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"; cp "$SRC/smag3d_driver.F90" "$D/"; cd "$D"
FLAGS="-w -ffree-line-length-none -O2"  # native-endian stream I/O (no WRF data files read)
INC="-I$W/dyn_em -I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
taskset -c 29 nice -n 19 $FC -c $FLAGS $INC smag3d_driver.F90 -o smag3d_driver.o
taskset -c 29 nice -n 19 $FC $FLAGS smag3d_driver.o "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib" -o smag3d_driver.exe
echo BUILD OK "$D/smag3d_driver.exe"
