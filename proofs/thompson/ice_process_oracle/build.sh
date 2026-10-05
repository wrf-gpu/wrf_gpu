#!/bin/bash
# Build TH08 driver against the pristine WRF library (lane cores, no <DATA_ROOT> writes).
set -euo pipefail
D=${TH08_BUILD_DIR:-<USER_HOME>/wrf_gpu2_lanes/b-thompson/TH08/driver}; mkdir -p "$D"; cp "$(dirname "$0")/thompson_column_driver.F90" "$D/"; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
taskset -c 24,25,28,29 nice -n 19 $FC -c $FLAGS $INC thompson_column_driver.F90 -o thompson_column_driver.o
taskset -c 24,25,28,29 nice -n 19 $FC $FLAGS thompson_column_driver.o "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib" -o thompson_column_driver.exe
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN; do ln -sf "$W/test/em_real/oracle_run/$f" .; done
echo BUILD OK
