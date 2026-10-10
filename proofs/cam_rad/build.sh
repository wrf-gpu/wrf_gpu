#!/bin/bash
# CAM01: build + run the pristine CAM radiation true-caller oracle (lane o1-camrad cores 28,29; QUIET-guarded).
#   CAM01_DIR (default: the o1-camrad lane dir) receives objects, the run dir with read-only table links and the outputs.
#   Regenerate:  python3 extract.py cam01_input.bin cam01_columns.json && bash build.sh
#                && (cd run && ../cam01_driver.exe ../cam01_input.bin ../cam01_output.bin ../cam01_lwint.bin ../cam01_swint.bin)
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
D=${CAM01_DIR:-<USER_HOME>/wrf_gpu2_lanes/o1-camrad/CAM01}; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"; [ "$SRC" = "$D" ] || cp "$SRC/cam01_driver.F90" "$D/"
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
taskset -c 28,29 nice -n 19 $FC -c $FLAGS $INC cam01_driver.F90 -o cam01_driver.o
taskset -c 28,29 nice -n 19 $FC $FLAGS cam01_driver.o "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib" -o cam01_driver.exe
mkdir -p run; cd run
for f in CAM_ABS_DATA CAM_AEROPT_DATA CAMtr_volume_mixing_ratio ozone.formatted ozone_lat.formatted ozone_plev.formatted; do
  ln -sf "$W/run/$f" .
done
echo BUILD OK
