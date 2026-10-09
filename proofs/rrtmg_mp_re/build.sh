#!/bin/bash
# RE01: build + run the pristine RRTMG true-caller oracle (TH08 recipe; lane cores 24,25,28,29; QUIET-guarded).
#   RE01_DIR (default: the fid-q2 lane dir) receives the objects, the run dir with read-only table links and the outputs.
#   Regenerate:  cd $RE01_DIR && python3 census.py && python3 extract.py re01_input.bin re01_columns.json
#                && bash build.sh && (cd run && ../re01_driver.exe ../re01_input.bin ../re01_output.bin)
#                && python3 parse.py re01_input.bin re01_output.bin re01_columns.json re01_fixture.npz re01_summary.json
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
D=${RE01_DIR:-<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE01}; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
mkdir -p "$D"; [ "$SRC" = "$D" ] || cp "$SRC/re01_driver.F90" "$D/"
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
taskset -c 24,25,28,29 nice -n 19 $FC -c $FLAGS $INC re01_driver.F90 -o re01_driver.o
taskset -c 24,25,28,29 nice -n 19 $FC $FLAGS re01_driver.o "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib" -o re01_driver.exe
mkdir -p run; cd run
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN RRTMG_LW_DATA RRTMG_SW_DATA CAMtr_volume_mixing_ratio \
         ozone.formatted ozone_lat.formatted ozone_plev.formatted; do
  for src in "$W/test/em_real/oracle_run_v090/$f" "$W/test/em_real/oracle_run/$f" "$W/run/$f"; do
    if [ -e "$src" ]; then ln -sf "$src" .; break; fi
  done
done
echo BUILD OK
