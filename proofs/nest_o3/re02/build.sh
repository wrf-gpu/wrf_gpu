#!/bin/bash
# RE02: build + run the pristine RRTMG nest-ozone sensitivity (own-column CAM vs parent copy) (TH08 recipe; lane cores 24,25,28,29; QUIET-guarded).
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
D=<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE02; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
taskset -c 24,25,28,29 nice -n 19 $FC -c $FLAGS $INC re02_o3.F90 -o re02_o3.o
taskset -c 24,25,28,29 nice -n 19 $FC $FLAGS re02_o3.o "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib" -o re02_o3.exe
mkdir -p run; cd run
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN RRTMG_LW_DATA RRTMG_SW_DATA CAMtr_volume_mixing_ratio \
         ozone.formatted ozone_lat.formatted ozone_plev.formatted; do
  for src in "$W/test/em_real/oracle_run_v090/$f" "$W/test/em_real/oracle_run/$f" "$W/run/$f"; do if [ -e "$src" ]; then ln -sf "$src" .; break; fi; done
done
ls -lL | head -20
echo BUILD OK
