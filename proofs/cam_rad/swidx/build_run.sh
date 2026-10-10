#!/bin/bash
# SWIDX: build the private pristine-WRF CAM SW driver and run it in mode 0 (P_* = 1, CAM01 state) and mode 1
# (P_SUL..P_VOLC = 2..13, real WRF).  Core 28 only, QUIET-guarded, sequential.
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
set -euo pipefail
D=<USER_HOME>/wrf_gpu2_lanes/o1-camrad/SWIDX; W=<USER_HOME>/src/wrf_pristine/WRF
ENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build; FC=$ENV/bin/x86_64-conda-linux-gnu-gfortran
REPO=<USER_HOME>/src/wrf_gpu2_wt/o1-camrad
FX=<USER_HOME>/wrf_gpu2_lanes/o1-camrad/CAM01/cam01_fixture.npz
cd "$D"
FLAGS="-w -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$W/phys -I$W/frame -I$W/share -I$W/main -I$W/inc"
T="taskset -c 28 nice -n 19"
$T $FC -c $FLAGS $INC swidx_driver.F90 -o swidx_driver.o
$T $FC $FLAGS swidx_driver.o "$W/main/libwrflib.a" \
  "$W/external/fftpack/fftpack5/libfftpack.a" "$W/external/io_grib1/libio_grib1.a" "$W/external/io_grib_share/libio_grib_share.a" \
  "$W/external/io_int/libwrfio_int.a" -L"$W/external/esmf_time_f90" -lesmf_time \
  "$W/frame/module_internal_header_util.o" "$W/frame/pack_utils.o" -L"$W/external/io_netcdf" -lwrfio_nf \
  -L"$ENV/lib" -lnetcdff -lnetcdf -Wl,-rpath,"$ENV/lib" -o swidx_driver.exe
mkdir -p run
for f in CAM_ABS_DATA CAM_AEROPT_DATA CAMtr_volume_mixing_ratio ozone.formatted ozone_lat.formatted ozone_plev.formatted; do
  ln -sf "$W/run/$f" run/
done
for mode in 0 1; do
  (cd $REPO && JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src $T python $D/gen_input.py "$FX" $D/swidx_in_m$mode.bin $mode $D/swidx_cols_m$mode.json)
  (cd run && $T ../swidx_driver.exe ../swidx_in_m$mode.bin ../swidx_out_m$mode.bin > ../run_m$mode.log 2>&1)
  tail -2 run_m$mode.log
  $T python parse_output.py swidx_in_m$mode.bin swidx_out_m$mode.bin swidx_cols_m$mode.json swidx_fixture_m$mode.npz
done
sha256sum swidx_fixture_m*.npz
