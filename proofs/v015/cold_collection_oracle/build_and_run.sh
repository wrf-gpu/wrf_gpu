#!/bin/bash
# Build + run the standalone COLD MIXED-PHASE single-column Thompson oracle.
# Drives the REAL WRF mp_gt_driver on a sub-freezing rain+snow+graupel column
# (activates qr_acr_qs / qr_acr_qg / freezeH2O cold-collection lanes) and dumps
# the bit-exact pre/post state to
#   /mnt/data/wrf_gpu2/physics_oracle/microphysics_coldmix/
#
# CPU-LIGHT: single column block, single 18 s step, serial, <1 s on cores 0-3.
# No GPU. Prereq: pristine WRF built with the oracle instrumentation + Thompson
# lookup tables (qr_acr_qsV2.dat, qr_acr_qg_V4.dat, freezeH2O.dat) available.
set -e
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild 2>/dev/null || true

WRF=/home/user/src/wrf_pristine/WRF
HERE=/home/user/src/wrf_pristine/coldmix_oracle   # build dir (outside git)
SRC="$(dirname "$0")/coldmix_column_oracle.F"
mkdir -p "$HERE"; cp "$SRC" "$HERE/"; cd "$HERE"

FLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I$WRF/phys -I$WRF/frame -I$WRF/share -I$WRF/main -I$WRF/inc"

taskset -c 0-3 gfortran -c $FLAGS $INC coldmix_column_oracle.F -o coldmix_column_oracle.o
taskset -c 0-3 gfortran $FLAGS coldmix_column_oracle.o \
  "$WRF/main/libwrflib.a" \
  "$WRF/external/fftpack/fftpack5/libfftpack.a" \
  "$WRF/external/io_grib1/libio_grib1.a" \
  "$WRF/external/io_grib_share/libio_grib_share.a" \
  "$WRF/external/io_int/libwrfio_int.a" \
  -L"$WRF/external/esmf_time_f90" -lesmf_time \
  "$WRF/frame/module_internal_header_util.o" \
  "$WRF/frame/pack_utils.o" \
  -L"$WRF/external/io_netcdf" -lwrfio_nf \
  -L"$CONDA_PREFIX/lib" -lnetcdff -lnetcdf \
  -o coldmix_column_oracle.exe

# Thompson lookup tables (link from the pristine WRF oracle_run dir).
for f in qr_acr_qg_V4.dat qr_acr_qsV2.dat freezeH2O.dat CCN_ACTIVATE.BIN; do
  for src in "$WRF/test/em_real/oracle_run/$f" "$WRF/run/$f" /mnt/data/canairy_meteo/runs/wrf_l2/*/$f; do
    [ -e "$src" ] && ln -sf "$src" . && break
  done
done

export WRFGPU2_ORACLE_ROOT=/mnt/data/wrf_gpu2/physics_oracle
rm -rf /mnt/data/wrf_gpu2/physics_oracle/microphysics_coldmix
taskset -c 0-3 ./coldmix_column_oracle.exe
echo "=== oracle written to /mnt/data/wrf_gpu2/physics_oracle/microphysics_coldmix ==="
