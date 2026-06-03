#!/usr/bin/env bash
# Build + run the WRF MYNN-EDMF column oracle against the pristine WRF objects.
# CPU-only. Requires the wrfbuild conda gfortran.
set -e
cd "$(dirname "$0")"
python ../emit_flat.py
PHYS=/home/enric/src/wrf_pristine/WRF/phys
SHARE=/home/enric/src/wrf_pristine/WRF/share
GF=/home/enric/miniconda3/envs/wrfbuild/bin/gfortran
$GF -I$PHYS -ffree-line-length-none -fbacktrace -g -c oracle.f90 -o oracle.o
$GF oracle.o $PHYS/module_bl_mynnedmf.o $PHYS/module_bl_mynnedmf_common.o \
    $SHARE/module_model_constants.o $PHYS/ccpp_kind_types.o -o oracle
taskset -c 0-3 ./oracle
echo "oracle_out.txt written."
