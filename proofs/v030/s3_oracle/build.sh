#!/usr/bin/env bash
# Build the S3 interp oracle shared library from the REAL WPS interp_module.F
# plus a stub module_debug (the real one needs MPI/parallel_module). Produces
# liboracle.so for ctypes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WPS_SRC="/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/src"
FC="/home/enric/miniconda3/envs/wrfbuild/bin/gfortran"

cd "$HERE"
rm -f ./*.o ./*.mod liboracle.so

# Compile order respects module deps:
#   misc_definitions -> (nothing)
#   module_debug (stub) -> (nothing)
#   bitarray -> module_debug
#   queue -> module_debug
#   interp_module -> bitarray, misc_definitions, module_debug, queue
#   oracle_driver -> interp_module
$FC -O2 -fPIC -ffree-form -ffree-line-length-none -cpp -D_METGRID -c "$WPS_SRC/misc_definitions_module.F" -o misc_definitions_module.o
$FC -O2 -fPIC -ffree-form -ffree-line-length-none -cpp -D_METGRID -c module_debug_stub.F -o module_debug.o
$FC -O2 -fPIC -ffree-form -ffree-line-length-none -cpp -D_METGRID -c "$WPS_SRC/bitarray_module.F" -o bitarray_module.o
$FC -O2 -fPIC -ffree-form -ffree-line-length-none -cpp -D_METGRID -c "$WPS_SRC/queue_module.F" -o queue_module.o
$FC -O2 -fPIC -ffree-form -ffree-line-length-none -cpp -D_METGRID -c "$WPS_SRC/interp_module.F" -o interp_module.o
$FC -O2 -fPIC -ffree-form -ffree-line-length-none -cpp -D_METGRID -c oracle_driver.F -o oracle_driver.o

$FC -shared -fPIC -o liboracle.so \
    misc_definitions_module.o module_debug.o bitarray_module.o \
    queue_module.o interp_module.o oracle_driver.o

echo "built $HERE/liboracle.so"
