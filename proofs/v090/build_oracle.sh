#!/usr/bin/env bash
# Build the MYNN-SL Fortran oracle from the BYTE-IDENTICAL pristine
# module_sf_mynn.F (sha256 in oracle_source_sha256.txt) plus a 4-constant shim
# (p1000mb, r_d, r_v, ep_2 -- the only module_model_constants symbols MYNN uses)
# and a thin driver that calls the UNMODIFIED SFCLAY1D_mynn.
#
# Default build = REAL*4 (single precision) -- this is how operational WRF runs,
# so it is the faithful numerical oracle. Pass DOUBLE=1 to build a REAL*8 variant
# (used only to separate fp32-roundoff from algorithm divergence).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GF="${GF:-/home/enric/miniconda3/envs/wrfbuild/bin/gfortran}"
cd "$HERE"
FLAGS="-ffree-line-length-none -O2"
OUT=mynn_oracle
if [ "${DOUBLE:-0}" = "1" ]; then
  FLAGS="$FLAGS -fdefault-real-8 -fdefault-double-8"
  OUT=mynn_oracle_r8
fi
rm -f *.o *.mod "$OUT"
taskset -c 0-3 "$GF" -c $FLAGS module_model_constants.f90 -o mmc.o
taskset -c 0-3 "$GF" -c $FLAGS module_sf_mynn_pristine.f90 -o msm.o
taskset -c 0-3 "$GF" $FLAGS mynn_oracle_driver.f90 mmc.o msm.o -o "$OUT"
echo "built $OUT"
