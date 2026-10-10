#!/usr/bin/env bash
# Build the v0.3.4 Grell-family tile oracles from UNMODIFIED pristine WRF sources.
# Usage: build.sh OUTDIR   -> OUTDIR/{g3,gd}_{r8,r4}.exe
# Sources are copied (never edited) from $WRF_PRISTINE_ROOT/phys; the build runs in
# OUTDIR (a lane directory), never inside the pristine tree (FINDINGS E104).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: build.sh OUTDIR}"
WRF_ROOT="${WRF_PRISTINE_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}"
FC="${FC:-<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran}"
mkdir -p "$OUT"
cd "$OUT"
cp "$WRF_ROOT/phys/module_cu_g3.F" "$WRF_ROOT/phys/module_cu_gd.F" .
cp "$HERE/grell_tile_oracle.F90" .
sha256sum module_cu_g3.F module_cu_gd.F grell_tile_oracle.F90 > sources.sha256
BASE="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none -fallow-argument-mismatch -std=legacy"
for prec in r8 r4; do
  if [ "$prec" = r8 ]; then P="-fdefault-real-8 -fdefault-double-8"; else P=""; fi
  for s in g3 gd; do
    d="build_${s}_${prec}"; mkdir -p "$d"
    S=$(echo "$s" | tr a-z A-Z)
    (cd "$d" && "$FC" $BASE $P -c "../module_cu_${s}.F" -o module_cu_${s}.o \
       && "$FC" $BASE $P -DSCHEME_${S} -c ../grell_tile_oracle.F90 -o drv.o \
       && "$FC" -O2 -o "../${s}_${prec}.exe" drv.o module_cu_${s}.o)
  done
done
ls -la "$OUT"/*.exe
