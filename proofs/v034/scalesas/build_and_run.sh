#!/usr/bin/env bash
# Build the CU_SCALESAS oracle against copied pristine WRF 4.7.1 modules (unmodified) in two
# precisions and run it on generated inputs. CPU only. Usage: build_and_run.sh <workdir> <input.txt>...
#   r4 = WRF build semantics (RWORDSIZE=4: default REAL and un-suffixed literals are float32;
#        kind_phys=8 internals, exactly like configure.wrf of the pristine build)
#   r8 = -fdefault-real-8 -fdefault-double-8 (the v0.17 oracle semantics; literals exact in double)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$1"; shift
WRF_PHYS="${WRF_PRISTINE_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}/phys"
export PATH="<USER_HOME>/miniconda3/envs/wrfbuild/bin:${PATH}"
ulimit -c 0
mkdir -p "${WORK}"
for prec in r4 r8; do
  B="${WORK}/build_${prec}"; mkdir -p "$B"; cd "$B"
  for m in module_gfs_machine module_gfs_physcons module_gfs_funcphys module_cu_scalesas; do
    cp "${WRF_PHYS}/${m}.F" .
  done
  cp "${HERE}/oracle_driver.F90" .
  BASE="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0"
  if [ "$prec" = r8 ]; then FL="$BASE -fdefault-real-8 -fdefault-double-8"; else FL="$BASE"; fi
  for m in module_gfs_machine module_gfs_physcons module_gfs_funcphys module_cu_scalesas; do
    gfortran $FL -c ${m}.F
  done
  gfortran $FL -c oracle_driver.F90
  gfortran -O2 -o scalesas_oracle module_gfs_machine.o module_gfs_physcons.o module_gfs_funcphys.o \
      module_cu_scalesas.o oracle_driver.o
  sha256sum ${WRF_PHYS}/module_cu_scalesas.F module_cu_scalesas.F > sources.sha256
  for inp in "$@"; do
    stem="$(basename "$inp" .txt)"
    ./scalesas_oracle < "$inp" > "${WORK}/${stem}_${prec}.out"
  done
done
echo "OK oracle outputs in ${WORK}"
