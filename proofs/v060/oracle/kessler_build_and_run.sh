#!/usr/bin/env bash
# Build the v0.6.0 Kessler single-column oracle against the UNMODIFIED WRF
# phys/module_mp_kessler.F source and emit gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. The canonical mode is fp32, matching WRF's
# default REAL*4 build. An fp64 audit mode compiles the same unmodified source
# with -fdefault-real-8 to distinguish reference precision dust from port error.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_PHYS="/home/enric/src/wrf_pristine/WRF/phys"
MODE="${1:-all}"

if [ -f /home/enric/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/enric/miniconda3/etc/profile.d/conda.sh
  conda activate wrfbuild
fi

export OMP_NUM_THREADS=2

build_one() {
  local mode="$1"
  local out_save="${HERE}/../savepoints"
  local blddir="${HERE}/build_${mode}"
  local extra_flags=""
  if [ "${mode}" = "fp64" ]; then
    out_save="${HERE}/../savepoints_fp64"
    extra_flags="-fdefault-real-8"
  fi

  rm -rf "${blddir}"
  mkdir -p "${blddir}" "${out_save}"
  cp "${WRF_PHYS}/module_mp_kessler.F" "${blddir}/module_mp_kessler.F"
  cp "${HERE}/kessler_oracle_driver.f90" "${blddir}/kessler_oracle_driver.f90"

  ( cd "${blddir}" && sha256sum module_mp_kessler.F > "${out_save}/wrf_source_checksums.txt" )
  local manifest_fflags="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none"
  if [ "${mode}" = "fp64" ]; then
    manifest_fflags="${manifest_fflags} -fdefault-real-8"
  fi
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_source=/home/enric/src/wrf_pristine/WRF/phys/module_mp_kessler.F"
    echo "compiler=$(gfortran --version | head -1)"
    echo "fflags=${manifest_fflags}"
  } > "${out_save}/build_manifest.txt"

  cd "${blddir}"
  local fflags="-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none ${extra_flags}"
  taskset -c 0-3 gfortran ${fflags} -c module_mp_kessler.F
  taskset -c 0-3 gfortran ${fflags} -c kessler_oracle_driver.f90
  taskset -c 0-3 gfortran ${fflags} -o kessler_oracle module_mp_kessler.o kessler_oracle_driver.o

  for c in 1 2 3 4 5; do
    taskset -c 0-3 ./kessler_oracle "$c" > "case_${c}.txt"
    python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${out_save}/kessler_case_${c}.json"
  done
  echo "OK: Kessler oracle mode=${mode} savepoints written to ${out_save}"
}

case "${MODE}" in
  fp32) build_one fp32 ;;
  fp64) build_one fp64 ;;
  all) build_one fp32; build_one fp64 ;;
  *) echo "usage: $0 [fp32|fp64|all]" >&2; exit 2 ;;
esac
