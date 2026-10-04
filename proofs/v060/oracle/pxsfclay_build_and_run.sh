#!/usr/bin/env bash
# Build the v0.6.0 Pleim-Xiu surface-layer oracle against UNMODIFIED WRF
# module_sf_pxsfclay.F and emit JSON savepoints.
#
# CPU-only, pinned to ORACLE_CPUS (default: wrf_gpu2 lane cores; never ALISIOS cores 0-7/16-23). Requires the conda env `wrfbuild`.
set -o pipefail
ORACLE_CPUS="${ORACLE_CPUS:-8,9,12,13,24,25,28,29}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

build_one() {
  local mode="$1"
  local extra_flags="$2"
  local out_save="$3"
  local build_dir="${HERE}/build_${mode}"
  rm -rf "${build_dir}"
  mkdir -p "${build_dir}" "${out_save}"

  cp "${WRF_PHYS}/module_sf_pxsfclay.F" "${build_dir}/module_sf_pxsfclay.F"
  cp "${HERE}/pxsfclay_oracle_driver.f90" "${build_dir}/pxsfclay_oracle_driver.f90"

  ( cd "${build_dir}" && sha256sum module_sf_pxsfclay.F > "${out_save}/pxsfclay_wrf_source_checksums.txt" )

  cd "${build_dir}"
  local fflags="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none ${extra_flags}"
  taskset -c ${ORACLE_CPUS} nice -n 19 gfortran ${fflags} -c module_sf_pxsfclay.F
  taskset -c ${ORACLE_CPUS} nice -n 19 gfortran ${fflags} -c pxsfclay_oracle_driver.f90
  taskset -c ${ORACLE_CPUS} nice -n 19 gfortran ${fflags} -o pxsfclay_oracle module_sf_pxsfclay.o pxsfclay_oracle_driver.o
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_source=<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_pxsfclay.F"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags}"
  } > "${out_save}/pxsfclay_build_manifest.txt"

  for c in 1 2 3 4 5 6; do
    taskset -c ${ORACLE_CPUS} nice -n 19 ./pxsfclay_oracle "$c" "${mode}" > "case_${c}.txt"
    python3 "${HERE}/pxsfclay_dump_to_json.py" "case_${c}.txt" "${out_save}/pxsfclay_case_${c}.json"
  done
}

build_one fp32 "" "${HERE}/../savepoints"
build_one fp64 "-fdefault-real-8 -fdefault-double-8" "${HERE}/../savepoints_fp64"
echo "OK: Pleim-Xiu surface-layer oracle built and savepoints written under ${HERE}/.."
