#!/usr/bin/env bash
# Build the v0.13 Tier-3 old-MM5 surface-layer single-column oracle against the
# UNMODIFIED WRF phys/module_sf_sfclay.F and emit fp64 gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/user/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

build_one() {
  local mode="$1"
  local extra_flags="$2"
  local out_save="$3"
  local build_dir="${HERE}/build_sfclay_old_mm5_${mode}"

  rm -rf "${build_dir}"
  mkdir -p "${build_dir}" "${out_save}"

  cp "${WRF_PHYS}/module_sf_sfclay.F" "${build_dir}/module_sf_sfclay.F"
  cp "${HERE}/sfclay_old_mm5_oracle_driver.f90" "${build_dir}/sfclay_old_mm5_oracle_driver.f90"

  ( cd "${build_dir}" && sha256sum module_sf_sfclay.F \
      > "${out_save}/sfclay_old_mm5_wrf_source_checksums.txt" )

  cd "${build_dir}"
  local fflags="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none ${extra_flags}"
  taskset -c 0-3 gfortran ${fflags} -c module_sf_sfclay.F
  taskset -c 0-3 gfortran ${fflags} -c sfclay_old_mm5_oracle_driver.f90
  taskset -c 0-3 gfortran ${fflags} -o sfclay_old_mm5_oracle \
      module_sf_sfclay.o sfclay_old_mm5_oracle_driver.o
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_source=${WRF_PHYS}/module_sf_sfclay.F"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags}"
  } > "${out_save}/sfclay_old_mm5_build_manifest.txt"

  for c in 1 2 3 4 5; do
    taskset -c 0-3 ./sfclay_old_mm5_oracle "$c" "${mode}" > "case_${c}.txt"
    python3 "${HERE}/sfclay_old_mm5_dump_to_json.py" "case_${c}.txt" "${out_save}/sfclay_old_mm5_case_${c}.json"
  done
}

build_one fp64 "-fdefault-real-8 -fdefault-double-8" "${HERE}/../../savepoints/surface_lsm/fp64"
echo "OK: old-MM5 surface-layer oracle built and fp64 savepoints written under ${HERE}/../../savepoints/surface_lsm/fp64"
