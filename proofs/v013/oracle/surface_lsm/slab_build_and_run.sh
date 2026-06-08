#!/usr/bin/env bash
# Build the v0.13 Tier-3 slab LSM single-column oracle against the UNMODIFIED
# WRF phys/module_sf_slab.F and emit fp64 gold savepoints as JSON.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

build_one() {
  local mode="$1"
  local extra_flags="$2"
  local out_save="$3"
  local build_dir="${HERE}/build_slab_${mode}"

  rm -rf "${build_dir}"
  mkdir -p "${build_dir}" "${out_save}"

  cp "${WRF_PHYS}/module_sf_slab.F" "${build_dir}/module_sf_slab.F"
  cp "${HERE}/slab_oracle_driver.f90" "${build_dir}/slab_oracle_driver.f90"

  ( cd "${build_dir}" && sha256sum module_sf_slab.F \
      > "${out_save}/slab_wrf_source_checksums.txt" )

  cd "${build_dir}"
  local fflags="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none ${extra_flags}"
  taskset -c 0-3 gfortran ${fflags} -c module_sf_slab.F
  taskset -c 0-3 gfortran ${fflags} -c slab_oracle_driver.f90
  taskset -c 0-3 gfortran ${fflags} -o slab_oracle \
      module_sf_slab.o slab_oracle_driver.o
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_source=${WRF_PHYS}/module_sf_slab.F"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags}"
  } > "${out_save}/slab_build_manifest.txt"

  for c in 1 2 3 4 5; do
    taskset -c 0-3 ./slab_oracle "$c" "${mode}" > "case_${c}.txt"
    python3 "${HERE}/slab_dump_to_json.py" "case_${c}.txt" "${out_save}/slab_case_${c}.json"
  done
}

build_one fp64 "-fdefault-real-8 -fdefault-double-8" "${HERE}/../../savepoints/surface_lsm/fp64"
echo "OK: slab LSM oracle built and fp64 savepoints written under ${HERE}/../../savepoints/surface_lsm/fp64"
