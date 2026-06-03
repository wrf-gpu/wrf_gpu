#!/usr/bin/env bash
# Build the v0.6.0 MYJ PBL single-column oracle against the UNMODIFIED WRF
# module_bl_myjpbl.F + module_sf_myjsfc.F + share/module_model_constants.F and
# emit gold savepoints as JSON. The driver runs the PAIRED Janjic surface layer
# (sf_sfclay=2) before MYJ PBL (bl_pbl=2), as WRF requires.
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_SHARE="${WRF}/share"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

build_one() {
  local mode="$1"
  local extra_flags="$2"
  local out_save="$3"
  local build_dir="${HERE}/build_myjpbl_${mode}"

  rm -rf "${build_dir}"
  mkdir -p "${build_dir}" "${out_save}"

  cp "${WRF_SHARE}/module_model_constants.F" "${build_dir}/module_model_constants.F"
  cp "${WRF_PHYS}/module_sf_myjsfc.F" "${build_dir}/module_sf_myjsfc.F"
  cp "${WRF_PHYS}/module_bl_myjpbl.F" "${build_dir}/module_bl_myjpbl.F"
  cp "${HERE}/myjpbl_oracle_driver.f90" "${build_dir}/myjpbl_oracle_driver.f90"

  ( cd "${build_dir}" && sha256sum module_model_constants.F module_sf_myjsfc.F module_bl_myjpbl.F \
      > "${out_save}/myjpbl_wrf_source_checksums.txt" )

  cd "${build_dir}"
  local fflags="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none ${extra_flags}"
  taskset -c 0-3 gfortran ${fflags} -c module_model_constants.F
  taskset -c 0-3 gfortran ${fflags} -c module_sf_myjsfc.F
  taskset -c 0-3 gfortran ${fflags} -c module_bl_myjpbl.F
  taskset -c 0-3 gfortran ${fflags} -c myjpbl_oracle_driver.f90
  taskset -c 0-3 gfortran ${fflags} -o myjpbl_oracle \
      module_model_constants.o module_sf_myjsfc.o module_bl_myjpbl.o myjpbl_oracle_driver.o
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_source=${WRF_SHARE}/module_model_constants.F"
    echo "wrf_source=${WRF_PHYS}/module_sf_myjsfc.F"
    echo "wrf_source=${WRF_PHYS}/module_bl_myjpbl.F"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags}"
  } > "${out_save}/myjpbl_build_manifest.txt"

  for c in 1 2 3 4 5 6; do
    taskset -c 0-3 ./myjpbl_oracle "$c" > "case_${c}.txt"
    python3 "${HERE}/myjpbl_dump_to_json.py" "case_${c}.txt" "${out_save}/myjpbl_case_${c}.json"
  done
}

build_one fp32 "" "${HERE}/../savepoints"
build_one fp64 "-fdefault-real-8 -fdefault-double-8" "${HERE}/../savepoints_fp64"
echo "OK: MYJ PBL oracle built and savepoints written under ${HERE}/.."
