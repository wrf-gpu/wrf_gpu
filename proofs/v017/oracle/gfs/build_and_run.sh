#!/usr/bin/env bash
# Build the v0.17 GFS single-column PBL oracle (bl_pbl_physics=3) against the
# UNMODIFIED pristine WRF phys/module_bl_gfs.F (+ its module_gfs_machine.F /
# module_gfs_physcons.F support modules) and emit gold savepoints as JSON.
# Builds BOTH fp32 (default WRF REAL) and fp64 (-fdefault-real-8). The GFS PBL
# internals use kind_phys=selected_real_kind(13,60) (real*8) regardless of the
# default REAL, so the v0.17 operational-parity gate target is the fp64
# savepoints (~1e-12, kind_phys-native).
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
SAVE_FP32="${HERE}/../../savepoints/gfs"
SAVE_FP64="${HERE}/../../savepoints_fp64/gfs"

source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=2
set -e

SRC=( module_gfs_machine.F module_gfs_physcons.F module_bl_gfs.F )

build_one() {
  local mode="$1"
  local extra_flags="$2"
  local out_save="$3"
  local build_dir="${HERE}/build_${mode}"

  rm -rf "${build_dir}"
  mkdir -p "${build_dir}" "${out_save}"

  local s
  for s in "${SRC[@]}"; do
    cp "${WRF_PHYS}/${s}" "${build_dir}/${s}"
  done
  cp "${HERE}/gfs_oracle_driver.f90" "${build_dir}/gfs_oracle_driver.f90"

  ( cd "${build_dir}" && sha256sum "${SRC[@]}" > "${out_save}/gfs_wrf_source_checksums.txt" )

  cd "${build_dir}"
  # -DEM_CORE=1 selects the WRF EM (Advanced Research WRF) core path, which
  # disables the NMM_CORE/HWRF ensemble-perturbation/dissipative-heating blocks.
  local fflags="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0 -DHWRF=0 -ffpe-summary=none ${extra_flags}"
  taskset -c 0-3 gfortran ${fflags} -c module_gfs_machine.F
  taskset -c 0-3 gfortran ${fflags} -c module_gfs_physcons.F
  taskset -c 0-3 gfortran ${fflags} -c module_bl_gfs.F
  taskset -c 0-3 gfortran ${fflags} -c gfs_oracle_driver.f90
  taskset -c 0-3 gfortran ${fflags} -o gfs_oracle \
      module_gfs_machine.o module_gfs_physcons.o module_bl_gfs.o gfs_oracle_driver.o
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_sources=${WRF_PHYS}/{module_gfs_machine.F,module_gfs_physcons.F,module_bl_gfs.F}"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags}"
  } > "${out_save}/gfs_build_manifest.txt"

  for c in 1 2 3 4 5 6; do
    taskset -c 0-3 ./gfs_oracle "$c" > "case_${c}.txt"
    python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${out_save}/gfs_case_${c}.json"
  done
}

build_one fp32 "" "${SAVE_FP32}"
build_one fp64 "-fdefault-real-8 -fdefault-double-8" "${SAVE_FP64}"
echo "OK: GFS oracle built; fp32 -> ${SAVE_FP32}, fp64 -> ${SAVE_FP64}"
