#!/usr/bin/env bash
# Build the v0.13 MRF single-column PBL oracle (bl_pbl_physics=99) against the
# UNMODIFIED pristine WRF phys/module_bl_mrf.F and emit gold savepoints as JSON.
# Builds BOTH fp32 (default WRF REAL) and fp64 (-fdefault-real-8). The fp64
# savepoints are the v0.13 operational-parity gate target (~1e-13).
#
# CPU-only, pinned to cores 0-3. Requires the conda env `wrfbuild`.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="/home/enric/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
SAVE_FP32="${HERE}/../../savepoints/mrf"
SAVE_FP64="${HERE}/../../savepoints_fp64/mrf"

source /home/enric/miniconda3/etc/profile.d/conda.sh
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

  cp "${WRF_PHYS}/module_bl_mrf.F" "${build_dir}/module_bl_mrf.F"
  cp "${HERE}/mrf_oracle_driver.f90" "${build_dir}/mrf_oracle_driver.f90"

  ( cd "${build_dir}" && sha256sum module_bl_mrf.F > "${out_save}/mrf_wrf_source_checksums.txt" )

  cd "${build_dir}"
  local fflags="-O2 -ffree-form -ffree-line-length-none -cpp -ffpe-summary=none ${extra_flags}"
  taskset -c 0-3 gfortran ${fflags} -c module_bl_mrf.F
  taskset -c 0-3 gfortran ${fflags} -c mrf_oracle_driver.f90
  taskset -c 0-3 gfortran ${fflags} -o mrf_oracle module_bl_mrf.o mrf_oracle_driver.o
  {
    echo "mode=${mode}"
    echo "full_wrf_exe=false"
    echo "wrf_source=${WRF_PHYS}/module_bl_mrf.F"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=${fflags}"
  } > "${out_save}/mrf_build_manifest.txt"

  for c in 1 2 3 4 5 6; do
    taskset -c 0-3 ./mrf_oracle "$c" > "case_${c}.txt"
    python3 "${HERE}/dump_to_json.py" "case_${c}.txt" "${out_save}/mrf_case_${c}.json"
  done
}

build_one fp32 "" "${SAVE_FP32}"
build_one fp64 "-fdefault-real-8 -fdefault-double-8" "${SAVE_FP64}"
echo "OK: MRF oracle built; fp32 -> ${SAVE_FP32}, fp64 -> ${SAVE_FP64}"
