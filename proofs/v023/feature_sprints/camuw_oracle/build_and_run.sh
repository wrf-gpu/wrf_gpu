#!/usr/bin/env bash
# Build and run the F3 CAM-UW standalone WRF-Fortran oracle on CPU only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
PHYS="${WRF}/phys"
SHARE="${WRF}/share"
FRAME="${WRF}/frame"
ORACLE_BUILD_DIR="${HERE}/build"
PROJECT_PYTHON="$(command -v python3)"

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2

set +u
# shellcheck disable=SC1091
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u

rm -rf "${ORACLE_BUILD_DIR}"
mkdir -p "${ORACLE_BUILD_DIR}"

FFLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2 -I${PHYS} -I${SHARE} -I${FRAME}"

cd "${ORACLE_BUILD_DIR}"
gfortran ${FFLAGS} -c "${HERE}/wrf_runtime_stubs.f90" -o wrf_runtime_stubs.o
gfortran ${FFLAGS} -c "${HERE}/camuw_oracle_driver.f90" -o camuw_oracle_driver.o

gfortran camuw_oracle_driver.o wrf_runtime_stubs.o \
  "${PHYS}/module_bl_camuwpbl_driver.o" \
  "${PHYS}/module_cam_bl_eddy_diff.o" \
  "${PHYS}/module_cam_bl_diffusion_solver.o" \
  "${PHYS}/module_cam_constituents.o" \
  "${PHYS}/module_cam_support.o" \
  "${PHYS}/module_cam_molec_diff.o" \
  "${PHYS}/module_cam_trb_mtn_stress.o" \
  "${PHYS}/module_cam_wv_saturation.o" \
  "${PHYS}/module_cam_physconst.o" \
  "${PHYS}/module_cam_shr_const_mod.o" \
  "${PHYS}/module_cam_shr_kind_mod.o" \
  "${PHYS}/module_cam_esinti.o" \
  "${PHYS}/module_cam_gffgch.o" \
  "${PHYS}/module_cam_upper_bc.o" \
  "${SHARE}/module_model_constants.o" \
  "${FRAME}/module_state_description.o" \
  -lmvec -lm -o camuw_oracle.exe

{
  echo "schema=gpuwrf.v023.camuw_oracle_build_manifest.v1"
  echo "compiler=$(gfortran --version | head -n 1)"
  echo "fflags=${FFLAGS}"
  echo "wrf_root=${WRF}"
  echo "cpu_affinity=$(taskset -cp $$ 2>/dev/null || true)"
  echo "jax_platforms=${JAX_PLATFORMS}"
  echo "jax_platform_name=${JAX_PLATFORM_NAME}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "objects=prebuilt pristine-WRF objects, default WRF REAL ABI"
  sha256sum \
    "${PHYS}/module_bl_camuwpbl_driver.F" \
    "${PHYS}/module_cam_bl_eddy_diff.F" \
    "${PHYS}/module_cam_bl_diffusion_solver.F"
} > "${HERE}/camuw_build_manifest.txt"

for case_id in 1 2 3; do
  ./camuw_oracle.exe "${case_id}" > "${HERE}/camuw_case_${case_id}.txt"
  "${PROJECT_PYTHON}" "${HERE}/dump_to_json.py" "${HERE}/camuw_case_${case_id}.txt" "${HERE}/camuw_case_${case_id}.json"
done

cd "${ROOT}"
PYTHONPATH=src "${PROJECT_PYTHON}" "${HERE}/compare_current_jax.py"
