#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_rrtmg_harness"
OBJ="${SCRATCH}/wrf_rrtmg_harness.o"
LOG="${SCRATCH}/wrf_rrtmg_harness_build.log"
RUNTIME="${SCRATCH}/rrtmg_runtime"
WRF_ROOT="/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF"
WRF_BUILD="${WRF_ROOT}/_build_gen2_dmpar"
WRF_ENV="/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build"
MOD_DIR="${WRF_ROOT}/install_gen2_dmpar/modules"
SW_OBJ="${WRF_BUILD}/CMakeFiles/WRF_Core.dir/phys/module_ra_rrtmg_sw.F.o"
LW_OBJ="${WRF_BUILD}/CMakeFiles/WRF_Core.dir/phys/module_ra_rrtmg_lw.F.o"
SW_DATA="${WRF_ROOT}/install_gen2_dmpar/run/RRTMG_SW_DATA"
LW_DATA="${WRF_ROOT}/install_gen2_dmpar/run/RRTMG_LW_DATA"

mkdir -p "${SCRATCH}" "${RUNTIME}"
ln -sf "${SW_DATA}" "${RUNTIME}/RRTMG_SW_DATA"
ln -sf "${LW_DATA}" "${RUNTIME}/RRTMG_LW_DATA"
: >"${LOG}"

if [[ -x "${OUT}" && "${OUT}" -nt "${ROOT}/scripts/wrf_rrtmg_harness.f90" ]]; then
  echo "reusing existing RRTMG harness build: ${OUT}" >>"${LOG}"
  sha256sum "${OUT}" | tee -a "${LOG}"
  echo "${OUT}"
  exit 0
fi

FC="${FC:-${WRF_ENV}/bin/gfortran}"
if [[ ! -x "${FC}" ]]; then
  FC="$(command -v gfortran || true)"
fi
if [[ -z "${FC}" ]]; then
  echo "gfortran not found; cannot compile RRTMG harness for the local GNU WRF build" | tee -a "${LOG}" >&2
  exit 1
fi

{
  echo "compiler=${FC}"
  echo "module_dir=${MOD_DIR}"
  echo "real_sw_object=${SW_OBJ}"
  echo "real_lw_object=${LW_OBJ}"
} >>"${LOG}"

if [[ ! -f "${SW_OBJ}" || ! -f "${LW_OBJ}" ]]; then
  echo "required WRF RRTMG object/library not found" | tee -a "${LOG}" >&2
  exit 1
fi

"${FC}" -I"${MOD_DIR}" -ffree-line-length-none -c -o "${OBJ}" "${ROOT}/scripts/wrf_rrtmg_harness.f90" >>"${LOG}" 2>&1

"${FC}" -o "${OUT}" "${OBJ}" "${SW_OBJ}" "${LW_OBJ}" -lmvec -lm >>"${LOG}" 2>&1

chmod 0755 "${OUT}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
