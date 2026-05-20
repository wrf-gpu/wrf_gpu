#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_ROOT="/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_thompson_harness"
OBJ="${SCRATCH}/wrf_thompson_harness.o"
LOG="${SCRATCH}/wrf_thompson_harness_build.log"

mkdir -p "${SCRATCH}"
: > "${LOG}"

if [[ -f /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh >>"${LOG}" 2>&1 || true
fi

if command -v gfortran >/dev/null 2>&1; then
  echo "gfortran is present, but the WRF .mod/.o files are NVHPC-built; using nvfortran for ABI compatibility." >>"${LOG}"
fi

FC="${FC:-$(command -v nvfortran || true)}"
if [[ -z "${FC}" ]]; then
  echo "nvfortran not found; cannot link NVHPC WRF objects" | tee -a "${LOG}" >&2
  exit 1
fi

MOD_DIR="${WRF_ROOT}/main"
if [[ ! -f "${MOD_DIR}/module_mp_thompson.mod" ]]; then
  MOD_DIR="${WRF_ROOT}/install_gen2_dmpar/modules"
fi

"${FC}" -c -I"${MOD_DIR}" -I"${WRF_ROOT}/main" -I"${WRF_ROOT}/install_gen2_dmpar/modules" \
  -I"${WRF_ROOT}/external/esmf_time_f90" -I"${WRF_ROOT}/install_gen2_dmpar/esmf_time_f90" \
  -o "${OBJ}" "${ROOT}/scripts/wrf_thompson_harness.f90" >>"${LOG}" 2>&1

"${FC}" -o "${OUT}" "${OBJ}" \
  "${WRF_ROOT}/phys/module_mp_thompson.o" \
  "${WRF_ROOT}/phys/module_mp_radar.o" \
  "${WRF_ROOT}/share/module_model_constants.o" \
  "${WRF_ROOT}/frame/module_wrf_error.o" \
  >>"${LOG}" 2>&1

chmod 0755 "${OUT}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
