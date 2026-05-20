#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_mynn_harness"
OBJ="${SCRATCH}/wrf_mynn_harness.o"
LOG="${SCRATCH}/wrf_mynn_harness_build.log"
WRF_ROOT="/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF"
EXACT_OBJ="${WRF_ROOT}/phys/module_bl_mynn.o"
ACTUAL_SRC="${WRF_ROOT}/phys/MYNN-EDMF/misc/module_bl_mynn.F90"

mkdir -p "${SCRATCH}"
: > "${LOG}"

if [[ -x "${OUT}" && "${OUT}" -nt "${ROOT}/scripts/wrf_mynn_harness.f90" ]]; then
  echo "reusing existing one-time MYNN harness build: ${OUT}" >>"${LOG}"
  sha256sum "${OUT}" | tee -a "${LOG}"
  echo "${OUT}"
  exit 0
fi

if [[ -f /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh >>"${LOG}" 2>&1 || true
fi

FC="${FC:-$(command -v nvfortran || true)}"
if [[ -z "${FC}" ]]; then
  echo "nvfortran not found; cannot compile MYNN harness" | tee -a "${LOG}" >&2
  exit 1
fi

if [[ -f "${EXACT_OBJ}" ]]; then
  echo "exact object present: ${EXACT_OBJ}" >>"${LOG}"
else
  echo "exact contract object absent: ${EXACT_OBJ}" >>"${LOG}"
  echo "using standalone source-derived harness; source reference: ${ACTUAL_SRC}" >>"${LOG}"
fi

"${FC}" -c -o "${OBJ}" "${ROOT}/scripts/wrf_mynn_harness.f90" >>"${LOG}" 2>&1
"${FC}" -o "${OUT}" "${OBJ}" >>"${LOG}" 2>&1
chmod 0755 "${OUT}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
