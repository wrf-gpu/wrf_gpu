#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_mynn_harness"
OBJ="${SCRATCH}/wrf_mynn_harness.o"
LOG="${SCRATCH}/wrf_mynn_harness_build.log"
WRF_ROOT="/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF"
WRF_MAIN="${WRF_ROOT}/main"
WRF_EDMF_SRC="${WRF_ROOT}/phys/MYNN-EDMF/module_bl_mynnedmf.F90"
WRF_EDMF_OBJ="${WRF_ROOT}/phys/module_bl_mynnedmf.o"
WRF_COMMON_OBJ="${WRF_ROOT}/phys/module_bl_mynnedmf_common.o"
WRF_KIND_OBJ="${WRF_ROOT}/phys/ccpp_kind_types.o"
WRF_CONST_OBJ="${WRF_ROOT}/share/module_model_constants.o"

mkdir -p "${SCRATCH}"
: > "${LOG}"

if [[ -f /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh >>"${LOG}" 2>&1 || true
fi

FC="${FC:-$(command -v nvfortran || true)}"
if [[ -z "${FC}" ]]; then
  echo "nvfortran not found; cannot compile WRF-linked MYNN harness" | tee -a "${LOG}" >&2
  exit 1
fi

for path in "${WRF_MAIN}/module_bl_mynnedmf.mod" "${WRF_MAIN}/module_bl_mynnedmf_common.mod" \
            "${WRF_EDMF_OBJ}" "${WRF_COMMON_OBJ}" "${WRF_KIND_OBJ}" "${WRF_CONST_OBJ}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required WRF MYNN object/module missing: ${path}" | tee -a "${LOG}" >&2
    exit 1
  fi
done

needs_build=0
if [[ ! -x "${OUT}" ]]; then
  needs_build=1
elif [[ "${ROOT}/scripts/wrf_mynn_harness.f90" -nt "${OUT}" || "${WRF_EDMF_OBJ}" -nt "${OUT}" || "${WRF_COMMON_OBJ}" -nt "${OUT}" ]]; then
  needs_build=1
fi

if [[ "${needs_build}" == "0" ]]; then
  echo "reusing existing WRF-object-linked MYNN harness: ${OUT}" >>"${LOG}"
  sha256sum "${OUT}" | tee -a "${LOG}"
  echo "${OUT}"
  exit 0
fi

echo "compiler=${FC}" >>"${LOG}"
echo "source=${WRF_EDMF_SRC}" >>"${LOG}"
echo "linked_objects=${WRF_EDMF_OBJ} ${WRF_COMMON_OBJ} ${WRF_KIND_OBJ} ${WRF_CONST_OBJ}" >>"${LOG}"

"${FC}" -I"${WRF_MAIN}" -c -o "${OBJ}" "${ROOT}/scripts/wrf_mynn_harness.f90" >>"${LOG}" 2>&1
"${FC}" -o "${OUT}" "${OBJ}" \
  "${WRF_EDMF_OBJ}" "${WRF_COMMON_OBJ}" "${WRF_KIND_OBJ}" "${WRF_CONST_OBJ}" >>"${LOG}" 2>&1
chmod 0755 "${OUT}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
