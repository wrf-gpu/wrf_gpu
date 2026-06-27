#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_sfclay_harness"
OBJ="${SCRATCH}/wrf_sfclay_harness.o"
LOG="${SCRATCH}/wrf_sfclay_harness_build.log"
DEFAULT_WRF_ROOT="$(
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
from gpuwrf.config.paths import wrf_root
print(wrf_root())
PY
)"
WRF_ROOT="${WRF_ROOT:-${GPUWRF_WRF_ROOT:-${DEFAULT_WRF_ROOT}}}"
WRF_MAIN="${WRF_ROOT}/main"
WRF_SFCLAY_SRC="${WRF_ROOT}/phys/module_sf_sfclay.F"
WRF_SFCLAY_OBJ="${WRF_ROOT}/phys/module_sf_sfclay.o"

mkdir -p "${SCRATCH}"
: > "${LOG}"

WRF_ENV="${WRF_ENV:-${GPUWRF_WRF_ENV:-${WRF_ROOT%/WRF}/env_wrf_gpu.sh}}"
if [[ -f "${WRF_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${WRF_ENV}" >>"${LOG}" 2>&1 || true
fi

FC="${FC:-$(command -v nvfortran || true)}"
if [[ -z "${FC}" ]]; then
  echo "nvfortran not found; cannot compile WRF-linked SFCLAY harness" | tee -a "${LOG}" >&2
  exit 1
fi

for path in "${WRF_MAIN}/module_sf_sfclay.mod" "${WRF_SFCLAY_OBJ}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required WRF SFCLAY object/module missing: ${path}" | tee -a "${LOG}" >&2
    exit 1
  fi
done

needs_build=0
if [[ ! -x "${OUT}" ]]; then
  needs_build=1
elif [[ "${ROOT}/scripts/wrf_sfclay_harness.f90" -nt "${OUT}" || "${WRF_SFCLAY_OBJ}" -nt "${OUT}" ]]; then
  needs_build=1
fi

if [[ "${needs_build}" == "0" ]]; then
  echo "reusing existing WRF-object-linked SFCLAY harness: ${OUT}" >>"${LOG}"
  nm "${OUT}" | grep -i "module_sf_sfclay_sfclay1d" >>"${LOG}"
  sha256sum "${OUT}" | tee -a "${LOG}"
  echo "${OUT}"
  exit 0
fi

echo "compiler=${FC}" >>"${LOG}"
echo "source=${WRF_SFCLAY_SRC}" >>"${LOG}"
echo "linked_objects=${WRF_SFCLAY_OBJ}" >>"${LOG}"

"${FC}" -I"${WRF_MAIN}" -c -o "${OBJ}" "${ROOT}/scripts/wrf_sfclay_harness.f90" >>"${LOG}" 2>&1
"${FC}" -o "${OUT}" "${OBJ}" "${WRF_SFCLAY_OBJ}" >>"${LOG}" 2>&1
chmod 0755 "${OUT}"
nm "${OUT}" | grep -i "module_sf_sfclay_sfclay1d" >>"${LOG}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
