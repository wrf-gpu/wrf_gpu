#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRF_ROOT="/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_thompson_harness"
OBJ="${SCRATCH}/wrf_thompson_harness.o"
THOMPSON_SRC="${ROOT}/../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre"
if [[ ! -f "${THOMPSON_SRC}" ]]; then
  THOMPSON_SRC="/home/user/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre"
fi
PATCHED_SRC="${SCRATCH}/module_mp_thompson_nosed.F90"
PATCHED_OBJ="${SCRATCH}/module_mp_thompson_nosed.o"
LOG="${SCRATCH}/wrf_thompson_harness_build.log"

mkdir -p "${SCRATCH}"
: > "${LOG}"

if [[ -f /home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh >>"${LOG}" 2>&1 || true
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

if [[ ! -f "${THOMPSON_SRC}" ]]; then
  echo "WRF Thompson source snapshot not found: ${THOMPSON_SRC}" | tee -a "${LOG}" >&2
  exit 1
fi

python - "${THOMPSON_SRC}" "${PATCHED_SRC}" >>"${LOG}" 2>&1 <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text(encoding="utf-8")
marker = "!..Sedimentation of mixing ratio is the integral of v(D)*m(D)*N(D)*dD,"
insert = """\
!..M5-S1 no-sedimentation harness patch.
!..The source/sink fixture excludes sedimentation. Zero the terminal-velocity
!..arrays after WRF computes them and before the sedimentation flux loops
!..at module_mp_thompson.F.pre lines 3854-4003.
      vtrk(:) = 0.
      vtnrk(:) = 0.
      vtik(:) = 0.
      vtnik(:) = 0.
      vtsk(:) = 0.
      vtgk(:) = 0.
      vtngk(:) = 0.
      vtck(:) = 0.
      vtnck(:) = 0.
"""
if marker not in text:
    raise SystemExit(f"sedimentation marker not found in {source}")
text = text.replace(marker, insert + marker, 1)
target.write_text(text, encoding="utf-8")
print(f"wrote patched no-sedimentation source: {target}")
PY

"${FC}" -c -Mpreprocess -module "${SCRATCH}" -I"${MOD_DIR}" -I"${WRF_ROOT}/main" \
  -I"${WRF_ROOT}/install_gen2_dmpar/modules" \
  -I"${WRF_ROOT}/external/esmf_time_f90" -I"${WRF_ROOT}/install_gen2_dmpar/esmf_time_f90" \
  -o "${PATCHED_OBJ}" "${PATCHED_SRC}" >>"${LOG}" 2>&1

"${FC}" -c -I"${SCRATCH}" -I"${MOD_DIR}" -I"${WRF_ROOT}/main" -I"${WRF_ROOT}/install_gen2_dmpar/modules" \
  -I"${WRF_ROOT}/external/esmf_time_f90" -I"${WRF_ROOT}/install_gen2_dmpar/esmf_time_f90" \
  -o "${OBJ}" "${ROOT}/scripts/wrf_thompson_harness.f90" >>"${LOG}" 2>&1

"${FC}" -o "${OUT}" "${OBJ}" \
  "${PATCHED_OBJ}" \
  "${WRF_ROOT}/phys/module_mp_radar.o" \
  "${WRF_ROOT}/share/module_model_constants.o" \
  "${WRF_ROOT}/frame/module_wrf_error.o" \
  >>"${LOG}" 2>&1

chmod 0755 "${OUT}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
