#!/usr/bin/env bash
# Maintainer-only oracle build tool. It links the *real* WRF Fortran RRTMG
# objects into a small harness so RRTMG kernels can be parity-checked against a
# genuine (non-JAX) WRF oracle. It requires a local WRF Fortran build; the paths
# below default to the maintainer's reference layout and are all overridable via
# environment variables. It is NOT needed to run a forecast (see docs/quickstart.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${ROOT}/data/scratch"
OUT="${SCRATCH}/wrf_rrtmg_harness"
OBJ="${SCRATCH}/wrf_rrtmg_harness.o"
LOG="${SCRATCH}/wrf_rrtmg_harness_build.log"
RUNTIME="${SCRATCH}/rrtmg_runtime"
WRF_ROOT="${WRF_ROOT:-/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF}"
WRF_BUILD="${WRF_BUILD:-${WRF_ROOT}/_build_gen2_dmpar}"
WRF_ENV="${WRF_ENV:-/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build}"
MOD_DIR="${WRF_MOD_DIR:-${WRF_ROOT}/install_gen2_dmpar/modules}"
SW_OBJ="${WRF_BUILD}/CMakeFiles/WRF_Core.dir/phys/module_ra_rrtmg_sw.F.o"
LW_OBJ="${WRF_BUILD}/CMakeFiles/WRF_Core.dir/phys/module_ra_rrtmg_lw.F.o"
SW_DATA="${WRF_ROOT}/install_gen2_dmpar/run/RRTMG_SW_DATA"
LW_DATA="${WRF_ROOT}/install_gen2_dmpar/run/RRTMG_LW_DATA"

# Fallback to the independent pristine WRFv4 build (gfortran serial, em_quarter_ss
# arbiter, see project memory) when the Gen2 NVHPC objects are unavailable. The
# pristine build ships the same module_ra_rrtmg_sw/lw objects + RRTMG data files
# and is a genuine (non-JAX) WRF Fortran oracle. It only lacks the WRF framework
# error handler wrf_error_fatal3, which we satisfy with a tiny local stub.
# All three are overridable for a different local WRF build.
PRISTINE_PHYS="${WRF_PRISTINE_PHYS:-/home/user/src/wrf_pristine/WRF/phys}"
PRISTINE_RUN="${WRF_PRISTINE_RUN:-/home/user/src/wrf_pristine/WRF/run}"
PRISTINE_FC="${WRF_PRISTINE_FC:-gfortran}"
USE_PRISTINE=0
if [[ ! -f "${SW_OBJ}" || ! -f "${LW_OBJ}" ]] \
   && [[ -f "${PRISTINE_PHYS}/module_ra_rrtmg_sw.o" && -f "${PRISTINE_PHYS}/module_ra_rrtmg_lw.o" ]]; then
  USE_PRISTINE=1
  MOD_DIR="${PRISTINE_PHYS}"
  SW_OBJ="${PRISTINE_PHYS}/module_ra_rrtmg_sw.o"
  LW_OBJ="${PRISTINE_PHYS}/module_ra_rrtmg_lw.o"
  SW_DATA="${PRISTINE_RUN}/RRTMG_SW_DATA"
  LW_DATA="${PRISTINE_RUN}/RRTMG_LW_DATA"
  [[ -x "${PRISTINE_FC}" ]] && FC="${PRISTINE_FC}"
fi

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

STUB_OBJ=""
LINK_LIBS="-lmvec -lm"
if [[ "${USE_PRISTINE}" == "1" ]]; then
  STUB_SRC="${SCRATCH}/wrf_error_fatal3_stub.f90"
  STUB_OBJ="${SCRATCH}/wrf_error_fatal3_stub.o"
  cat >"${STUB_SRC}" <<'STUBEOF'
subroutine wrf_error_fatal3(file, line, str)
  character(len=*), intent(in) :: file, str
  integer, intent(in) :: line
  write(0,*) 'WRF_ERROR_FATAL3 at ', trim(file), ' line ', line, ': ', trim(str)
  stop 1
end subroutine wrf_error_fatal3
STUBEOF
  "${FC}" -ffree-line-length-none -c -o "${STUB_OBJ}" "${STUB_SRC}" >>"${LOG}" 2>&1
  LINK_LIBS="-lm"
fi

"${FC}" -o "${OUT}" "${OBJ}" ${STUB_OBJ} "${SW_OBJ}" "${LW_OBJ}" ${LINK_LIBS} >>"${LOG}" 2>&1

chmod 0755 "${OUT}"
sha256sum "${OUT}" | tee -a "${LOG}"
echo "${OUT}"
