#!/usr/bin/env bash
# Build compact real-WRF oracles for v0.18 RA tail schemes 3/5/7/99.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
WRF_ROOT="${WRF_PRISTINE_ROOT:-/home/user/src/wrf_pristine/WRF}"
BASE_RUN="${WRF_RA_BASE_RUN:-${WRF_ROOT}/test/em_real/oracle_run_v090}"
RUN_ROOT="${HERE}/run"
SAVE_ROOT="${ROOT}/proofs/v018/savepoints/ra_tail_wrf"

mkdir -p "${RUN_ROOT}" "${SAVE_ROOT}"

for code in 3 5 7 99; do
  case_dir="${RUN_ROOT}/ra${code}"
  rm -rf "${case_dir}"
  mkdir -p "${case_dir}"
  cp -a -s "${BASE_RUN}/." "${case_dir}/"
  rm -f "${case_dir}"/namelist.input \
        "${case_dir}"/ra_tail_iofields.txt \
        "${case_dir}"/rsl.* \
        "${case_dir}"/run.log \
        "${case_dir}"/wrf_stdout.log \
        "${case_dir}"/wrfout_d01_*

  python3 "${HERE}/make_ra_namelist.py" "${BASE_RUN}/namelist.input" "${case_dir}/namelist.input" "${code}"
  cp "${HERE}/ra_tail_iofields.txt" "${case_dir}/ra_tail_iofields.txt"

  (
    cd "${case_dir}"
    export OMP_NUM_THREADS=1
    taskset -c 0-3 ./wrf.exe > wrf_stdout.log 2>&1
  )
  if ! grep -q "SUCCESS COMPLETE WRF" "${case_dir}/wrf_stdout.log" "${case_dir}"/rsl.* 2>/dev/null; then
    echo "ERROR: WRF did not complete successfully for RA${code}" >&2
    tail -120 "${case_dir}/wrf_stdout.log" >&2 || true
    exit 1
  fi

  python3 "${HERE}/dump_ra_oracle.py" \
    --scheme "${code}" \
    --run-dir "${case_dir}" \
    --out "${SAVE_ROOT}/ra${code}_wrf_real.json"
done

# Provenance: checksum the exact upstream-identical radiation modules the driver
# dispatches for each scheme, the WRFGPU2_ORACLE-instrumented radiation driver,
# the compiled-out 14/24 stub modules, and the configure.wrf BUILD flags that
# prove those stubs are dummy in this build. This is the physics-pristine source
# evidence the oracle leans on (no JAX self-compare).
CHECKSUMS="${SAVE_ROOT}/wrf_source_checksums.txt"
{
  echo "# v0.18 RA tail -- physics-pristine radiation module/source checksums (sha256)"
  echo "# WRF_ROOT=${WRF_ROOT}"
  echo "# Exact RA modules are upstream-identical; module_radiation_driver.F is WRFGPU2_ORACLE-instrumented."
  echo "# Exact-driver oracle modules (ra_lw/sw_physics dispatch targets):"
  for rel in \
    phys/module_radiation_driver.F \
    phys/module_wrfgpu2_oracle.F \
    phys/module_ra_cam.F \
    phys/module_ra_goddard.F \
    phys/module_ra_flg.F \
    phys/module_ra_gfdleta.F \
    phys/module_ra_rrtmg_lwk.F \
    phys/module_ra_rrtmg_swk.F \
    phys/module_ra_rrtmg_lwf.F \
    phys/module_ra_rrtmg_swf.F \
    configure.wrf
  do
    f="${WRF_ROOT}/${rel}"
    if [ -f "${f}" ]; then
      printf '%s  %s\n' "$(sha256sum "${f}" | cut -d' ' -f1)" "${rel}"
    else
      printf '%s  %s\n' "MISSING" "${rel}"
    fi
  done
} > "${CHECKSUMS}"

echo "OK: RA tail real-WRF savepoints + source checksums written to ${SAVE_ROOT}"
