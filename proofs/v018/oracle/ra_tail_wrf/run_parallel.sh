#!/usr/bin/env bash
# Parallel driver for the v0.18 RA tail real-WRF oracles (schemes 3/5/7/99).
# Each scheme's physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe is pinned
# to its own core (0..3) and run concurrently; savepoints + source checksums are
# dumped after all complete.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
WRF_ROOT="${WRF_PRISTINE_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}"
BASE_RUN="${WRF_RA_BASE_RUN:-${WRF_ROOT}/test/em_real/oracle_run_v090}"
RUN_ROOT="${HERE}/run"
SAVE_ROOT="${ROOT}/proofs/v018/savepoints/ra_tail_wrf"
mkdir -p "${RUN_ROOT}" "${SAVE_ROOT}"

CODES=(3 5 7 99)
CORES=(0 1 2 3)

launch() {
  local code="$1" core="$2"
  local case_dir="${RUN_ROOT}/ra${code}"
  rm -rf "${case_dir}"; mkdir -p "${case_dir}"
  cp -a -s "${BASE_RUN}/." "${case_dir}/"
  rm -f "${case_dir}"/namelist.input "${case_dir}"/ra_tail_iofields.txt \
        "${case_dir}"/rsl.* "${case_dir}"/wrf_stdout.log "${case_dir}"/wrfout_d01_*
  python3 "${HERE}/make_ra_namelist.py" "${BASE_RUN}/namelist.input" "${case_dir}/namelist.input" "${code}"
  cp "${HERE}/ra_tail_iofields.txt" "${case_dir}/ra_tail_iofields.txt"
  ( cd "${case_dir}" && export OMP_NUM_THREADS=1 && taskset -c "${core}" ./wrf.exe > wrf_stdout.log 2>&1 )
}

echo "launching RA tail oracle runs (18h) in parallel: codes=${CODES[*]}"
pids=()
for idx in "${!CODES[@]}"; do
  launch "${CODES[$idx]}" "${CORES[$idx]}" &
  pids+=("$!")
done

fail=0
for idx in "${!CODES[@]}"; do
  wait "${pids[$idx]}" || true
  code="${CODES[$idx]}"
  log="${RUN_ROOT}/ra${code}/wrf_stdout.log"
  if grep -q "SUCCESS COMPLETE WRF" "${log}" 2>/dev/null; then
    echo "RA${code}: WRF SUCCESS"
  else
    echo "RA${code}: WRF FAILED" >&2; tail -40 "${log}" >&2 2>/dev/null; fail=1
  fi
done
[ "${fail}" -eq 0 ] || { echo "ABORT: a WRF oracle run failed" >&2; exit 1; }

for code in "${CODES[@]}"; do
  python3 "${HERE}/dump_ra_oracle.py" --scheme "${code}" \
    --run-dir "${RUN_ROOT}/ra${code}" --out "${SAVE_ROOT}/ra${code}_wrf_real.json" \
    || { echo "DUMP FAILED (trivial oracle?) RA${code}" >&2; fail=1; }
done

CHECKSUMS="${SAVE_ROOT}/wrf_source_checksums.txt"
{
  echo "# v0.18 RA tail -- physics-pristine radiation module/source checksums (sha256)"
  echo "# WRF_ROOT=${WRF_ROOT}"
  echo "# Exact RA modules are upstream-identical; module_radiation_driver.F is WRFGPU2_ORACLE-instrumented."
  echo "# Exact-driver oracle modules (ra_lw/sw_physics dispatch targets) + compiled-out 14/24 stubs + BUILD flags:"
  for rel in \
    phys/module_radiation_driver.F phys/module_wrfgpu2_oracle.F \
    phys/module_ra_cam.F phys/module_ra_goddard.F \
    phys/module_ra_flg.F phys/module_ra_gfdleta.F \
    phys/module_ra_rrtmg_lwk.F phys/module_ra_rrtmg_swk.F \
    phys/module_ra_rrtmg_lwf.F phys/module_ra_rrtmg_swf.F configure.wrf
  do
    f="${WRF_ROOT}/${rel}"
    if [ -f "${f}" ]; then printf '%s  %s\n' "$(sha256sum "${f}" | cut -d' ' -f1)" "${rel}"
    else printf '%s  %s\n' "MISSING" "${rel}"; fi
  done
} > "${CHECKSUMS}"

[ "${fail}" -eq 0 ] && echo "OK: RA tail savepoints + checksums in ${SAVE_ROOT}" || { echo "FAIL: see above" >&2; exit 1; }
