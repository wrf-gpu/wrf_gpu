#!/usr/bin/env bash
# Build compact real-WRF oracles for v0.18 CU tail schemes 7/10/11.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
WRF_ROOT="${WRF_PRISTINE_ROOT:-/home/user/src/wrf_pristine/WRF}"
BASE_RUN="${WRF_TAIL_BASE_RUN:-${WRF_ROOT}/test/em_real/oracle_run_v090}"
RUN_ROOT="${HERE}/run"
SAVE_ROOT="${ROOT}/proofs/v018/savepoints/cumulus_tail_wrf"

mkdir -p "${RUN_ROOT}" "${SAVE_ROOT}"

for code in 7 10 11; do
  case_dir="${RUN_ROOT}/cu${code}"
  rm -rf "${case_dir}"
  mkdir -p "${case_dir}"
  cp -a -s "${BASE_RUN}/." "${case_dir}/"
  rm -f "${case_dir}"/namelist.input \
        "${case_dir}"/cu_tail_iofields.txt \
        "${case_dir}"/rsl.* \
        "${case_dir}"/run.log \
        "${case_dir}"/wrfout_d01_*

  python3 "${HERE}/make_tail_namelist.py" "${BASE_RUN}/namelist.input" "${case_dir}/namelist.input" "${code}"
  cp "${HERE}/cu_tail_iofields.txt" "${case_dir}/cu_tail_iofields.txt"

  (
    cd "${case_dir}"
    export OMP_NUM_THREADS=1
    taskset -c 0-3 ./wrf.exe > wrf_stdout.log 2>&1
  )
  if ! grep -q "SUCCESS COMPLETE WRF" "${case_dir}/wrf_stdout.log" "${case_dir}"/rsl.* 2>/dev/null; then
    echo "ERROR: WRF did not complete successfully for CU${code}" >&2
    tail -120 "${case_dir}/wrf_stdout.log" >&2 || true
    exit 1
  fi

  python3 "${HERE}/dump_tail_oracle.py" \
    --scheme "${code}" \
    --run-dir "${case_dir}" \
    --out "${SAVE_ROOT}/cu${code}_wrf_real.json"
done

echo "OK: CU tail real-WRF savepoints written to ${SAVE_ROOT}"
