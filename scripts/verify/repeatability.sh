#!/usr/bin/env bash
# ROW 8 — Reproducibility: deterministic re-run + restart-continuity.
#
# Re-derives FROM SOURCE: runs scripts/m7_daily_pipeline.py with BOTH --repeat
# (run the full GPU forecast a second time and compare the final wrfout) and
# --restart-at-hour (checkpoint mid-run, restart, and compare the final wrfout to
# the continuous run). The pipeline writes repeatability.json + restart_in_pipeline.json
# whose comparison status must be PASS (bitwise / within-tol identical final state).
# A short --hours window is used (deterministic re-run is independent of length).
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row8_repeatability"

HOURS="${VERIFY_REPEAT_HOURS:-2}"
RESTART_AT="${VERIFY_RESTART_AT_HOUR:-1}"
PROOF_DIR="proofs/v010_validation"
RUNNER="${TASKSET} ${PYBIN} scripts/m7_daily_pipeline.py --hours ${HOURS} --repeat --restart-at-hour ${RESTART_AT} --proof-dir ${PROOF_DIR}"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
# shellcheck disable=SC2086
${TASKSET} "${PYBIN}" scripts/m7_daily_pipeline.py \
  --hours "${HOURS}" --repeat --restart-at-hour "${RESTART_AT}" --proof-dir "${PROOF_DIR}"
# pipeline returns 0 only on PIPELINE_GREEN; the determinism/restart gates are
# additionally enforced in the proof JSONs we assert below.
"${PYBIN}" - "${PROOF_DIR}/repeatability.json" "${PROOF_DIR}/restart_in_pipeline.json" <<'PY'
import sys, json, os
rep_p, res_p = sys.argv[1], sys.argv[2]
def load(p):
    return json.load(open(p)) if os.path.exists(p) else {"status": "MISSING"}
rep = load(rep_p); res = load(res_p)
rep_ok = rep.get("status") == "PASS"
res_ok = res.get("status") == "PASS"
print(f"repeatability.status={rep.get('status')} (comparison={rep.get('comparison',{}).get('status')})")
print(f"restart.status={res.get('status')} (restart_at_hour={res.get('restart_at_hour')} "
      f"comparison={res.get('comparison',{}).get('status')})")
ok = rep_ok and res_ok
print("ASSERT", "PASS" if ok else "FAIL", "(both deterministic re-run AND restart-continuity must PASS)")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "deterministic re-run + restart-continuity both PASS (final wrfout within-tol identical)"
else
  verify_result "${ROW}" "FAIL" "repeatability and/or restart-continuity gate not met"
fi
exit $rc
