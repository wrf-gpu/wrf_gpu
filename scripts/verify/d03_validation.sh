#!/usr/bin/env bash
# ROW 5 — Canary 1 km (d03): finite & stable to 24 h, near-CPU-WRF, beats
# persistence, passes the bounded gate (T2 RMSE <= 3.0 K gate; beats persistence).
#
# Re-derives FROM SOURCE: runs scripts/d03_replay.py (the real GPU 1 km Tenerife
# replay-nest forecast from replayed d02 boundaries, scored vs corpus L3 d03 truth
# with a persistence baseline). Asserts verdict == "D03_1KM_VALIDATED".
#
# STATUS NOTE (VERIFICATION.md row 5): this row is BLOCKED pending the HFX fix
# (#56). The current code yields verdict=D03_1KM_BOUNDED_FAIL (T2 RMSE ~3.01 K just
# over the 3.0 K gate). This script asserts the REAL gate honestly, so it records
# FAIL until the HFX fix lands -- it does NOT relax the gate to manufacture a PASS.
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row5_d03_validation"

TAG="verify_release"
RUNNER="${TASKSET} ${PYBIN} scripts/d03_replay.py --hours 24 --tag ${TAG}"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
# d03_replay.py returns 0 only on D03_1KM_VALIDATED, 2 on bounded-fail/blocked.
# shellcheck disable=SC2086
${TASKSET} "${PYBIN}" scripts/d03_replay.py --hours 24 --tag "${TAG}"
run_rc=$?
SUMMARY="proofs/v010_validation/d03_summary_${TAG}.json"
"${PYBIN}" - "${SUMMARY}" <<'PY'
import sys, json, os
p = sys.argv[1]
if not os.path.exists(p):
    print(f"FAIL: d03 summary {p} not written"); sys.exit(1)
d = json.load(open(p))
verdict = d.get("verdict"); vstat = d.get("validation_status")
flf = d.get("final_lead_fields", {})
t2 = flf.get("T2", {}); v10 = flf.get("V10", {})
ok = (verdict == "D03_1KM_VALIDATED") and (vstat == "PASS")
print(f"verdict={verdict} validation_status={vstat} "
      f"T2_rmse={t2.get('rmse')}(gate={t2.get('threshold')},within={t2.get('within_threshold')}) "
      f"V10_beats_persistence={v10.get('beats_persistence')} "
      f"wall_per_fh_s={d.get('wall_clock_per_forecast_hour_s')}")
print("ASSERT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "d03 verdict=D03_1KM_VALIDATED (T2 RMSE <= 3.0 K gate, beats persistence)"
else
  verify_result "${ROW}" "FAIL" "d03 bounded gate not met (BLOCKED pending HFX fix #56; T2 RMSE over 3.0 K gate)"
fi
exit $rc
