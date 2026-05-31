#!/usr/bin/env bash
# ROW 4 — Canary 3 km (d02): finite & stable, no blow-up, near-CPU-WRF, beats
# persistence on winds.
#
# Re-derives FROM SOURCE: runs proofs/v010_validation/v010_d02_validate.py --execute
# (the real GPU d02 forecast via the operational segmented entry, scored vs the
# nightly CPU-WRF corpus wrfout on T2/U10/V10/PRECIP, full + Tenerife region, plus
# a persistence baseline). Asserts verdict == "D02_VALIDATED" with all_pass &&
# no_blowup. MUST be re-run on the FINAL post-HFX-fix code (VERIFICATION.md row 4).
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row4_d02_validation"

OUT="proofs/v010_validation/v010_d02_result.json"
RUNNER="${TASKSET} ${PYBIN} proofs/v010_validation/v010_d02_validate.py --execute --out ${OUT}"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
# Re-run the real GPU d02 validation (writes ${OUT}).
# shellcheck disable=SC2086
${TASKSET} "${PYBIN}" proofs/v010_validation/v010_d02_validate.py --execute --out "${OUT}"
run_rc=$?
if [ $run_rc -ne 0 ]; then
  verify_result "${ROW}" "FAIL" "d02 validation runner exited ${run_rc}"
  exit $run_rc
fi
# Assert the gate from the freshly-written result.
"${PYBIN}" - "${OUT}" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))
verdict = d.get("verdict"); all_pass = bool(d.get("all_pass")); no_blow = bool(d.get("no_blowup"))
ok = (verdict == "D02_VALIDATED") and all_pass and no_blow
print(f"verdict={verdict} all_pass={all_pass} no_blowup={no_blow} wall_s={d.get('wall_s')}")
print("ASSERT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "d02 verdict=D02_VALIDATED (all_pass && no_blowup) on $(verify_commit_sha)"
else
  verify_result "${ROW}" "FAIL" "d02 verdict != D02_VALIDATED or blow-up/score gate not met"
fi
exit $rc
