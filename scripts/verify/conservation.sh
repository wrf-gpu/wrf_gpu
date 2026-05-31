#!/usr/bin/env bash
# ROW 7 — Conservation: dry-mass / water / energy budgets bounded; guards NOT
# load-bearing.
#
# Re-derives FROM SOURCE: runs scripts/sprintU_guards_off_proof.py, which (a) steps
# the REAL Canary d02 operational dycore with disable_guards=True (no theta
# limiter, no dry-mass guard, no finite fallback) and shows the bare dycore stays
# finite AND genuinely fp64; and (b) runs the full warm bubble guards-OFF and
# re-evaluates the F7N verdict checks (incl. the relative dry-mass-drift <= 1e-8
# conservation check), proving the dycore PASSES on its own and the guards are a
# safety net, not a prop. Asserts verdict == "PASS".
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row7_conservation"

RUNNER="${TASKSET} ${PYBIN} scripts/sprintU_guards_off_proof.py"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
# shellcheck disable=SC2086
${TASKSET} "${PYBIN}" scripts/sprintU_guards_off_proof.py
run_rc=$?
if [ $run_rc -ne 0 ]; then
  verify_result "${ROW}" "FAIL" "guards-off conservation runner exited ${run_rc}"
  exit $run_rc
fi
PROOF="proofs/sprintU/guards_off_operational_proof.json"
"${PYBIN}" - "${PROOF}" <<'PY'
import sys, json, os
p = sys.argv[1]
if not os.path.exists(p):
    print(f"FAIL: {p} not written"); sys.exit(1)
d = json.load(open(p))
verdict = d.get("verdict")
real = d.get("real_case_guards_off", {})
warm = d.get("warm_bubble_guards_off", {})
real_fin = bool(real.get("all_finite_without_guards"))
real_fp64 = bool(real.get("all_prognostics_fp64"))
warm_pass = bool(warm.get("passes_without_guards"))
# the dry-mass-drift conservation check value from the guards-off warm bubble:
mass = warm.get("checks", {}).get("relative_mass_drift", {}).get("value")
ok = (verdict == "PASS") and real_fin and real_fp64 and warm_pass
print(f"verdict={verdict} real_finite_guards_off={real_fin} real_fp64={real_fp64} "
      f"warm_bubble_guards_off_PASS={warm_pass} guards_off_mass_drift={mass} "
      f"limiter_not_load_bearing={warm.get('limiter_not_load_bearing')}")
print("ASSERT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "guards-off dycore finite+fp64; warm bubble PASSES guards-off incl. dry-mass-drift conservation check (guards not load-bearing)"
else
  verify_result "${ROW}" "FAIL" "conservation/guards-off proof did not reach PASS"
fi
exit $rc
