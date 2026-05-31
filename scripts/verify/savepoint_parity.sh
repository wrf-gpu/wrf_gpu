#!/usr/bin/env bash
# ROW 3 — Operator parity vs pristine WRF v4 savepoints.
#
# Re-derives FROM SOURCE: runs the JAX coupled-step dycore on the real Canary
# wrfout initial condition and compares EVERY savepoint field against the real
# WRF v4 reference fixture (NOT a JAX-vs-JAX self-compare -- compare_tier sets
# oracle.self_compare=False) at the per-operator tolerance ladder. Asserts all
# three tiers (column / patch16 / golden) PASS.
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row3_savepoint_parity"

RUNNER="${TASKSET} ${PYBIN} -c 'from scripts.m6b6_coupled_step_compare import compare_tier; ...'  (compares 3 tiers x 10 steps JAX-vs-WRF)"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
${TASKSET} "${PYBIN}" - <<'PY'
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, ".")  # so `import scripts.*` resolves
from scripts.m6b6_coupled_step_compare import compare_tier, SOURCE_WRFOUT, SOURCE_WRFBDY

if not (SOURCE_WRFOUT.exists() and SOURCE_WRFBDY.exists()):
    print(f"FAIL: source WRF fixture missing: {SOURCE_WRFOUT} / {SOURCE_WRFBDY}")
    sys.exit(1)

STEPS = 10  # the published B6 parity depth (column/golden/patch16 = 10 savepoints each)
tmp = Path(tempfile.mkdtemp(prefix="verify_row3_savepoints_"))
all_ok = True
summary = {}
for tier in ("column", "patch16", "golden"):
    res = compare_tier(tier, STEPS, tmp)
    ok = bool(res.get("passed"))
    self_cmp = res.get("oracle", {}).get("self_compare")
    summary[tier] = {"passed": ok, "savepoint_count": res.get("savepoint_count"),
                     "outcome": res.get("outcome"), "self_compare": self_cmp}
    print(f"  tier={tier} passed={ok} count={res.get('savepoint_count')} self_compare={self_cmp} outcome={res.get('outcome')}")
    all_ok = all_ok and ok and (self_cmp is False)
print("SUMMARY", json.dumps(summary))
print("ASSERT", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "JAX-vs-WRF coupled-step parity PASS @ 10 savepoints x 3 tiers (column/patch16/golden), self_compare=False"
else
  verify_result "${ROW}" "FAIL" "operator parity vs WRF savepoints did not pass (see above)"
fi
exit $rc
