#!/usr/bin/env bash
# ROW 1 — Dycore: Skamarock warm bubble matches the benchmark reference.
#
# Re-derives FROM SOURCE: runs gpuwrf.ic_generators.idealized.run_warm_bubble_case
# (the SAME function the close-gate CI and f7n_official_run use) on the GPU, then
# asserts verdict == "PASS" with all 6 published checks passing.
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row1_idealized_warmbubble"

RUNNER="${TASKSET} ${PYBIN} scripts/f7n_official_run.py bubble"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
PROOF_DIR="proofs/verify_run/row1"
# Re-run the real case to a fresh proof dir, then assert the emitted verdict.
${TASKSET} "${PYBIN}" - "$PROOF_DIR" <<'PY'
import sys, json
from jax import config
config.update("jax_enable_x64", True)
from gpuwrf.ic_generators.idealized import run_warm_bubble_case
proof_dir = sys.argv[1]
r = run_warm_bubble_case(proof_dir=proof_dir, require_gpu=True)
payload = r.payload
checks = payload.get("checks", {})
all_pass = all(c.get("passed") for c in checks.values())
print("VERDICT", payload.get("verdict"), "STATUS", payload.get("status"))
for k, c in sorted(checks.items()):
    print(f"  {k}: value={c.get('value')} passed={c.get('passed')}")
ok = (payload.get("verdict") == "PASS") and (payload.get("status") == "RAN_TO_COMPLETION") and all_pass
mw = checks.get("max_abs_w_500s", {}).get("value")
rise = checks.get("thermal_rise_500s", {}).get("value")
md = checks.get("relative_mass_drift", {}).get("value")
note = f"verdict={payload.get('verdict')} max|w|500s={mw} thermal_rise={rise} mass_drift={md} checks={sum(bool(c.get('passed')) for c in checks.values())}/{len(checks)}"
print("ASSERT", "PASS" if ok else "FAIL", note)
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "warm bubble verdict=PASS (6/6 checks, RAN_TO_COMPLETION)"
else
  verify_result "${ROW}" "FAIL" "warm bubble did not reach PASS verdict (see above)"
fi
exit $rc
