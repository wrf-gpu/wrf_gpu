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
# Run the comparator on the VALIDATED platform (GPU). The m6b6 module defaults to
# JAX CPU via os.environ.setdefault; on CPU the RRTMG physics adapter hard-segfaults
# (native crash), so force the GPU here unless the caller already pinned a platform.
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cuda}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
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
    gap = res.get("comparator_composition_gap")
    summary[tier]["comparator_composition_gap"] = bool(gap)
    if gap:
        summary[tier]["is_production_dycore_defect"] = bool(gap.get("is_production_dycore_defect"))
    print(f"  tier={tier} passed={ok} count={res.get('savepoint_count')} self_compare={self_cmp} outcome={res.get('outcome')}")
    all_ok = all_ok and ok and (self_cmp is False)
print("SUMMARY", json.dumps(summary))
# Distinguish an HONEST comparator-harness gap (NOT a production-dycore defect)
# from a real production parity failure. The dycore is independently validated by
# the idealized + conservation + d02/d03 rows; this lane is a superseded validation
# composition missing the small_step_prep-derived leaves.
gap_only = (not all_ok) and all(
    bool(v.get("comparator_composition_gap")) and not bool(v.get("is_production_dycore_defect"))
    for v in summary.values()
)
print("ASSERT", "PASS" if all_ok else ("FAIL_COMPARATOR_HARNESS_GAP" if gap_only else "FAIL"))
sys.exit(0 if all_ok else (3 if gap_only else 1))
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "JAX-vs-WRF coupled-step parity PASS @ 10 savepoints x 3 tiers (column/patch16/golden), self_compare=False"
elif [ $rc -eq 3 ]; then
  verify_result "${ROW}" "FAIL" "COMPARATOR-HARNESS gap (NOT a production-dycore defect): theta=None threaded WRF-faithfully (_seed_coupled_work_theta), exposing the deeper gap that the validation-only coupled_timestep_core core path is fed a bare from_mapping state lacking the ~30 small_step_prep-derived leaves (c2a/alt/al/phb/ph_1/cf*/c1f/c2f/rdn/ht/pm1/rw_tend_pg_buoy); calc_p_rho/advance_w then emit non-finite p/ph. Production dycore validated by idealized+conservation+d02/d03. v0.2.0: port small_step_prep into the comparator or route it through the operational _rk_scan_step."
else
  verify_result "${ROW}" "FAIL" "operator parity vs WRF savepoints did not pass (see above)"
fi
exit $rc
