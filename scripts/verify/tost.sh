#!/usr/bin/env bash
# ROW 6 — Equivalence (paired TOST) on usable corpus cases, T2/U10/V10.
#
# Re-derives FROM SOURCE: runs proofs/m20/paired_tost_scorer.py in CPU-vs-CPU
# plumbing self-test mode (no --gpu-run) on a real corpus CPU wrfout_d02 + the real
# AEMET station obs. This proves the paired-pipeline and TOST machinery are sound:
# the paired delta RMSE_GPU-RMSE_CPU must be EXACTLY 0 when both sides are the same
# CPU run, and >= MIN_PAIRS_PER_BLOCK (30) complete pairs must be formed for the
# 0-24h block of every variable.
#
# HONEST SCOPE (VERIFICATION.md row 6): the full equivalence claim runs the SAME
# scorer with a real GPU forecast (--gpu-run) on the achievable N corpus cases; the
# corpus is single-season, so SEASONAL breadth is a documented v0.2.0 limitation,
# not a v0.1.0 blocker. This script verifies the scorer machinery on CPU; the
# achievable-N GPU pass is manager-sequenced once the GPU forecasts exist.
#
# CPU row -> built AND tested now (forces JAX onto CPU).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
verify_force_cpu
ROW="row6_tost_selftest"

cd "${REPO_ROOT}"
CPU_RUN="/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z"
OUT="proofs/m20/selftest_verify_release.json"

if [ ! -d "${CPU_RUN}" ]; then
  echo "FAIL: corpus CPU run dir missing: ${CPU_RUN}"
  verify_result "${ROW}" "FAIL" "corpus CPU wrfout dir absent (cannot run TOST self-test)"
  exit 1
fi

# shellcheck disable=SC2086
${TASKSET} "${PYBIN}" proofs/m20/paired_tost_scorer.py \
  --cpu-run "${CPU_RUN}" --case-id 20260521_18z_l3_verify --domain d02 \
  --init 2026-05-21T18:00:00+00:00 --fh 24 --out "${OUT}"
run_rc=$?
if [ $run_rc -ne 0 ]; then
  verify_result "${ROW}" "FAIL" "TOST scorer self-test runner exited ${run_rc}"
  exit $run_rc
fi
"${PYBIN}" - "${OUT}" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))
assert d.get("self_test_cpu_vs_cpu") is True, "self-test flag missing"
pb = d["per_block"]
ok = True; lines = []
for var in ("T2", "U10", "V10"):
    blk = pb[var]["0-24h"]
    npairs = blk.get("n_pairs", 0); delta = blk.get("paired_delta_rmse")
    status = blk.get("status")
    good = (status == "OK") and (npairs >= 30) and (delta == 0.0)
    ok = ok and good
    lines.append(f"{var}/0-24h: status={status} n_pairs={npairs} delta_rmse={delta}")
print(f"total_complete_pairs={d.get('total_complete_pairs')}")
for ln in lines: print("  " + ln)
print("ASSERT", "PASS" if ok else "FAIL", "(CPU-vs-CPU paired delta must be 0.0, >=30 pairs)")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "TOST self-test: paired delta RMSE_GPU-RMSE_CPU = 0.0 on >=30 pairs/var (machinery sound; GPU achievable-N pass + seasonal limitation manager-sequenced)"
else
  verify_result "${ROW}" "FAIL" "TOST self-test plumbing did not yield zero paired delta / sufficient pairs"
fi
exit $rc
