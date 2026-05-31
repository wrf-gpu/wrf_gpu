#!/usr/bin/env bash
# ROW 11 — Device residency: zero host<->device transfer inside the timestep loop.
#
# Re-derives FROM SOURCE: runs proofs/perf/fusion_transfer_audit.py, which warms
# the real coupled run_forecast_operational, profiles a warmed forecast under
# jax.profiler.trace, and counts post-init H2D/D2H memcpy bytes (and scans the
# compiled HLO for copy-start/outfeed/infeed/send/recv host-transfer ops). Asserts
# transfer count == 0 (device-resident timestep loop).
#
# GPU row -> manager-sequenced. Build now, run on the idle GPU with VERIFY_RUN_GPU=1.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ROW="row11_device_residency"

HOURS="${VERIFY_AUDIT_HOURS:-0.5}"
RUNNER="${TASKSET} ${PYBIN} proofs/perf/fusion_transfer_audit.py --hours ${HOURS}"
verify_gpu_guard "${ROW}" "${RUNNER}"

cd "${REPO_ROOT}"
# fusion_transfer_audit reduces XLA_PYTHON_CLIENT_MEM_FRACTION in its docstring;
# use the documented 0.5 to leave headroom alongside the manager's GPU work.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
# shellcheck disable=SC2086
${TASKSET} "${PYBIN}" proofs/perf/fusion_transfer_audit.py --hours "${HOURS}"
run_rc=$?
if [ $run_rc -ne 0 ]; then
  verify_result "${ROW}" "FAIL" "transfer audit runner exited ${run_rc}"
  exit $run_rc
fi
PROOF="proofs/perf/fusion_transfer_audit.json"
"${PYBIN}" - "${PROOF}" <<'PY'
import sys, json, os
p = sys.argv[1]
if not os.path.exists(p):
    print(f"FAIL: {p} not written"); sys.exit(1)
d = json.load(open(p))
ta = d.get("transfer_audit", {})
h2d = ta.get("host_to_device_bytes_post_init")
d2h = ta.get("device_to_host_bytes_post_init")
hlo = d.get("hlo_stats", {})
host_ops = sum(int(hlo.get(k, 0)) for k in ("copy_start", "outfeed", "infeed", "send", "recv"))
ok = (d2h == 0) and (h2d == 0)
print(f"post_init_D2H_bytes={d2h} post_init_H2D_bytes={h2d} "
      f"hlo_host_transfer_ops(copy-start/outfeed/infeed/send/recv)={host_ops} "
      f"fusion_instructions={hlo.get('fusion_instructions')}")
print("ASSERT", "PASS" if ok else "FAIL", "(zero post-init host<->device transfer inside the warmed loop)")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "zero post-init host<->device transfer (D2H=0, H2D=0) in the warmed timestep loop"
else
  verify_result "${ROW}" "FAIL" "non-zero host<->device transfer detected inside the timestep loop"
fi
exit $rc
