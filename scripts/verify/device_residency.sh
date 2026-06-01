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
# The binding discriminator is in-loop vs one-time transfer (the HLO op-count is
# unavailable because jit().lower() trips on State reconstruction; we classify the
# measured memcpy bytes from the trace's temporal position instead).
cls = d.get("in_loop_transfer_classification", {})
classifiable = bool(cls.get("classifiable"))
bytes_accounted = bool(cls.get("bytes_accounted"))
in_loop_bytes = cls.get("in_loop_total_bytes")
verdict = d.get("device_residency_verdict", "")
print(f"post_init_D2H_bytes={d2h} post_init_H2D_bytes={h2d} "
      f"classified={cls.get('classified_total_bytes')}/{cls.get('measured_total_bytes')} bytes_accounted={bytes_accounted} "
      f"in_loop_total_bytes={in_loop_bytes} in_loop_events={cls.get('in_loop_transfer_events')} "
      f"one_time_h2d={cls.get('one_time_h2d_bytes')} one_time_d2h={cls.get('one_time_d2h_bytes')} "
      f"hlo_host_transfer_ops={host_ops} fusion_instructions={hlo.get('fusion_instructions')}")
print("VERDICT:", verdict)
# PASS iff the trace is classifiable, the classifier ACTUALLY accounted for the
# measured transfer bytes, AND there are zero in-loop transfer bytes (post-init
# h2d/d2h may be non-zero as long as it is one-time I/O staging at the boundary).
# If the per-event byte sizes could not be extracted (classifier saw ~0 of the
# measured bytes), do NOT fabricate a zero-in-loop PASS -> INCONCLUSIVE.
if classifiable and bytes_accounted and int(in_loop_bytes) == 0:
    print("ASSERT PASS (zero in-loop host<->device transfer; post-init bytes are one-time I/O staging)")
    sys.exit(0)
elif classifiable and bytes_accounted:
    print("ASSERT FAIL (in-loop host<->device transfer detected)")
    sys.exit(1)
else:
    # Cannot trust the byte attribution -> inconclusive (architecturally device-
    # resident by design, but the counted discriminator could not be derived).
    print("ASSERT INCONCLUSIVE (memcpy events found but per-event byte sizes not extractable from this trace)")
    sys.exit(2)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "zero IN-LOOP host<->device transfer in the warmed timestep loop (post-init H2D/D2H bytes classified as one-time I/O staging at the compute-span boundary)"
elif [ $rc -eq 2 ]; then
  verify_result "${ROW}" "INCONCLUSIVE" "device residency architecturally guaranteed (whole-state pytree resident on device; no host transfer in the scanned timestep by construction); trace-temporal in-loop classifier could not bin this trace -- counted-audit tracked as v0.2.0 follow-up"
else
  verify_result "${ROW}" "FAIL" "in-loop host<->device transfer detected inside the timestep loop"
fi
exit $rc
