# Worker Report — M7 Profiler-Window Fix

Summary: PASS. The profiler window now builds the 20260521 replay case before `cudaProfilerStart`, the audit script can fall back to the `jit_run_forecast_operational` XLA module window, and the recaptured 360-step trace reports `counts.d2h_inter_kernel_inside_window == 0`.

## Files Changed

- `scripts/m7_gpu_profile_1h.py`
- `scripts/m7_d2h_audit.py`
- `tests/test_m7_profiler_window.py`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/nsys_capture_call_log_v2.json`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_existing_xla_fallback.json`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/m7_profile_verdict_update.md`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/command_outputs/*`
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/worker-report.md`

## Commands Run + Output

Full stdout/stderr is captured under `.agent/sprints/2026-05-27-m7-profiler-window-fix/command_outputs/`.

- `python -m py_compile scripts/m7_gpu_profile_1h.py scripts/m7_d2h_audit.py tests/test_m7_profiler_window.py`
  - exit 0; stdout empty; stderr empty.
- `pytest -q tests/test_m7_profiler_window.py`
  - exit 0; stdout: `.. [100%] 2 passed in 0.03s`; stderr empty.
- `python scripts/m7_d2h_audit.py /tmp/m7_profile_artifacts/m7_20260521_warm_360.nsys-rep --sqlite /tmp/m7_profile_artifacts/m7_20260521_warm_360.sqlite --output .../d2h_audit_existing_xla_fallback.json --marker-regex DOES_NOT_EXIST`
  - exit 0; output status `PASS`, method `xla_module_nvtx`, provenance `xla-module-fallback`, `d2h_inter_kernel_inside_window=0`.
- `nsys profile --force-overwrite=true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output=/tmp/m7_profile_artifacts/m7_20260521_warm_360_v2 taskset -c 0-3 python scripts/m7_gpu_profile_1h.py profile-window --run-key 20260521 --steps 360 --warmups 2 --cuda-profiler-range --output .../nsys_capture_call_log_v2.json`
  - exit 0; stdout included `Capture range started in the application`, `Capture range ended in the application`, and generated `/tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.nsys-rep`.
  - call log output: `status=PASS`, `profile_wall_s=6.593829017999269`, warmups `[104.14050613600011, 6.020811981999941]`; stderr empty.
- `python scripts/m7_d2h_audit.py /tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.nsys-rep --sqlite /tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.sqlite --output .../d2h_audit_v2.json --marker-regex m7_profile_window`
  - exit 0; output status `PASS`, method `nvtx_marker`, provenance `explicit-marker`.
  - counts: `d2h_total_trace=25`, `d2h_pre_kernel_inside_window=25`, `d2h_inter_kernel_inside_window=0`, `h2d_inside_loop_window=0`.
  - bytes: `d2h_inter_kernel_inside_window=0`, `h2d_inside_loop_window=0`; stderr empty.
- `taskset -c 0-3 python scripts/m7_gpu_profile_1h.py reproducibility --run-key 20260521 --hours 1.0 --runs 3 --output .../reproducibility_v2.json`
  - exit 0; output status `PASS`, samples `[5.738986880999619, 5.696859701999529, 5.6816901279999]`, mean `5.7058455703330155`, CV `0.0042480879014396965`; stderr empty.
- `python -m json.tool .../d2h_audit_v2.json` and `python -m json.tool .../reproducibility_v2.json`
  - both exit 0; formatted JSON emitted to command-output stdout files; stderr empty.
- `git diff --check`
  - exit 0; stdout empty; stderr empty.

## Proof Objects

- Raw recapture: `/tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.nsys-rep`
- SQLite export: `/tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.sqlite`
- AC2 fallback proof: `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_existing_xla_fallback.json`
- AC3 D2H proof: `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`
- AC4 reproducibility proof: `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json`
- AC5 verdict flip memo: `.agent/sprints/2026-05-27-m7-profiler-window-fix/m7_profile_verdict_update.md`

## Risks

- The explicit `m7_profile_window` range still contains 25 pre-kernel D2H copies totaling 932448 bytes; these occur before the first captured kernel. The hard invariant in this sprint is the inter-kernel/in-loop path, which is zero copies and zero bytes. The XLA-module fallback also reports zero D2H inside `jit_run_forecast_operational`.
- NCU permission work remains out of scope per contract.
- I did not remote-push because the sprint contract hard rule says "No remote push"; this branch is committed locally.

## Handoff

- objective: narrow the M7 profiler window, add audit fallback windowing, recapture the D2H proof, and update the M7 profile verdict.
- files changed: listed above; no `src/gpuwrf/**`, governance, predecessor reports, tester reports, reviewer reports, manager closeout, or memory-patch files were touched.
- commands run: listed above; stdout/stderr captured in `command_outputs/`.
- proof objects produced: listed above.
- unresolved risks: pre-kernel D2H distinction must remain explicit in manager closeout; NCU permissions remain unresolved from the predecessor sprint.
- next decision needed: manager can apply `m7_profile_verdict_update.md` during integration to flip the M7 profile D2H result from `BLOCKED-D2H` to `PASS-D2H`.
