# M7 Profile Verdict Update

Summary: PASS-D2H replaces the predecessor `BLOCKED-D2H` verdict for the M7 profile window.

The predecessor profile window started the CUDA profiler before replay-case construction, so setup copies from Gen2 `wrfout` loading were included in the audit range. This sprint moved replay-case construction outside the CUDA profiler range and reran the 20260521 warm 360-step profile.

Proof:

- Raw trace: `/tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.nsys-rep`
- Audit: `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`
- Result: `status=PASS`, `method=nvtx_marker`, `window_provenance=explicit-marker`, `counts.d2h_inter_kernel_inside_window=0`, `bytes.d2h_inter_kernel_inside_window=0`
- Reproducibility: `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json`, CV `0.0042480879014396965` <= `0.05`

Apply during merge by keeping the predecessor wall-clock evidence but replacing the D2H verdict with this corrected window/audit result. Do not rewrite predecessor sprint reports.
