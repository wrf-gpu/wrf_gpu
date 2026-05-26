# Sprint Contract — M7 Profiler-Window Fix + D2H Recapture

**Sprint ID**: `2026-05-27-m7-profiler-window-fix`
**Created**: 2026-05-27 (autonomous overnight loop)
**Status**: READY
**Predecessors**:
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/` (BLOCKED-D2H, profile measurement complete)
- `.agent/sprints/2026-05-27-m7-d2h-probe-codex/` (root cause: profiler window includes pre-forecast setup; D2H during forecast = 0)
- `.agent/sprints/2026-05-27-m7-d2h-probe-opus/` (S1/S2/S3 theoretical risks logged for future fusion sprints)

## Objective

The M7 GPU profile sprint self-declared `BLOCKED-D2H` based on 89 D2H copies / 102 MB inside the audit window. The codex probe proved the audit window included pre-forecast replay-case loading (initial state from Gen2 wrfout disk files); D2H during the actual `jit_run_forecast_operational` is **0 copies, 0 bytes**.

This sprint applies the two surgical fixes localized by the codex probe, recaptures a clean Nsight Systems trace, and produces the corrected proof object that flips `BLOCKED-D2H` → `PASS-D2H` for M7.

## Acceptance

- **AC1 — Profiler-window placement fix**: in `scripts/m7_gpu_profile_1h.py:243-245`, split case construction from the profiled section. Pseudocode:
  ```python
  # BEFORE: case construction inside profiler range
  with TraceAnnotation(...), cuda_profiler():
      profile_wall_s, meta = _timed_forecast(run_key, hours, annotation)
  # AFTER: build case, then narrow profiler range
  state, namelist, meta = _build_case(run_key)
  with TraceAnnotation(...), cuda_profiler():
      profile_wall_s = _run_forecast(state, namelist, hours)
  ```
  Preserve the `m7_profile_window` NVTX marker; emit it via `nvtx.annotate` if not already present.

- **AC2 — Audit-script marker fallback**: in `scripts/m7_d2h_audit.py`, when `m7_profile_window` marker is absent, derive the loop window from the NVTX `XlaModule` label matching `jit_run_forecast_operational`. Only fall back to first-to-last kernel if both markers absent, and tag the result with `window_provenance: "broad-fallback"` so the audit JSON explicitly distinguishes broad vs in-loop counts.

- **AC3 — Recapture clean trace**: rerun the warm 360-step 1h profile on 20260521. Capture artifact at `/tmp/m7_profile_artifacts/m7_20260521_warm_360_v2.nsys-rep`. Emit `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` with `counts.d2h_inter_kernel_inside_window == 0`.

- **AC4 — Reproducibility re-check**: 3 warm reruns at the new profiler placement; CV ≤ 5%. Emit `reproducibility_v2.json`.

- **AC5 — M7 profile sprint verdict flip**: write `.agent/sprints/2026-05-27-m7-profiler-window-fix/m7_profile_verdict_update.md` summarizing what changed and why — to be applied to the M7 profile sprint's verdict during the merge step. Do NOT modify the predecessor sprint's worker-report.md directly.

- **AC6 — Worker report** with verdict `PASS` if AC3 confirms 0 in-forecast D2H, else `BLOCKED_RECAPTURE` with diagnosis.

## Files Worker May Modify

- `scripts/m7_gpu_profile_1h.py` — narrow the profiler range only; no logic change to wall-clock measurement
- `scripts/m7_d2h_audit.py` — fallback window selection only; do not change cluster-attribution logic
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/**`
- `tests/test_m7_profiler_window.py` (NEW — pin the fix with a unit test for window selection logic)

## Files Worker Must Not Modify

- `src/gpuwrf/**` — measurement-script-only sprint, no model code change
- governance files
- predecessor sprint reports (worker-report.md of m7-gpu-profile-prep, m7-d2h-probe-*)
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **No model code changes.** Audit script + profiler driver only.
2. **No new test files outside the explicit AC6 unit test.**
3. **CPU pinning**: `taskset -c 0-3`.
4. **Do not interfere with tmux `0:1`** (nightly WRF).
5. **No remote push.** Local commit on `worker/gpt/m7-profiler-window-fix` only.
6. **Preserve M7 wall-clock measurements**: the predecessor's 5.7 s warm number must remain reproducible with the new profiler placement.
7. **NCU permissions are out of scope** — the predecessor sprint hit ERR_NVGPUCTRPERM; do not attempt to fix that here.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 1-3 h
- Branch: `worker/gpt/m7-profiler-window-fix`
- Worktree: `/tmp/wrf_gpu2_winfix`

## What this enables

After this sprint:
- M7 wall-clock claim is defensible (D2H invariant holds — proven, not just measured-without-violation)
- The M7 profile + this fix sprint merge together to close the M7 perf-measurement step
- Next M7 sub-sprint: 1km readiness + memory audit (per M7-canary-operational-v0.md)
- The opus S1/S2/S3 theoretical risks become OPTIONAL future fusion sprints (no longer blockers)
