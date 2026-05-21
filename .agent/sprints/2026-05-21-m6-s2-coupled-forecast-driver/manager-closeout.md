# M6-S2 Manager Closeout — Coupled Forecast Driver

**Sprint**: M6-S2 coupled forecast driver
**Status**: **CLOSED — Opus ACCEPT-WITH-MINOR-FOLLOWUPS; M6-S3..S8 UNBLOCKED**
**Date**: 2026-05-21 ~15:15
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

Codex worker (~2h05m wall):
- 1h+6h+24h coupled forecast on real d02 (160×67×45), Gen2 IC + M6-S2a boundary replay
- 0 H2D / 0 D2H / 0 transfer (MEASURED via JAX profiler trace + XLA compiled.memory_analysis())
- 23/23 M6 tests pass
- All 5 M6-S1 prereqs closed: R-3 FP32 Path A, R-5 real GridSpec dz, R-7 measured temp bytes, R-9 robust cadence, R-13 boundary State leaves
- WRF-style boundary apply (specified + relaxation zone per module_bc.F)
- ADR-010 amended with M6-S2 amendments block

## Opus reviewer 22 R-findings: 16 PASS, 6 PASS-WITH-FOLLOWUP/DISCLOSED

**Critical reviewer findings binding downstream sprints**:

- **R-16 (1s dycore cap)**: `dycore_dt_s = min(float(dt_s), 1.0)` is a permanent stability guard in this driver. Dynamics advances 1s per 60s coupled step → 60× mismatch. Honestly disclosed but binds M6-S5 (wall numbers cannot be used) + M6-S7 (must lift cap for real RMSE).
- **R-17 (finite-state guard)**: `sanitize_state` replaces non-finite + clips broad bounds. At 24h every prognostic clip fires at one or both ends. Residency proof, not physical validation. Binds M6-S4 (measure conservation BEFORE sanitize_state runs).
- **R-18 (mu_bdy first-step replay)**: Gen2 d02 tree exposed only `wrfinput_d02`, not `wrfout_d02_*` history; all 25 mu_bdy slots = time-zero MU+MUB. Binds M6-S3 (extend Gen2 accessor) + M6-S8 (restrict comparison or get real history).

## M6-S3 prereqs (F-S3-1/2/3)

1. Measure surface diagnostics with sanitize_state ON and OFF; report deltas. OR demonstrate Noah-MP drives State so sanitize_state never fires at 1h/6h/12h.
2. Extend Gen2 accessor to surface real `wrfout_d02_*` history for mu_bdy time series, OR document M6-S8 will be interior-only.
3. Surface adapter boundary cast at `physics_couplers.py:204-224` must stay FP64 for ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv per precision matrix.

## M6-S5 prereqs (F-S5-1/2/3)

1. Lift 1s dycore_dt_s cap before any speedup measurement. Stabilise M4 dycore for 60s coupled step OR accept smaller coupled dt (6-10s, more WRF-like for 3km).
2. Replace audit-extrapolated wall with observed end-to-end 24h forecast wall (302.77s including compile + output).
3. Bind one CPU denominator (3106 grid-points vs 4859 raw-timing) with documented rationale.

## M6 dispatch impact

- **M6-S3 + M6-S4 + M6-S5 + M6-S6 + M6-S7 + M6-S8: ALL UNBLOCKED** for sprint dispatch
- **Tier-4 RMSE operational acceptance: BLOCKED** until M5-S3.zzzz + M5-S3.zzzzz close (SW + LW broadband)
- **ADR-007 4× verdict: BLOCKED** until M6-S5 closes per the 3 prereqs above

Manager dispatches M6-S3 worker now in parallel with M5-S3.zzzzz LW (per separate decision).

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 15:15
