# Sprint Contract — M17 (PROMOTED EARLY): Thompson microphysics silent-failure fix

**Sprint ID**: `2026-05-28-m17-microphysics-silent-failure-fix`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m17-microphysics-silent-failure-fix`
**Worktree**: `/tmp/wrf_gpu2_m17`
**Wall-time**: 6-12 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`
**Promoted from Phase E to inline with Phase B per diagnostic-harness finding (2026-05-28)**

## Why this sprint exists now (was originally M17 later in roadmap)

The comprehensive diagnostic harness (Opus subagent deliverable `3d7372a`, smoke proof `proofs/diagnostic_harness/diagnostic_report_smoke_3step.json`) ran a 3-step smoke on the operational forecast loop and uncovered a **SILENT FAILURE**:

> **`microphysics_thompson: NOISY_ZERO (6/7 expected fields flat: qr, qc, qg, qs, qi, qv)`**

Thompson microphysics is wired into the operator sequence in `_physics_boundary_step`, but produces **zero increment on every moisture variable**. The code runs but does nothing. This is the most likely root cause of:
- M12's failure to move T2 RMSE (T2 RMSE 10.80 K → 10.80 K despite correct surface flux formula): if qv is not evolving, LH cannot reach correct values regardless of formula.
- The persistent 10.80 K T2 RMSE that all three of M11/M12/M13 could not move alone.
- The "QVAPOR healthy" verdict from the wrfout-level comparator (M9.C reported QVAPOR small RMSE 0.0027) was misleading — small RMSE is consistent with **qv held nearly constant by silent microphysics**, not with microphysics being correct.

The harness also found `dycore_rk3` partially silent on `p_perturbation` and `ph_perturbation`. That's a separate finding (M11 scope or new sub-sprint). This contract focuses on microphysics alone.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/diagnostic_harness/diagnostic_report_smoke_3step.json` — the smoking gun
2. `src/gpuwrf/diagnostics/comprehensive_harness.py` — instrumentation source (read; use to verify fix)
3. `src/gpuwrf/coupling/physics_couplers.py` — Thompson adapter location
4. `src/gpuwrf/runtime/operational_mode.py` — `_physics_boundary_step` operator sequence
5. `tests/savepoint/test_diagnostic_harness.py` — harness pytest entry point
6. M9.C confirmed: theta convention is fine, surface_layer is ACTIVE, MYNN is ACTIVE, lateral BC is ACTIVE; microphysics is the anomaly.
7. WRF Thompson reference: `phys/module_mp_thompson.F` if present (read-only) — for sign conventions, output tendency vs absolute, hydrometeor advection order

## Acceptance

### AC1 — Root cause identified + documented

`.agent/sprints/2026-05-28-m17-microphysics-silent-failure-fix/root_cause_analysis.md` with:
- Exact line(s) in `src/gpuwrf/coupling/physics_couplers.py` where the silent failure originates
- Whether the cause is (a) function returns early, (b) inputs not connected to state, (c) outputs not written back to state, (d) jit-cached stale path, (e) tendency-vs-absolute confusion, or (f) other
- Evidence: how the diagnostic identified it (specifically what delta = 0 looks like in the harness output)

### AC2 — Fix applied

`src/gpuwrf/coupling/physics_couplers.py` Thompson adapter section: the silent failure is fixed. The fix is **minimal** — do NOT refactor unrelated code. If a structural change is needed (e.g. plumbing tendency-vs-absolute through to the state update), document why and keep the diff focused.

### AC3 — Harness re-verification

Re-run `scripts/run_diagnostic_harness.py` with a SHORT (e.g. 1h Canary 20260521) horizon. The output `proofs/m17/diagnostic_report_after_fix.json` must show:
- `microphysics_thompson` verdict is `ACTIVE` (not `NOISY_ZERO`)
- At least 4 of the 7 moisture fields (qv, qr, qc, qg, qs, qi, qni-or-equivalent) show non-zero delta over the run
- No other operator regresses (boundary_guards, surface_layer, mynn_pbl, lateral_boundary remain ACTIVE or PASSIVE_OK as before)

### AC4 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC5 — 24h skill diff

Re-run Canary 20260521 24h with the fix. Emit `proofs/m17/post_m17_skill_diff.json`. Acceptance is **directional, not absolute**:
- T2 RMSE drops by ≥ 20 % vs post_iter2_skill_diff baseline (10.80 K → ≤ 8.6 K)
- QVAPOR RMSE may *increase* (it was small because qv was frozen; now it evolves — that's a SUCCESS signal, not regression)
- HFX/LH may also change — document direction

### AC6 — Worker report

`.agent/sprints/2026-05-28-m17-microphysics-silent-failure-fix/worker-report.md` standard format. Verdict `M17_COMPLETE` if AC1-AC5 all delivered; `M17_PARTIAL` with explicit unfinished gaps otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`. Coordinate with parallel m11 worker (also using GPU). If OOM, retry once with `JAX_PLATFORMS=cpu` for the diagnosis runs.
3. **Files writable**: `src/gpuwrf/coupling/physics_couplers.py` (Thompson section only — don't touch surface/MYNN/RRTMG functions M12/M13 just modified), `proofs/m17/**`, `.agent/sprints/2026-05-28-m17-microphysics-silent-failure-fix/**`.
4. **Files NOT writable**: `src/gpuwrf/diagnostics/comprehensive_harness.py` (Opus deliverable, USE it for verification), `src/gpuwrf/runtime/operational_mode.py`, dycore, BC, state contracts, governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m17 DONE exit=$?" Enter`.
8. **End with verdict**: `M17_COMPLETE` / `M17_PARTIAL` + headline T2 RMSE reduction percentage (or "diagnosis-only").
