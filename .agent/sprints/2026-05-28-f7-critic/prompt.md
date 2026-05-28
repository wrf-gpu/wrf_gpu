# F7 Plan Critic — Bounded Rewrite Strategy

**Worker**: codex gpt-5.5 xhigh (critic mode)
**Wall-time**: 2-4 hours (analysis only)
**No code changes.**

## Context

After F3 (Opus arch review), F5 (WRF cadence spec), F6 (12-step transaction audit) all converged, manager is launching **F7 — bounded dycore rewrite** in 4 sub-sprints:

- **F7.A** (in flight): `small_step_prep_wrf` + `small_step_finish_wrf` cross-RK save carry + `advance_uv_wrf`
- **F7.B** (next): `calc_p_rho_wrf` + `advance_w_wrf` with full RHS + ph_tend
- **F7.C** (later): `rk_addtend_dry` + WRF flux-form mass-coupled advection
- **F7.D** (last): Scalar flux accumulation + scalar tendency cadence

Total estimated scope: ~1000-1300 LOC. Total estimated wall-time: 5-8 days code + 2-3 days validation.

## What I need from you

Read these files NOW with care:

1. `.agent/sprints/2026-05-28-f7a-save-family-and-advance-uv/sprint-contract.md` — the F7.A contract
2. `proofs/f5/wrf_cadence_spec.md` — the 12-item gap map driving F7 design
3. `.agent/sprints/2026-05-28-f3-agy-architecture-followup/findings.md` — Opus arch review
4. `proofs/f6/audit_summary.md` — F6 audit
5. `proofs/f6/invariant_violations.json` — current failure pattern
6. `src/gpuwrf/runtime/operational_mode.py` (current `_rk_scan_step`, `_with_save_family`, `_acoustic_scan`)
7. `src/gpuwrf/dynamics/core/acoustic.py` (current `acoustic_substep_core`)
8. `src/gpuwrf/dynamics/mu_t_advance.py` (keep-able kernel)

Then argue against the strategy — be honest, no diplomatic hedging.

### Question 1 — Is the F7.A → F7.B → F7.C → F7.D sequence right?

F7.A fixes save-family + adds advance_uv. F7.B fixes calc_p_rho + advance_w. F7.C adds rk_addtend_dry + flux-form advection. F7.D adds scalar flux accumulation.

Should any step move earlier? E.g. is `advance_w_wrf` actually a blocker for `advance_uv_wrf` (because the implicit p propagation might be needed for consistent acoustic substeps)? Or is `rk_addtend_dry` cheap enough to fold into F7.A?

### Question 2 — Is bounded better than one large sprint?

Manager chose 4 bounded sprints over one mega-sprint to reduce review burden + enable parallel critic passes. Is this the right call, or does it over-engineer? A single ~1000 LOC sprint could ship faster if reviewer is competent. What's your honest verdict?

### Question 3 — Is the F7.A contract overscoped/underscoped?

The F7.A contract claims 2-3 day wall-time for `small_step_prep_wrf` + `small_step_finish_wrf` + `advance_uv_wrf` + verification. Is this realistic? Is the AC list right? Is anything critical missing? Is anything bloated?

### Question 4 — Are we missing a cheaper first move?

Is there a 1-day fix that would unblock the pure-dycore step-1 failure WITHOUT the full F7.A scope? E.g., could we hack `mu_save` carry inline in `_rk_scan_step` without writing new module files, just to see if the step-1 invariant passes? That would tell us in hours, not days, whether the architecture diagnosis is right.

### Question 5 — Honest verdict on the F7 plan

Score the F7 plan 0-10 with one-sentence rationale. If <7, what would you do differently?

## Deliverable

Write `.agent/sprints/2026-05-28-f7-critic/critique.md` with answers to Q1-Q5.

End with `F7_CRITIQUE_COMPLETE`.

## Hard rules

- CPU pinning: `taskset -c 0-3`.
- No model code changes — analysis only.
- No remote push.
- Manager repo only.
- Auto-notify on exit: `tmux send-keys -t 0:0 "AGENT REPORT: f7-critic DONE exit=$?" Enter`.
