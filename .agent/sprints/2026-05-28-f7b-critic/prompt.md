# F7.B Plan Critic — Contract Review Before Worker Commits

**Worker**: codex gpt-5.5 xhigh (critic mode)
**Wall-time**: 2-4 hours (analysis only)
**No code changes.**

## Context

F7.A is complete: cross-RK `_1` family carry, `advance_uv_wrf`, loop-entry `calc_p_rho(step=0)`. First critical violation moved from step 1/RK1/sub1 → step 1/RK3/sub8 BUT acoustic u/v magnitude is 3.873e+121 (active but unstable — expected because `advance_w_wrf` is still a stub).

F7.B is dispatched in parallel with this critique. The contract is at `.agent/sprints/2026-05-28-f7b-advance-w-and-calc-p-rho-iteration/sprint-contract.md`.

## What I need from you

Read these files NOW with care:

1. `.agent/sprints/2026-05-28-f7b-advance-w-and-calc-p-rho-iteration/sprint-contract.md` — the F7.B contract you are critiquing
2. `proofs/f5/wrf_cadence_spec.md` — the binding cadence spec
3. `.agent/sprints/2026-05-28-f7a-save-family-and-advance-uv/worker-report.md` — what F7.A landed
4. `proofs/f7a/audit_summary.md` + `proofs/f7a/invariant_violations.json` — current state
5. `.agent/sprints/2026-05-28-f7-critic/critique.md` — the F7-critic methodology lessons (apply them here)
6. WRF Fortran:
   - `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:1178-1584` (advance_w)
   - `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:492-563` (calc_p_rho)
7. Current JAX stubs being replaced:
   - `src/gpuwrf/dynamics/core/acoustic.py` (look for `_advance_geopotential`, `_ph_tend_increment`, `_diagnose_pressure`)
   - `src/gpuwrf/dynamics/core/calc_p_rho.py` (F7.A's step=0)

## Questions to answer

### Q1 — Is the `advance_w_wrf` scope right?

The contract says advance_w includes "rw_tend large-step tendency, vertical PGF perturbation, buoyancy from theta+qv, divergence damping via c2a, terrain lower boundary contribution". Is that complete? Is anything bloated? Specifically: does the contract risk pulling in `rk_addtend_dry` work that should stay in F7.C?

### Q2 — Is `calc_p_rho(step=iteration)` correctly placed AFTER `advance_w` in the substep order?

WRF order per `solve_em.F:3088, 3398, 3837, 4164`: `advance_uv → advance_mu_t → advance_w → calc_p_rho(step=iteration)`. Is the contract honoring this? Are there any subtle ordering issues with how `c2a` and `p_pert` flow into the next substep's `advance_uv`?

### Q3 — Is AC4 (hardened gates) strict enough?

The contract requires:
- first critical later than step 1/RK3/sub8
- acoustic u/v in [0, 10] m/s

Is this the right gate? Could a wrong `advance_w` pass these but fail downstream physics? What's the cheapest additional check?

### Q4 — Should `advance_w_wrf` and `calc_p_rho(step=iteration)` be in ONE sprint or split?

Manager bundled them because they are intimately coupled (advance_w consumes c2a, calc_p_rho updates c2a). But scope is large. Split into F7.B.1 (calc_p_rho iter only) + F7.B.2 (advance_w)? Or keep bundled?

### Q5 — Wall-time and risk

3-5 days realistic? What's the failure mode if F7.B regresses worse than F7.A? Is there a rollback plan?

### Q6 — Honest verdict

Score 0-10. If <7, what would you change before the worker commits?

## Deliverable

Write `.agent/sprints/2026-05-28-f7b-critic/critique.md` with answers to Q1-Q6.

End with `F7B_CRITIQUE_COMPLETE`.

## Hard rules

- CPU pinning: `taskset -c 0-3`.
- No model code changes — analysis only.
- No remote push.
- Manager repo only.
- **DO NOT include tmux send-keys in your work plan** — manager's bash wrapper handles notify on exit.
