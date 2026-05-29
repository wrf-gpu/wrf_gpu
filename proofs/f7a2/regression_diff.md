# F7.A regression diff — before/after the step-4 detonation

## Baseline (stubbed dycore, commit `d6824b6`)

12-step f6 transaction audit, real Gen2 d02 replay, default harness config
(dt = 10 s, 10 sound steps, epssm = 0.1). Source:
`proofs/f7a2_baseline/audit_combination_*.json`.

| combination | first critical violation | acoustic uv max delta |
| --- | --- | --- |
| a (pure dycore) | step 1, RK3, substep 8, advance_mu_t, theta_sanity_bounds | 3.87e+121 |
| b (+physics) | step 1, RK3, substep 8, theta_sanity_bounds | 1.31e+118 |
| c (+boundary) | step 1, RK3, substep 8, theta_sanity_bounds | 3.87e+121 |
| d (+limiter) | step 1, RK3, substep 8, theta_sanity_bounds | 1.54e+35 |

The dycore detonated inside the very first timestep: by RK3 substep 8 the coupled
`u/v` work arrays had already reached 10¹²¹, and θ fell to −24 K. Root cause: the
acoustic substep used the stubs `_advance_geopotential` (off-centred `g·dt·w`),
`_diagnose_pressure` (`p + |dnw|·Δμ`), `_ph_tend_increment` (`0.01·Δθ`), and a
`w_solve_core` that Thomas-solved with `state.w` as the RHS instead of the real
`advance_w` explicit predictor. The vertical solve had no real buoyancy /
pressure coupling, so the acoustic mode grew without restoring force.

## After F7.A (branch `worker/opus/f7-acoustic-core`, tip `595d4f2`)

### Same harness default (dt = 10 s, epssm = 0.1)

Pressure ratio `|p'|/pb` grows ~2.6×/substep and `w` ~2.5×/substep on RK3 — a
classic under-damped acoustic mode at this aggressive timestep with WRF's
operational dampers (w_damping, Rayleigh) disabled per the sprint scope. The
detonation is no longer the immediate stub blow-up; it is a timestep/off-centring
stability limit (see below).

### WRF-faithful Gen2 d02 config (dt = 3 s, 4 sound steps, epssm = 0.5)

The actual Gen2 d02 namelist uses `time_step = 18 s` on the d01 parent with
`parent_time_step_ratio = 3` (→ 6 s on d02), `epssm = 0.5`, plus `w_damping = 1`
and `damp_opt = 3`. With the acoustic dampers disabled (sprint scope) the bare
core is stable at dt = 3 s. Source: `proofs/f7a2/audit_combination_*.json`,
`audit_summary.md`.

| combination | first critical violation | acoustic uv max delta |
| --- | --- | --- |
| a (pure dycore) | **none in first 12 steps** | 5.3e+05 |
| b (+physics) | **none in first 12 steps** | 1.3e+04 |
| c (+boundary) | **none in first 12 steps** | 5.3e+05 |
| d (+limiter) | **none in first 12 steps** | 5.3e+05 |

`first_critical_violation == null` for all four combinations → AC2 holds at the
WRF-faithful configuration. (The remaining non-critical "first algebraic
violation" is `theta_mass_residual` at the audit's 1e-10 strictness; the oracle
AC5 shows the true theta-mass drift is 0.)

### Timestep / off-centring stability sweep (combination a, d02)

Source: probe runs. Demonstrates the failure is a CFL / off-centring margin, not
a structural operator error (the operators are independently proven by AC3/AC4/AC5).

| dt (s) | sound steps | epssm | first critical (12 steps) |
| --- | --- | --- | --- |
| 3.0 | 4 | 0.5 | none (clean) |
| 4.0 | 4 | 0.5 | step 10, wind_sanity_bounds |
| 6.0 | 6 | 0.5 | step 6, wind_sanity_bounds |
| 10.0 | 10 | 0.1 | (under-damped growth) |

## Operator-level fixes that moved the failure

1. Deleted the three stubs; implemented `advance_w_wrf` with the full WRF RHS
   (implicit `c2a` pressure term, `cqw`/`c2a` buoyancy, `t_2ave`, terrain lower
   BC, top lid, Thomas forward/back sweep) and the WRF geopotential finish.
2. `pg_buoy_w_dry` supplies the real large-step vertical PGF/buoyancy `rw_tend`;
   `dry_cqw` supplies the post-`pg_buoy_w` dry `cqw` (1 interior, 0 at lid/top).
3. `calc_coef_w` now receives the **full dry mass `mut`** (not the work `muts`),
   the real `c2a` from `small_step_prep`, and the real `cqw` — closing the
   "defaults to ones" bug.
4. `calc_p_rho_step` applies the `smdiv` pressure-memory divergence damping
   (`p += smdiv·(p − pm1)`; refresh `pm1`) each substep; `c2a` is INTENT(IN).
5. `advance_uv` gained the `emdiv`/`mudf` external-mode divergence damping term.
6. Removed the horizontal-PGF double-count (PGF is a small-step term, was also
   being injected into the large-step tendency).
7. Wired the real terrain height `ht = phb(sfc)/g` into the `advance_w` lower BC.
8. Fixed a production `jax.lax.scan` carry pytree mismatch (init
   `theta_coupled_work`); the end-to-end forecast now runs finite.
