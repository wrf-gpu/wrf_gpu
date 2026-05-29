# F6 Transaction Audit Summary

Generated: 2026-05-29T00:22:58.726559+00:00
Device: {'jax_platforms_env': None, 'jax_platform_name_env': None, 'devices': [{'id': 'cuda:0', 'platform': 'gpu'}], 'gpu_visible': True, 'cpu_state_allocation_patch': False}

## Toggle Results
- a (physics_off + boundary_off + guards_off): first critical violation = none in first 12 steps; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_mass_residual; acoustic uv max delta = 5.312e+05.
- b (physics_on + boundary_off + guards_off): first critical violation = none in first 12 steps; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_mass_residual; acoustic uv max delta = 1.304e+04.
- c (physics_off + boundary_on + guards_off): first critical violation = none in first 12 steps; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_mass_residual; acoustic uv max delta = 5.312e+05.
- d (physics_off + boundary_off + guards_on): first critical violation = none in first 12 steps; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_mass_residual; acoustic uv max delta = 5.311e+05.

## Questions
1. Where does the blow-up actually start? No finite/dry-mass/pressure critical violation appears in the pure dycore combination during the first 12 steps. The earliest algebraic defect is step 1, RK1, substep 1, advance_mu_t, theta_mass_residual.
2. Does it match F3 Opus? Mostly yes. the previous zero-u/v signal is cleared by nonzero acoustic u/v deltas; cross-stage saved-state loss is not hit by `rk_saved_state_theta_1`; `muts`/work-mu inconsistency is hit somewhere in the 12-step audit; the pressure bound is not hit within this 12-step algebraic threshold.
3. Does it match agy? It matches the original agy direction on restored advection tests, theta_1 reference testing, and mu-save/mass tracking. It extends agy's three-bug diagnosis by separating pure-dycore, physics, boundary, and limiter toggles.
4. Most targeted next fix scope: do not patch physics or boundary first if combination a has the critical failure. The narrow next code sprint should repair RK/acoustic state ownership: carry the RK1 saved family across RK2/RK3, add/validate small-step `advance_uv`, and replace the pressure stub only after the save-family invariants are clean.

## Proof Objects
- `proofs/f7a2/audit_combination_a.json`
- `proofs/f7a2/audit_combination_b.json`
- `proofs/f7a2/audit_combination_c.json`
- `proofs/f7a2/audit_combination_d.json`
- `proofs/f7a2/invariant_violations.json`

F6_AUDIT_SUMMARY_COMPLETE
