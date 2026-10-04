# F6 Transaction Audit Summary

Generated: 2026-05-28T19:55:44.214195+00:00
Device: {'jax_platforms_env': None, 'jax_platform_name_env': None, 'devices': [{'id': 'cuda:0', 'platform': 'gpu'}], 'gpu_visible': True, 'cpu_state_allocation_patch': False}

## Toggle Results
- a (physics_off + boundary_off + guards_off): first critical violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; acoustic uv max delta = 0.000e+00.
- b (physics_on + boundary_off + guards_off): first critical violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; acoustic uv max delta = 0.000e+00.
- c (physics_off + boundary_on + guards_off): first critical violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; acoustic uv max delta = 0.000e+00.
- d (physics_off + boundary_off + guards_on): first critical violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; first algebraic violation = step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds; acoustic uv max delta = 0.000e+00.

## Questions
1. Where does the blow-up actually start? In the pure dycore combination, the first critical violation is step 1, RK1, substep 1, advance_mu_t, theta_sanity_bounds. That means the failure is dycore-internal before physics, boundary replay, or guards are needed.
2. Does it match F3 Opus? Mostly yes. Missing `advance_uv` is supported by zero u/v delta inside acoustic substeps; cross-stage saved-state loss is hit by `rk_saved_state_theta_1`; `muts`/work-mu inconsistency is hit; the pressure bound is hit within this 12-step algebraic threshold.
3. Does it match agy? It matches the original agy direction on restored advection tests, theta_1 reference testing, and mu-save/mass tracking. It extends agy's three-bug diagnosis by separating pure-dycore, physics, boundary, and limiter toggles.
4. Most targeted next fix scope: do not patch physics or boundary first if combination a has the critical failure. The narrow next code sprint should repair RK/acoustic state ownership: carry the RK1 saved family across RK2/RK3, add/validate small-step `advance_uv`, and replace the pressure stub only after the save-family invariants are clean.

## Proof Objects
- `proofs/f6/audit_combination_a.json`
- `proofs/f6/audit_combination_b.json`
- `proofs/f6/audit_combination_c.json`
- `proofs/f6/audit_combination_d.json`
- `proofs/f6/invariant_violations.json`

F6_AUDIT_SUMMARY_COMPLETE
