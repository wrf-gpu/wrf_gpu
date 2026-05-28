# F7A Regression Diff

Generated: 2026-05-28

## Inputs

- Baseline: `proofs/f6/invariant_violations.json`
- Candidate: `proofs/f7a/invariant_violations.json`
- Audit command: `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f7a`

## Combination A Result

| Signal | F6 baseline | F7A candidate |
| --- | --- | --- |
| Acoustic u/v max delta | `0.000e+00` | `3.873e+121` |
| First critical violation | step 1, RK1, substep 1, `advance_mu_t`, `theta_sanity_bounds` | step 1, RK3, substep 8, `advance_mu_t`, `theta_sanity_bounds` |
| First `muts_mut_work_mu_consistency` | step 1, RK1, substep 1, `acoustic_substep_commit`, max error `1.603e+01` | step 2, RK2, substep 1, `advance_mu_t`, max error `2.441e-04` |
| `rk_saved_state_theta_1` | step 1, RK2, `rk_stage_candidate`, max error `3.745e-01` | not hit in 12-step audit |
| First pressure bound | step 9, RK3, `rk_stage_candidate`, ratio `4.765e+01` | step 2, RK1, `rk_stage_candidate`, ratio `1.136e+11` |

## Verdict

F7A clears the original F6 pure-dycore first critical point: `theta_sanity_bounds` no longer fires at step 1/RK1/substep 1, and the same earliest position no longer has `muts_mut_work_mu_consistency`. The RK2/RK3 `_1` saved-state invariant is also clean in the 12-step audit.

This is not a stable dycore yet. The first remaining critical violation is still within step 1, now at RK3/substep 8, and the pressure bound worsens after the loop-entry pressure work. That is consistent with the known F7.B scope: full `advance_w_wrf`, geopotential RHS, and `calc_p_rho(step=iteration)`.
