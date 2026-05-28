# Root Cause Analysis - M11.2 Dycore Theta Increment

Verdict: root cause not fully isolated; `M11P2_PARTIAL`.

## What the current worktree shows

The sprint contract starts from `proofs/m17/diagnostic_report_after_fix.json`, where `theta_in_bounds` first fails at step 141 after `dycore_rk3`. Re-running the 1h harness on this worktree produced an earlier dycore failure:

- first invariant break: `wind_in_bounds`, step 72, operator `dycore_rk3`
- first nonfinite: step 93, operator `dycore_rk3`
- `theta_in_bounds`: step 85, operator `dycore_rk3`

This means the current branch is not in the same failure regime as the M17 premise; wind/mass instability appears before the theta-only target gate can be satisfied.

## Lines investigated

1. `src/gpuwrf/dynamics/core/acoustic.py`, `_decouple_theta_after_advance`
   - Current line uses `state.theta` in the numerator.
   - WRF reference: `dyn_em/module_small_step_em.F:408-413` reconstructs `t_2` with saved `t_save`; `small_step_prep` stores that saved value at `:259-264`.
   - Test: changed numerator to `state.theta_1`. Result: 1h harness regressed to `wind_in_bounds` step 72 and nonfinite dycore cells at step 93. Reverted.

2. `src/gpuwrf/dynamics/mu_t_advance.py`, `mu_tendency`
   - Current line uses `-dmdt + mu_tend`.
   - WRF reference: `dyn_em/module_small_step_em.F:1099-1105` updates `MU` with `DMDT + MU_TEND`.
   - Test: changed to `dmdt + mu_tend`. Result: limiter still clipped all 360 first-hour steps and mass residual became huge. Reverted.

3. `src/gpuwrf/dynamics/mu_t_advance.py`, theta horizontal flux terms
   - Current lines use `inputs.u`/`inputs.v` in the theta flux divergence.
   - WRF reference: `dyn_em/module_small_step_em.F:238-254` mass-couples small-step `u_2`/`v_2`; `:1162-1171` then uses those small-step fields in theta flux.
   - Tests: mass-coupled and perturbation-only theta flux variants were tried. They either introduced earlier wind/mass failure or left limiter activity at 360/360 first-hour steps. Reverted.

## Conclusion

The exact theta-increment line remains unresolved. The best-supported diagnosis is that the acoustic theta path is coupled to an already-unstable dycore mass/wind update in the current worktree; the current first failure is not theta-only. A follow-up sprint should first restore the M17 failure regime or directly localize the step-72 wind/mu blow-up before continuing theta-increment repair.

