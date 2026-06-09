# V0.14 Live-Nest Base Hook

Verdict: `NATIVE_PORT_PLAN_READY`.

## Summary

- No production JAX source was patched.
- CPU-WRF `wrfout_h0` is used only as validation oracle.
- Next source fix should port WRF live-nest parent interpolation plus `blend_terrain`, then run `start_domain_em` base recomputation natively.
- Native wrfinput vs CPU h0 target-patch max deltas: HGT `89.50347900390625` m, PB `1047.015625` Pa, MUB `1050.3046875` Pa.
- WRF base formula on CPU h0 HGT residuals: PB `0.04889917548280209` Pa, MUB `0.044447155625675805` Pa, PHB `0.09328280997578986` m2/s2.

## Source Hook

- `share/mediation_integrate.F`: live nest calls `med_interp_domain`, reads child input, blends `ht/mub/phb`, then adjusts state.
- `inc/nest_interpdown_interp.inc`: generated calls interpolate parent `PHB/MUB/PB/HT` into child arrays via `interp_fcn`.
- `dyn_em/nest_init_utils.F::blend_terrain`: parent strip, blend zone, and child interior formula.
- `dyn_em/start_em.F::start_domain_em`: recomputes `PB/MUB/PHB/T_INIT/ALB` before perturbation fields are recalculated.

Full line ranges and stats are in `proofs/v014/live_nest_base_hook.json`.
