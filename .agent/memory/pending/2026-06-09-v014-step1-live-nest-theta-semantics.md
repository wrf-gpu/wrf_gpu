# V0.14 Step-1 Live-Nest Theta Semantics

Date: 2026-06-09 16:03 WEST

Active sprint:
`.agent/sprints/2026-06-09-v014-step1-live-nest-theta-semantics`.

Trigger proof: `proofs/v014/step1_jax_loader_tstate.*`, verdict
`STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

The previous sprint ruled out boundary package, initial carry, halo, and physics
for the current Step-1 `T_STATE` residual. JAX live-nest base init closes
`PB/PHB/MUB` but keeps raw `wrfinput_d02` theta unchanged.

WRF source target:

- `share/mediation_integrate.F`: after `blend_terrain` of `ht`, `mub`, and
  `phb`, WRF calls `adjust_tempqv(nest%mub, nest%mub_save, nest%c3h, nest%c4h,
  nest%znw, nest%p_top, nest%t_2, nest%p, QVAPOR, use_theta_m, ...)`.
- `dyn_em/nest_init_utils.F::adjust_tempqv`: computes old/new pressure from
  saved/current `MUB`, conserves relative humidity, updates `th`/`t_2` and
  `qv`.

Closed evidence:

- `proofs/v014/step1_live_nest_theta_semantics.json` verdict:
  `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.
- Direct `adjust_tempqv` on raw dry NetCDF `T` does not close `T_STATE`.
- For `USE_THETA_M=1`, WRF solve-time `grid%t_2` uses moist-theta semantics.
- Dry-to-moist theta conversion plus `adjust_tempqv` reduces `T_STATE` max_abs
  from `5.490173101425171` to `0.00541785382188209`, RMSE
  `5.068868142015466e-05`, but this remains above the prior `1e-3 K` material
  gate.
- No production source patch is allowed yet because accepted same-boundary WRF
  pre-call `QVAPOR` truth is missing.

Do not resume TOST, Switzerland, FP32, memory source work, or GPU validation
until the same-boundary `QVAPOR` savepoint exists and the live-nest theta
semantics gap is fixed or proven not to close the grid divergence.
