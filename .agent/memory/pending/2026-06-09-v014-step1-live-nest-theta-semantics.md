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

Do not resume TOST, Switzerland, FP32, memory source work, or GPU validation
until this live-nest theta semantics gap is fixed or proven not to close the
grid divergence.
