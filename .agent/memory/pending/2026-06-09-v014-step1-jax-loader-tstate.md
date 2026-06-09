# V0.14 Step-1 JAX Loader T-State

Date: 2026-06-09 15:58 WEST

Sprint closed:
`.agent/sprints/2026-06-09-v014-step1-jax-loader-tstate`.

Proof: `proofs/v014/step1_jax_loader_tstate.*`, verdict
`STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

Known ruled out:

- WRF solve_em pre-call `T_STATE` mutation: max_abs `0.0` from
  `after_step_increment` to `before_first_rk_step_part1_call`.
- WRF `first_rk_step_part1` as the source: prior part1 proof shows no material
  internal `T_STATE` mutation.
- Full-vs-perturbation theta mapping: WRF `grid%t_2` maps to
  `State.theta - 300 K`.

New facts:

- `T_STATE` max_abs versus WRF pre-call stays `5.490173101425171` for raw,
  live, boundary-packaged, carry, and haloed step-entry states.
- `T_STATE` transition max_abs raw->live, live->boundary, boundary->carry, and
  carry->halo are all `0.0`.
- `PB` improves from raw max_abs `2627.3828125` to live max_abs
  `0.05357326504599769`; `PHB/MUB` similarly close.
- The residual is not boundary-only: haloed step-entry interior max_abs is
  `5.490173101425171`, boundary-band max_abs is `5.284271240234375`.

Conclusion:

JAX live-nest base initialization updates `PB/PHB/MUB` but carries raw
`wrfinput_d02` theta unchanged. Boundary package, initial carry, and halo are
ruled out for this `T_STATE` residual. The next target is WRF
`med_nest_initial` / `start_domain_em` `t_2` semantics and an initialization-only
GPU/JAX fix if the candidate formula closes WRF pre-call truth.

Do not resume TOST, Switzerland, FP32, memory source work, or GPU validation
until this grid-field divergence stage is explained or fixed.
