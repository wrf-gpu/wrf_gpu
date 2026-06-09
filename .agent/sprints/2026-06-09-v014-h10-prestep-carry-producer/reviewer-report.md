# Reviewer Report

Decision: accept with the narrower next-step condition below.

The sprint satisfied its contract: it produced a full CPU-loadable
`OperationalCarry` checkpoint at the requested completed step and used it to run
the pre-halo comparison against the established WRF target. No production
`src/` files or WRF source files were edited.

Evidence reviewed:

- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry.md`

Accepted findings:

- Checkpoint path:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`
- Checkpoint status: `LOAD_OK`
- Runtime state type: `OperationalCarry`
- Namelist type: `OperationalNamelist`
- Step index: `5999`
- Comparison status: `RAN`
- Verdict: `JAX_MISMATCH_T`
- First mismatch: `T`, max_abs `3.3545763228707983`, RMSE
  `1.0296598586362888`, worst native key `[12, 17]`

Review caveat:

The worker's suggested next decision says "source-fix sprint". That is too
broad. The WRF target proved that history `T` comes from `grid%th_phy_m_t0`,
while other WRF THM-side arrays can mislead this comparison. The first follow-up
must attribute the JAX `T` source/history mapping and compare available JAX
theta/history candidates before a production dycore source fix is allowed.

Residual risk:

The patch comparison is not a full-grid validation and does not by itself prove
the root cause for `P/PB/U/V/W/PH/MU/MUB`. It proves only that the same-surface
JAX-vs-WRF comparison is now unblocked and currently fails first on `T`.
