# Manager Closeout

Merge Decision: accept and land the producer proof artifacts.

Objective:

Produce a full JAX `OperationalCarry` checkpoint for `d02` completed step 5999,
then rerun the pre-halo hook comparison against Boole's green WRF
`post_after_all_rk_steps_pre_halo` target.

Accepted verdict:

`JAX_MISMATCH_T`.

Accepted evidence:

- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`
- `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

Manager validation:

- JSON validation for producer and canonical h10 compare outputs.
- Python compilation for both proof scripts.
- CPU-only canonical h10 compare rerun using
  `WRFGPU2_H10_PRESTEP_CARRY=/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- Process/GPU check after terminating a redundant second producer process.

Roadmap effect:

The previous blocker `CHECKPOINT_BLOCKED_NO_H10_PRESTEP_CARRY` is closed. The
current blocker is a same-surface JAX-vs-WRF mismatch whose first contract-order
failure is `T`.

Next decision:

Open a T history/source-attribution sprint before any production numerical fix.
That sprint must compare JAX theta/history candidates against WRF
`T_HIST_SRC`/`grid%th_phy_m_t0` and WRF THM-side candidates, then decide whether
the first failure is a source/cadence mapping issue or a real dycore theta
evolution issue.
