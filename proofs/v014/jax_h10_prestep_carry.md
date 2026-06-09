# V0.14 H10 Pre-Step Carry Checkpoint

Verdict: `CHECKPOINT_BLOCKED_NO_H10_PRESTEP_CARRY`.

## Target

- WRF target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
- Domain/step: `d02`, step `6000`, valid `2026-05-02T04:00:00+00:00`.
- Required JAX pre-step checkpoint: completed step `5999`.

## Checkpoint Probe

- Candidates inspected: `6`.
- Usable h10 pre-step candidates: `0`.
- Real same-surface comparison run: `False`.

## Blocker

- Reason: `NO_CPU_LOADABLE_JAX_H10_PRESTEP_OPERATIONAL_CARRY`.
- Missing input/API: A CPU-loadable d02 OperationalCarry with paired OperationalNamelist/grid at completed step_index=5999, immediately before WRF/JAX step 6000. The checkpoint must include State, promoted carry leaves t_2ave/ww/mudf/muave/muts/ph_tend/u_save/v_save/w_save/t_save/ph_save/mu_save/ww_save/rthraten, active physics carry leaves, boundary leaves, real d02 metrics, tendencies, and boundary_config.
- Next command: `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src WRFGPU2_H10_PRESTEP_CARRY=/abs/path/to/d02_step5999_full_carry.pkl python proofs/v014/jax_h10_prestep_carry.py`
