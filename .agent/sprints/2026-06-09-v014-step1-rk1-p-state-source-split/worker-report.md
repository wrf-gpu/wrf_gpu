# Worker Report

Summary: produced the proof-only RK1 `P_STATE` source split and refuted the
fresh post-Mythos `P_STATE` material-source hypothesis as a stale proof-loader
artifact.

Objective:

- Test whether the RK1 `after_rk_addtend_before_small_step_prep` material
  `P_STATE` residual was still a production source bug after the Mythos
  live-nest/start-domain init fix.

Files changed:

- `proofs/v014/step1_rk1_p_state_source_split.py`
- `proofs/v014/step1_rk1_p_state_source_split.json`
- `proofs/v014/step1_rk1_p_state_source_split.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

Commands run:

- `python -m py_compile proofs/v014/step1_rk1_p_state_source_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_p_state_source_split.py`
- `python -m json.tool proofs/v014/step1_rk1_p_state_source_split.json >/tmp/step1_rk1_p_state_source_split.validated.json`
- `git diff --check`

Proof objects:

- `proofs/v014/step1_rk1_p_state_source_split.json`
- `proofs/v014/step1_rk1_p_state_source_split.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

Result:

- Verdict:
  `STEP1_RK1_P_STATE_SOURCE_REFUTED_STALE_PROOF_LOADER_BYPASS_NEXT_T_TENDF`.
- Stale proof loader `P_STATE` max_abs: `69.96875 Pa`.
- Patched Mythos init capture `P_STATE` max_abs: `0.0390625 Pa`.
- Gate: `1.0 Pa`.
- No production source edit; no GPU used.

Next:

- Split WRF `first_rk_step_part2` `T_TENDF` and RK1
  `T_TEND/PH_TEND/RW_TEND` against JAX `compute_advection_tendencies` and
  `_augment_large_step_tendencies` using a patched-init capture.
