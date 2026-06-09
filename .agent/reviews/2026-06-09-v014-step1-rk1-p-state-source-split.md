# Review: V0.14 Step-1 RK1 P-State Source Split

Verdict: `STEP1_RK1_P_STATE_SOURCE_REFUTED_STALE_PROOF_LOADER_BYPASS_NEXT_T_TENDF`.

objective: localize or fix the post-Mythos Step-1 RK1 material `P_STATE` divergence before acoustic substeps.

files changed:
- `proofs/v014/step1_rk1_p_state_source_split.py`
- `proofs/v014/step1_rk1_p_state_source_split.json`
- `proofs/v014/step1_rk1_p_state_source_split.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

commands run:
- `python -m py_compile proofs/v014/step1_rk1_p_state_source_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_p_state_source_split.py`
- `python -m json.tool proofs/v014/step1_rk1_p_state_source_split.json >/tmp/step1_rk1_p_state_source_split.validated.json`
- `git diff --check`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_rk1_p_state_source_split.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_rk1_p_state_source_split.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

unresolved risks:
- The older proof-local live.build_live_nest_step1_inputs helper still bypasses the production perturbation init; this proof patches the capture locally rather than editing shared proof helpers.
- The WRF truth surface does not split internal rk_tendency before/after every tendency component, so the remaining T/PH/RW tendency family needs one focused tendency-contract split.
- The proof does not enter acoustic substeps because RK1 P_STATE is below material gate before small_step_prep.

next decision needed: Split WRF first_rk_step_part2 T_TENDF and then RK1 after_rk_addtend T_TEND/PH_TEND/RW_TEND against JAX compute_advection_tendencies/_augment_large_step_tendencies with a patched-init capture. Do not enter acoustic substeps for this P_STATE issue; P_STATE is below material gate before small_step_prep.
