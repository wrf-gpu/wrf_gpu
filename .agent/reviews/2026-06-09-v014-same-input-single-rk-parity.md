# Review: V0.14 Same-Input Single-RK Parity

verdict: `SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.

objective: test the strict WRF pre-RK input -> one JAX RK step -> WRF post-RK/pre-halo boundary, or name the exact blocker without producing a weak same-input comparison.

files changed:
- `proofs/v014/same_input_single_rk_parity.py`
- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`

commands run:
- `python -m py_compile proofs/v014/same_input_single_rk_parity.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity.json >/tmp/same_input_single_rk_parity.validated.json`
- `git diff -- src`

proof objects produced:
- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`

result:
- `SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.
- The current pre-RK WRF hook emits only `MASS_K1` T/P/PB/MU/MUB fields.
- It does not emit full native U/V/W/PH state, full columns, JAX base `Tendencies`, WRF `DryPhysicsTendencies`, or the OperationalCarry/Namelist loader needed to feed `_rk_scan_step_with_pre_halo_capture`.
- The optional `post_final_calc_p_rho_phi` files are output surfaces with the same schema as the post-RK surface, not missing input/tendency surfaces.

unresolved risks:
- Once the missing WRF/JAX input wrapper exists, the 17x17 horizontal patch leaves only one conservative mass-grid score cell with an 8-cell halo; widen the hook if more scored cells are required.
- This proof does not decide upstream drift, final-RK PGF/mass-wind, or theta/source causality; it blocks the current instrumentation.

next decision needed: add the full pre-RK native-state plus RK-fixed tendency/source WRF hook and a proof-only JAX OperationalCarry loader, then rerun this same boundary.
