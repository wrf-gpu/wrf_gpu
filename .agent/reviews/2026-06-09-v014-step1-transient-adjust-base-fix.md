# Review: V0.14 Step-1 Transient Adjust-Base MUB Fix

Verdict: `STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

Findings:
- HIGH: The new helper `_wrf_live_nest_transient_adjust_mub` transcribes WRF `med_nest_initial`'s `copy mub->mub_save; blend_terrain(mub_fine,mub)`, exposing the transient post-blend current MUB that `adjust_tempqv` consumes. The final post-`start_domain` BaseState is untouched (additive diff only).
- HIGH: Transient adjust-base MUB matches the WRF adjust hook at the target cell within `4.521e-04` Pa.
- HIGH: Final BaseState MUB still matches the WRF pre-part1 final target within `4.648e-03` Pa.
- MEDIUM: Corrected theta max_abs `5.788684885033035e-05` K vs prior `0.00541785382188209` K (closure ratio `93.593863364203`) against same-boundary WRF pre-call truth.

Evidence:
- WRF adjust hook: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- Same-boundary WRF pre-call truth: `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`
- Source helper: `gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub`

objective: implement the smallest production-source fix for the Step-1 transient adjust-base MUB mismatch.

files changed:
- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_transient_adjust_base_fix.py`
- `proofs/v014/step1_transient_adjust_base_fix.json`
- `proofs/v014/step1_transient_adjust_base_fix.md`
- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`

commands run:
- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_transient_adjust_base_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_transient_adjust_base_fix.py`
- `python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json >/tmp/step1_transient_adjust_base_fix.validated.json`
- `git diff --stat`

unresolved risks:
- The corrected theta/QV candidate is validated in this CPU proof; wiring theta_m+adjust_tempqv into the production live-nest init consumer is a separate, larger grid-parity step.

next decision needed: Wire WRF theta_m conversion + adjust_tempqv (with the transient adjust-base MUB) into the production live-nest init consumer of _apply_live_nest_base_init, then run the next larger grid-parity step: the full step-1 same-input d02 comparison (step1_live_nest_init_rerun) across all 16 fields.
