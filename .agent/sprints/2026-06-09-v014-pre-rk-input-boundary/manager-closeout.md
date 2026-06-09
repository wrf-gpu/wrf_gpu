# Manager Closeout

Merge Decision: accept and land the proof artifacts.

Objective:

Produce explicit WRF and JAX step-6000 pre-RK input-boundary truth for h10 d02
over `T/P/PB/MU/MUB`, then decide whether the produced JAX step-5999 prestep
carry is already wrong before current-step physics/RK.

Accepted verdict:

`PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.

Accepted evidence:

- `proofs/v014/pre_rk_input_boundary.py`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`

Manager validation:

- Python compilation.
- CPU-only proof rerun.
- JSON validation.
- dmpar CPU-WRF hook run outside the worker sandbox.
- Process check confirming no WRF/MPI process remained.

Key finding:

The produced JAX h10 step-5999 carry is already wrong at the explicit WRF
step-6000 pre-RK input boundary. The first mismatch is `T` with max_abs
`6.218735851548047` and RMSE `4.638818160588427`; `P/PB/MU/MUB` are also
divergent before current-step physics/RK.

Roadmap effect:

Do not start with current-step RK/acoustic, final `small_step_finish`,
post-RK refresh, or history-source remapping. The next debug sprint must trace
the JAX checkpoint/prestep carry producer and previous-step WRF/JAX update path.

Next decision:

Identify whether the first wrong state is introduced by checkpoint extraction,
prestep carry load, previous-step final carry assembly, boundary/tendency
packaging, or earlier integration.
