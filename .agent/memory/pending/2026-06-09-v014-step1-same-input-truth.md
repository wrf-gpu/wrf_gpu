# V0.14 Step-1 Same-Input Truth

Date: 2026-06-09

Verdict:
`STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.

What changed:

- A full-domain CPU-WRF d02 step-1 `post_after_all_rk_steps_pre_halo` truth npz
  now exists at
  `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.
- `proofs/v014/step1_same_input_truth.py` runs the accepted same-input
  comparison: WRF step-1 truth versus JAX one-step
  `_rk_scan_step_with_pre_halo_capture(...).pre_halo_state`.
- The forbidden weak comparison against the JAX initial state was avoided.
- No production `src/gpuwrf/**` source changed.

Result:

- First divergent schema field: `T`.
- Dominant residuals are base/mass:
  - `MUB` max_abs `2635.640625`, RMSE `98.13000038547803`
  - `PB` max_abs `2627.3828125`, RMSE `47.826296821589736`
  - `PHB` max_abs `2237.9423828125`, RMSE `45.35253861292826`
  - `P` max_abs `1561.1123921205437`, RMSE `305.75054216524205`

Manager note:

The next sprint should target native live-nest child base-state initialization
or a decisive init-override falsifier, then rerun
`proofs/v014/step1_same_input_truth.py`. Do not resume TOST, Switzerland, FP32,
or memory cleanup while this step-1 full-domain grid mismatch remains.
