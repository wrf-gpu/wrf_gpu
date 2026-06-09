# V0.14 Step-1 Same-Input Truth

Verdict: `STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.

## Result

- WRF truth status: `TRUTH_NPZ_READY_EXISTING`.
- Truth NPZ: `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.
- Strict JAX pre-halo comparison run: `True`.
- First divergent field in schema order: `T`.
- Largest max_abs field: `MUB` max_abs `2635.640625` rmse `98.13000038547803`.

Detailed per-field metrics are in `proofs/v014/step1_same_input_truth.json`.
