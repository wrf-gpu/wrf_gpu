# Tester Report

Decision: validation gates passed.

Commands run by the worker and rerun/checked by the manager:

- `python -m py_compile proofs/v014/step1_part2_source_leaves_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/step1_part2_source_leaves_split.validated.json`
- `git diff --check`

The proof run regenerated all requested artifacts and returned:

`STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`.

Key measured checks:

- `update_phy_ten` formula versus WRF active RTH: nested-interior max_abs `0.0`.
- `conv_t_tendf_to_moist` formula: nested-interior max_abs
  `0.00016236981809925055`.
- post-conversion versus `after_first_rk_step_part2`: nested-interior max_abs
  `0.0`.
- current JAX dry `T_TENDF` residual: max_abs `2457.5830078125`, RMSE
  `21.674279301376934`.
- source-save sparse `T_TENDF` versus current JAX dry: max_abs
  `1326.432250976562`, RMSE `97.71886125389001`.

No GPU, TOST, Switzerland, FP32, memory source, or Fable/Mythos path was used.
