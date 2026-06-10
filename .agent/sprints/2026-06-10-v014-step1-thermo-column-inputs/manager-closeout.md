# Manager Closeout

Date: 2026-06-10 04:26 WEST
Manager: Codex
Worker: GPT-5.5 xhigh in tmux `0:3`

## Outcome

Closed as a valid local correctness fix and a successful narrowing sprint, not
as full Step-1 parity closure.

Verdict:

`THERMO_COLUMN_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`

The sprint fixed WRF `phy_prep` thermodynamic inputs at the `sfclay_mynn`
boundary and narrowed the remaining error to later MYNN surface-layer output
algebra.

## Proof Objects

- `proofs/v014/step1_thermo_column_inputs.py`
- `proofs/v014/step1_thermo_column_inputs.json`
- `proofs/v014/step1_thermo_column_inputs.md`
- `.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`

## Manager-Rerun Evidence

- Focused test: `2 passed`.
- `step1_thermo_column_inputs.py` reran successfully.
- `step1_tsk_znt_sourcing_fix.py` reran successfully and now reports
  `TSK_ZNT_THERMO_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.
- `step1_source_fidelity_closure.py` reran successfully and now reports
  `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_OUTPUT_ALGEBRA`.
- `mynn_driver_source_output_fix.py` reran successfully.
- JSON validation and `git diff --check` passed.

## Merge Decision:

Merge. This is a WRF-sourced, grid-backed surface input fix. It improves the
strict Step-1 residual but does not justify TOST, Switzerland, broad FP32, or
long GPU validation yet.

## Key Numbers

- Fixed `th_phy(kts)` max_abs `6.71089752017906e-05 K`.
- Fixed `t_phy(kts)` max_abs `0.013577942721781255 K`.
- Fixed hydrostatic `p_phy(kts)` max_abs `0.015625 Pa`.
- Fixed `dz8w(kts)` max_abs `0.00018988715282830526 m`.
- strict after-conv `T_TENDF` max_abs `847.1445725702908`, RMSE
  `9.56593990212596`.

## Next Sprint

Open a GPT-5.5 xhigh surface-layer-output sprint:

- add a narrow WRF internal hook inside `module_sf_mynn.F` / `SFCLAY1D_mynn`;
- capture `thx/thgb/br/zol/psim/psih/ust/hfx/qfx`;
- compare against `surface_layer_with_diagnostics` on the fixed input tuple;
- fix local algebra if proven;
- rerun strict Step-1.
