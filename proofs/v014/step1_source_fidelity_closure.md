# V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_OUTPUT_ALGEBRA`.

## Evidence

- Strict WRF after `conv_t_tendf_to_moist` vs current JAX dry `T_TENDF`: max_abs `847.1445725702908`, rmse `9.56593990212596`.
- JAX mass-coupled MYNN `RTHBLTEN` remains too weak: max_abs `2299.1812825746474` vs WRF `2522.90576171875`.
- JAX mass-coupled MYNN qv source is also too weak: max_abs `0.8961410964058757` vs WRF `QV_TEND` `0.4930315017700195`.
- Available same-boundary scalar inputs are not the order-10 error: T max_abs `5.788684885033035e-05`, QV max_abs `5.969281098756885e-08`, P max_abs `0.0390625`.
- Forcing radiation only moves max_abs to `851.1710005443492`; held `RTHRATEN` is secondary.
- WRF qv/moist conversion is represented now and remains secondary: removing the WRF `QV_TEND` term would leave max_abs `231.55175407007107`.
- WRF oracle active sources close the accepted formula: max_abs `0.00016236981809925055`, rmse `8.089162788029723e-07`.

## Single Blocker

JAX MYNN source outputs remain below WRF at Step 1 because the surface-layer outputs feeding MYNN are still not WRF-compatible. `proofs/v014/mynn_driver_source_output_fix` already proved the MYNN kernel and fixed the missing WRF cold-start qke init; `proofs/v014/step1_sfclay_boundary_fix` ports WRF's sfclay_mynn first-call UST/QSFC/MOL/zol seed; and `proofs/v014/step1_tsk_znt_sourcing_fix` plus `proofs/v014/step1_thermo_column_inputs` now prove TSK/ZNT/MAVAIL and thermodynamic-column input parity at the exact sfclay_mynn hook. The surviving WRF-anchored blocker is strictly later surface-layer output algebra.

## Fastest Next Route

DONE 2026-06-10: MYNN driver kernel/init, sfclay_mynn first-call semantics, TSK/ZNT/MAVAIL input sourcing, and sfclay thermodynamic column inputs are no longer active blockers. Next route: add a narrow WRF internal hook inside module_sf_mynn.F/SFCLAY1D_mynn for `thx/thgb/br/zol/psim/psih/ust/hfx/qfx`, then compare against surface_layer_with_diagnostics on the fixed input tuple.

Proof objects: `proofs/v014/step1_source_fidelity_closure.json`.
