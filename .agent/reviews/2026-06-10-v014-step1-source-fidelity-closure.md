# Review: V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_OUTPUT_ALGEBRA`.

Production change is narrow: `rad_rk_tendf=1` source-leaf mode now carries MYNN `rqvblten` and applies WRF `conv_t_tendf_to_moist`; the default `rad_rk_tendf=0` branch is preserved.

The strict Step-1 proof does not close: after-conv residual max_abs `847.1446969755725`, rmse `9.627208432391289`.

Accepted remaining blocker: JAX MYNN source outputs remain below WRF at Step 1 because the surface-layer outputs feeding MYNN are still not WRF-compatible. `proofs/v014/mynn_driver_source_output_fix` already proved the MYNN kernel and fixed the missing WRF cold-start qke init; `proofs/v014/step1_sfclay_boundary_fix` ports WRF's sfclay_mynn first-call UST/QSFC/MOL/zol seed; and `proofs/v014/step1_tsk_znt_sourcing_fix` plus `proofs/v014/step1_thermo_column_inputs` now prove TSK/ZNT/MAVAIL and thermodynamic-column input parity at the exact sfclay_mynn hook. The surviving WRF-anchored blocker is strictly later surface-layer output algebra.

Next proof/fix route: DONE 2026-06-10: MYNN driver kernel/init, sfclay_mynn first-call semantics, TSK/ZNT/MAVAIL input sourcing, and sfclay thermodynamic column inputs are no longer active blockers. Next route: add a narrow WRF internal hook inside module_sf_mynn.F/SFCLAY1D_mynn for `thx/thgb/br/zol/psim/psih/ust/hfx/qfx`, then compare against surface_layer_with_diagnostics on the fixed input tuple.
