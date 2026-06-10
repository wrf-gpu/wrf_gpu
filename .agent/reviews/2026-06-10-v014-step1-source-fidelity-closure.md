# Review: V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_TSK_ZNT_INPUTS`.

Production change is narrow: `rad_rk_tendf=1` source-leaf mode now carries MYNN `rqvblten` and applies WRF `conv_t_tendf_to_moist`; the default `rad_rk_tendf=0` branch is preserved.

The strict Step-1 proof does not close: after-conv residual max_abs `1497.6112467075195`, rmse `13.296448784742802`.

Accepted remaining blocker: JAX MYNN source outputs remain below WRF at Step 1 because the surface boundary feeding MYNN is still not WRF-compatible. `proofs/v014/mynn_driver_source_output_fix` already proved the MYNN kernel and fixed the missing WRF cold-start qke init; `proofs/v014/step1_sfclay_boundary_fix` now ports WRF's sfclay_mynn first-call UST/QSFC/MOL/zol seed and narrows the surviving blocker to WRF-anchored TSK/ZNT surface input sourcing.

Next proof/fix route: DONE 2026-06-10: MYNN driver kernel/init and sfclay_mynn first-call semantics are no longer the active blocker. Next route: emit a tiny WRF surface-driver hook around module_surface_driver/module_sf_mynn for incoming TSK/ZNT/UST/QSFC/MOL and outgoing UST/HFX/QFX/ZNT on the current d02 Step-1 case; compare those exact arrays against JAX `_surface_column_view` inputs and diagnostics; fix TSK/ZNT sourcing if confirmed.
