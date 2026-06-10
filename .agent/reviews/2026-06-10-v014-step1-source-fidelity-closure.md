# Review: V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_THERMODYNAMIC_COLUMN_INPUTS`.

Production change is narrow: `rad_rk_tendf=1` source-leaf mode now carries MYNN `rqvblten` and applies WRF `conv_t_tendf_to_moist`; the default `rad_rk_tendf=0` branch is preserved.

The strict Step-1 proof does not close: after-conv residual max_abs `1497.6112467075195`, rmse `13.252694871222973`.

Accepted remaining blocker: JAX MYNN source outputs remain below WRF at Step 1 because the surface boundary feeding MYNN is still not WRF-compatible. `proofs/v014/mynn_driver_source_output_fix` already proved the MYNN kernel and fixed the missing WRF cold-start qke init; `proofs/v014/step1_sfclay_boundary_fix` ports WRF's sfclay_mynn first-call UST/QSFC/MOL/zol seed; and `proofs/v014/step1_tsk_znt_sourcing_fix` now proves TSK/ZNT/MAVAIL source parity at the exact sfclay_mynn hook. The surviving WRF-anchored blocker is the non-surface thermodynamic column input entering sfclay_mynn.

Next proof/fix route: DONE 2026-06-10: MYNN driver kernel/init, sfclay_mynn first-call semantics, and TSK/ZNT/MAVAIL input sourcing are no longer active blockers. Next route: localize the non-surface thermodynamic column inputs at the exact sfclay_mynn hook (`th_phy(kts)`, `t_phy(kts)`, `p_phy(kts)`, and `dz8w`) against JAX `_surface_column_view`; then fix the Step-1 temperature/pressure sourcing if local.
