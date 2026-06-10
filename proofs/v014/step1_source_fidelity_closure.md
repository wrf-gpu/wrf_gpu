# V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_THERMODYNAMIC_COLUMN_INPUTS`.

## Evidence

- Strict WRF after `conv_t_tendf_to_moist` vs current JAX dry `T_TENDF`: max_abs `1497.6112467075195`, rmse `13.252694871222973`.
- JAX mass-coupled MYNN `RTHBLTEN` remains too weak: max_abs `1141.117233018646` vs WRF `2522.90576171875`.
- JAX mass-coupled MYNN qv source is also too weak: max_abs `0.555597268524649` vs WRF `QV_TEND` `0.4930315017700195`.
- Available same-boundary scalar inputs are not the order-10 error: T max_abs `5.788684885033035e-05`, QV max_abs `5.969281098756885e-08`, P max_abs `0.0390625`.
- Forcing radiation only moves max_abs to `1494.2972530323946`; held `RTHRATEN` is secondary.
- WRF qv/moist conversion is represented now and remains secondary: removing the WRF `QV_TEND` term would leave max_abs `231.55175407007107`.
- WRF oracle active sources close the accepted formula: max_abs `0.00016236981809925055`, rmse `8.089162788029723e-07`.

## Single Blocker

JAX MYNN source outputs remain below WRF at Step 1 because the surface boundary feeding MYNN is still not WRF-compatible. `proofs/v014/mynn_driver_source_output_fix` already proved the MYNN kernel and fixed the missing WRF cold-start qke init; `proofs/v014/step1_sfclay_boundary_fix` ports WRF's sfclay_mynn first-call UST/QSFC/MOL/zol seed; and `proofs/v014/step1_tsk_znt_sourcing_fix` now proves TSK/ZNT/MAVAIL source parity at the exact sfclay_mynn hook. The surviving WRF-anchored blocker is the non-surface thermodynamic column input entering sfclay_mynn.

## Fastest Next Route

DONE 2026-06-10: MYNN driver kernel/init, sfclay_mynn first-call semantics, and TSK/ZNT/MAVAIL input sourcing are no longer active blockers. Next route: localize the non-surface thermodynamic column inputs at the exact sfclay_mynn hook (`th_phy(kts)`, `t_phy(kts)`, `p_phy(kts)`, and `dz8w`) against JAX `_surface_column_view`; then fix the Step-1 temperature/pressure sourcing if local.

Proof objects: `proofs/v014/step1_source_fidelity_closure.json`.
