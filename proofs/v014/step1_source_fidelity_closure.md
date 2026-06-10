# V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.

## Evidence

- Strict WRF after `conv_t_tendf_to_moist` vs current JAX dry `T_TENDF`: max_abs `1497.6112512148795`, rmse `13.468453371786723`.
- JAX mass-coupled MYNN `RTHBLTEN` remains too weak: max_abs `1118.1877178980255` vs WRF `2522.90576171875`.
- JAX mass-coupled MYNN qv source is also too weak: max_abs `0.5918027936475765` vs WRF `QV_TEND` `0.4930315017700195`.
- Available same-boundary scalar inputs are not the order-10 error: T max_abs `5.788684885033035e-05`, QV max_abs `5.969281098756885e-08`, P max_abs `0.0390625`.
- Forcing radiation only moves max_abs to `1494.2972624261442`; held `RTHRATEN` is secondary.
- WRF qv/moist conversion is represented now and remains secondary: removing the WRF `QV_TEND` term would leave max_abs `231.55175407007107`.
- WRF oracle active sources close the accepted formula: max_abs `0.00016236981809925055`, rmse `8.089162788029723e-07`.

## Single Blocker

JAX MYNN source outputs remain below WRF at Step 1. Root-caused 2026-06-10 (proofs/v014/mynn_driver_source_output_fix): the order-10 deficit was the missing WRF mym_initialize level-2 equilibrium cold-start qke (now implemented); the remaining residual is the step-1 surface-layer flux boundary (ust/HFX/QFX + TSK/ZNT inputs + sfclayrev first-call semantics), bounded additionally by WRF's own uninitialized-rmol init UB at this boundary.

## Fastest Next Route

DONE 2026-06-10: the WRF MYNN driver hook was emitted and compared (proofs/v014/mynn_driver_source_output_fix). Next route: emit a WRF step-1 surface-driver hook around sfclayrev (TSK/ZNT/UST/HFX/QFX in/out), port the sfclayrev first-call semantics and skin-temperature/roughness sourcing into the JAX surface adapter, then rerun the strict Step-1 proofs against the deterministic rmol-pinned WRF truth.

Proof objects: `proofs/v014/step1_source_fidelity_closure.json`.
