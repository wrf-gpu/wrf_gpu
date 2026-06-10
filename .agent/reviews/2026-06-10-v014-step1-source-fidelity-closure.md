# Review: V0.14 Step-1 Source-Fidelity Closure

Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.

Production change is narrow: `rad_rk_tendf=1` source-leaf mode now carries MYNN `rqvblten` and applies WRF `conv_t_tendf_to_moist`; the default `rad_rk_tendf=0` branch is preserved.

The strict Step-1 proof does not close: after-conv residual max_abs `1497.6112512148795`, rmse `13.468453371786723`.

Accepted remaining blocker: JAX MYNN source outputs remain below WRF at Step 1. Root-caused 2026-06-10 (proofs/v014/mynn_driver_source_output_fix): the order-10 deficit was the missing WRF mym_initialize level-2 equilibrium cold-start qke (now implemented); the remaining residual is the step-1 surface-layer flux boundary (ust/HFX/QFX + TSK/ZNT inputs + sfclayrev first-call semantics), bounded additionally by WRF's own uninitialized-rmol init UB at this boundary.

Next proof/fix route: DONE 2026-06-10: the WRF MYNN driver hook was emitted and compared (proofs/v014/mynn_driver_source_output_fix). Next route: emit a WRF step-1 surface-driver hook around sfclayrev (TSK/ZNT/UST/HFX/QFX in/out), port the sfclayrev first-call semantics and skin-temperature/roughness sourcing into the JAX surface adapter, then rerun the strict Step-1 proofs against the deterministic rmol-pinned WRF truth.
