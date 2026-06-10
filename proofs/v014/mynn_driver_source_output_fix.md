# V0.14 MYNN Driver Source-Output Fix

Verdict: `MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`.

## Root cause and fix

- The order-10-weak JAX MYNN Step-1 sources were a MISSING WRF first-call
  turbulence initialization: `mym_initialize` level-2 equilibrium qke
  (WRF post-init qke max `25.000001907` vs the old uniform
  taper seed `4.99220118e-05`).
- Implemented in production: `mynn_pbl.mynn_coldstart_init_columns` +
  `d02_replay` cold-start seed; focused tests pass.

## WRF-anchored evidence (disposable Step-1 MYNN driver hook)

- Kernel response (WRF inputs + WRF init qke): strong-cell ratio median `0.9982`,
  corr `1.0000`, rmse `2.6e-06` vs raw WRF `RTHBLTEN` — the JAX MYNN kernel
  reproduces WRF's driver source output at the same boundary.
- Init formula oracle (WRF ust + emitted rmol): rel p50 `2.38e-07`, rmse `0.00306`.
- WRF init-UB proven: emitted init `rmol` equals the PREVIOUS column's line-879 value for
  `10494/10494` columns (uninitialized local).
- Deterministic rmol-pinned WRF truth: production formula (rmol=0, WRF ust) vs pinned WRF init qke:
  rel p50 `2.38e-07`, rmse `0.00184`, max_abs `0.676`.

## Attribution (strong-cell ratio median / corr, theta source)

- A qke=WRF, inputs=WRF: `0.9982` / `1.0000`
- B qke=prod, inputs=WRF: `0.7177` / `0.9927`
- C qke=WRF, inputs=JAX: `0.2681` / `0.8181`
- D qke=prod, inputs=JAX (production): `0.2354` / `0.8488`

The dominant remaining residual is the step-1 surface-layer flux boundary
(C/D), not the MYNN kernel or its turbulence init (A/B).

## Strict Step-1 metric (vs existing part2 truth)

- prior: max_abs `2457.578397008898`, rmse `21.364579991779515`
- now: max_abs `1497.6112512148795`, rmse `13.468453371786723`
- Note: the existing truth embeds WRF's uninitialized-rmol init, so strict
  closure against it is bounded by the proven UB envelope.

## Single remaining blocker

Step-1 surface-layer flux boundary: the JAX step-1 sfclay outputs feeding MYNN differ from WRF's (ustar bias -0.077/max 0.176; HFX rmse 24.6 W/m^2; QFX bias -2.1e-5), driven by (a) land skin-temperature input differences up to 8.3 K, (b) roughness-length differences up to 0.97 m, and (c) sfclayrev FIRST-CALL semantics (JAX starts from ustar=0 while WRF iterates from a first guess: identical-input ocean columns still show 4x ustar deficits). With WRF fluxes substituted, the production init already reaches strong-cell ratio 0.72/corr 0.993 (case B), and with WRF init qke too it reaches 1.00 (case A).

## Fastest next route

One sprint: emit a WRF step-1 surface-driver hook (same disposable pattern) around module_sf_mynn/sfclayrev for TSK/ZNT/UST/HFX/QFX in/out, port the sfclayrev first-call (flag_iter/UST first-guess) semantics + skin-temperature/roughness sourcing into the JAX surface adapter, and gate on case-D converging to case-B levels; then rerun the strict Step-1 proofs against the rmol-pinned WRF truth.

Proof objects: `proofs/v014/mynn_driver_source_output_fix.json`.
