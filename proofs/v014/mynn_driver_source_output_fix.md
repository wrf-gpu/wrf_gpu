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
- C qke=WRF, inputs=JAX: `0.2695` / `0.8131`
- D qke=prod, inputs=JAX (production): `0.2365` / `0.8314`

The dominant remaining residual is the step-1 surface-layer flux boundary
(C/D), not the MYNN kernel or its turbulence init (A/B).

## Strict Step-1 metric (vs existing part2 truth)

- prior: max_abs `2457.578397008898`, rmse `21.364579991779515`
- now: max_abs `1497.6112467075195`, rmse `13.296448784742802`
- Note: the existing truth embeds WRF's uninitialized-rmol init, so strict
  closure against it is bounded by the proven UB envelope.

## Single remaining blocker

Step-1 surface-layer flux boundary remains upstream of MYNN. The follow-up proof `proofs/v014/step1_sfclay_boundary_fix.md` ports and validates WRF's first-call MYNN surface semantics (UST first guess, MOL=0, land QSFC=qv/(1+qv), Li_etal_2010 z/L seed): UST rmse improves 0.0867->0.0295 and qv-flux rmse improves 1.98e-5->1.44e-5. Strict Step-1 remains red (max_abs 1497.611, rmse 13.296), with the narrower surviving WRF-anchored blocker now TSK/ZNT surface input sourcing (TSK max_abs 8.34 K; ZNT max_abs 0.974 m).

## Fastest next route

Next sprint: emit a tiny WRF step-1 surface-driver hook around module_surface_driver/module_sf_mynn for incoming TSK/ZNT/UST/QSFC/MOL and outgoing UST/HFX/QFX/ZNT on the current d02 Step-1 case, compare those exact arrays against JAX `_surface_column_view` inputs and diagnostics, then fix the TSK/ZNT sourcing if the hook confirms it.

Proof objects: `proofs/v014/mynn_driver_source_output_fix.json`.
