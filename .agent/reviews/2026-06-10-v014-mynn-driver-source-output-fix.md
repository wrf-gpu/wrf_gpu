# Review: V0.14 MYNN Driver Source-Output Fix

Verdict: `MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`.

Production change: WRF `mym_initialize` level-2 equilibrium cold-start qke
init (was: taper-only seed, 3-5 orders too small in unstable layers).
`_mym_length_option1` refactored (bit-preserving) into a core that accepts
frozen PBLH/Psig_bl/rmol; MYNN test battery `18 passed` plus 4 new focused
tests.

Kernel proven at the WRF driver boundary: ratio median `0.9982`, corr `1.0000`.
Strict Step-1 after-conv residual: `2457.578397008898` -> `1497.6112467075195` (rmse `21.364579991779515` -> `13.296448784742802`).

WRF cold-start init consumes an uninitialized `rmol` (proven for every
column from the hook itself); step-1 bitwise truth is therefore
build/stack/decomposition dependent — strict gates against the existing
part2 truth are UB-bounded. Deterministic rmol-pinned truth emitted.

Single remaining blocker: Step-1 surface-layer flux boundary remains upstream of MYNN. The follow-up proof `proofs/v014/step1_sfclay_boundary_fix.md` ports and validates WRF's first-call MYNN surface semantics (UST first guess, MOL=0, land QSFC=qv/(1+qv), Li_etal_2010 z/L seed): UST rmse improves 0.0867->0.0295 and qv-flux rmse improves 1.98e-5->1.44e-5. Strict Step-1 remains red (max_abs 1497.611, rmse 13.296), with the narrower surviving WRF-anchored blocker now TSK/ZNT surface input sourcing (TSK max_abs 8.34 K; ZNT max_abs 0.974 m).

Next route: Next sprint: emit a tiny WRF step-1 surface-driver hook around module_surface_driver/module_sf_mynn for incoming TSK/ZNT/UST/QSFC/MOL and outgoing UST/HFX/QFX/ZNT on the current d02 Step-1 case, compare those exact arrays against JAX `_surface_column_view` inputs and diagnostics, then fix the TSK/ZNT sourcing if the hook confirms it.
