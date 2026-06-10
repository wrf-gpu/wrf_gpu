# Review: V0.14 MYNN Driver Source-Output Fix

Verdict: `MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`.

Production change: WRF `mym_initialize` level-2 equilibrium cold-start qke
init (was: taper-only seed, 3-5 orders too small in unstable layers).
`_mym_length_option1` refactored (bit-preserving) into a core that accepts
frozen PBLH/Psig_bl/rmol; MYNN test battery `18 passed` plus 4 new focused
tests.

Kernel proven at the WRF driver boundary: ratio median `0.9982`, corr `1.0000`.
Strict Step-1 after-conv residual: `2457.578397008898` -> `1497.6112512148795` (rmse `21.364579991779515` -> `13.468453371786723`).

WRF cold-start init consumes an uninitialized `rmol` (proven for every
column from the hook itself); step-1 bitwise truth is therefore
build/stack/decomposition dependent — strict gates against the existing
part2 truth are UB-bounded. Deterministic rmol-pinned truth emitted.

Single remaining blocker: Step-1 surface-layer flux boundary: the JAX step-1 sfclay outputs feeding MYNN differ from WRF's (ustar bias -0.077/max 0.176; HFX rmse 24.6 W/m^2; QFX bias -2.1e-5), driven by (a) land skin-temperature input differences up to 8.3 K, (b) roughness-length differences up to 0.97 m, and (c) sfclayrev FIRST-CALL semantics (JAX starts from ustar=0 while WRF iterates from a first guess: identical-input ocean columns still show 4x ustar deficits). With WRF fluxes substituted, the production init already reaches strong-cell ratio 0.72/corr 0.993 (case B), and with WRF init qke too it reaches 1.00 (case A).

Next route: One sprint: emit a WRF step-1 surface-driver hook (same disposable pattern) around module_sf_mynn/sfclayrev for TSK/ZNT/UST/HFX/QFX in/out, port the sfclayrev first-call (flag_iter/UST first-guess) semantics + skin-temperature/roughness sourcing into the JAX surface adapter, and gate on case-D converging to case-B levels; then rerun the strict Step-1 proofs against the rmol-pinned WRF truth.
