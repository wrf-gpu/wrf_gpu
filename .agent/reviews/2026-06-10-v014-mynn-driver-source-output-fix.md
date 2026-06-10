# Review: V0.14 MYNN Driver Source-Output Fix

Verdict: `MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`.

Production change: WRF `mym_initialize` level-2 equilibrium cold-start qke
init (was: taper-only seed, 3-5 orders too small in unstable layers).
`_mym_length_option1` refactored (bit-preserving) into a core that accepts
frozen PBLH/Psig_bl/rmol; MYNN test battery `18 passed` plus 4 new focused
tests.

Kernel proven at the WRF driver boundary: ratio median `0.9982`, corr `1.0000`.
Strict Step-1 after-conv residual: `2457.578397008898` -> `438.5379097262689` (rmse `21.364579991779515` -> `5.4654420375782955`).

WRF cold-start init consumes an uninitialized `rmol` (proven for every
column from the hook itself); step-1 bitwise truth is therefore
build/stack/decomposition dependent — strict gates against the existing
part2 truth are UB-bounded. Deterministic rmol-pinned truth emitted.

Single remaining blocker: Step-1 surface-layer flux boundary remains upstream of MYNN. The follow-up proofs `proofs/v014/step1_sfclay_boundary_fix.md` and `proofs/v014/step1_tsk_znt_sourcing_fix.md` port WRF first-call MYNN surface semantics and prove exact TSK/ZNT/MAVAIL input sourcing at the sfclay_mynn hook (TSK max_abs 0.0 K; ZNT max_abs 1.19e-8 m). Strict Step-1 remains red (max_abs 1497.611, rmse 13.253), with the narrower surviving WRF-anchored blocker now the non-surface thermodynamic column inputs entering sfclay_mynn: th_phy/t_phy/p_phy.

Next route: Next sprint: localize the non-surface thermodynamic column inputs at the exact sfclay_mynn hook (`th_phy(kts)`, `t_phy(kts)`, `p_phy(kts)`, and `dz8w`) against JAX `_surface_column_view`; then fix the Step-1 temperature/pressure sourcing if local.
