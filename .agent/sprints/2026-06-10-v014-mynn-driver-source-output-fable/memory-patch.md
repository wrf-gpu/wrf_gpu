# Memory Patch

Reviewer Status: accepted by manager as the current compact v0.14 handoff.

## Durable Facts

- The MYNN source-output order-10 deficit is root-caused and partially fixed.
- Missing production behavior was WRF `mym_initialize` first-call level-2
  equilibrium QKE initialization. Implemented through
  `mynn_coldstart_init_columns`, `mynn_coldstart_qke_from_state`, and d02 replay
  cold-start seeding.
- JAX MYNN kernel is proven faithful at the WRF driver boundary with WRF inputs
  and WRF-init QKE: strong-cell ratio median `0.9982`, corr `1.0000`.
- Strict Step-1 residual improved from max_abs `2457.578397008898`, RMSE
  `21.364579991779515` to max_abs `1497.6112512148795`, RMSE
  `13.468453371786723`.
- The remaining blocker is no longer MYNN kernel semantics. It is Step-1
  surface-layer flux/input boundary: `TSK/ZNT/UST/HFX/QFX` and sfclayrev
  first-call semantics.

## Next Action

Dispatch a GPT-5.5 xhigh surface-layer boundary sprint. Use Fable again only if
that GPT sprint cannot localize/fix the boundary or the method becomes
uncertain.
