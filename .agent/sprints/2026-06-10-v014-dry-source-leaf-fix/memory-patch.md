# Memory Patch

Reviewer Status: accepted by manager as a compact memory update for the next
sprint and compaction handoff.

## Durable Facts

- v0.14 grid parity remains the active release gate.
- The dry source-leaf sprint did not close Step-1 `T_TENDF`; it localized the
  remaining source-fidelity gaps after adding narrow MYNN source-leaf plumbing.
- Fable/Mythos must remain conserved. This is not yet a Fable escalation case:
  the current blocker is structured enough for another GPT-5.5 xhigh sprint.

## Current Source Boundary

Accepted proof:

`proofs/v014/step1_dry_source_leaf_fix.md`

Verdict:

`DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`

Ranked blockers:

1. JAX MYNN `RTHBLTEN` source is not WRF-compatible.
2. Held JAX `rthraten` is zero/stale at Step 1 while WRF has active
   `RTHRATEN`.
3. WRF `conv_t_tendf_to_moist` / `QV_TEND` coupling is missing before the JAX
   dry source bundle.

## Next Action

Dispatch a coherent GPT source-fidelity sprint covering MYNN
`RTHBLTEN/RQVBLTEN`, held-radiation Step-1 initialization/cadence, and
`conv_t_tendf_to_moist`. Gate remains the strict Step-1 proof collapsing toward
the prior target (`max_abs <= 1e-3`, `RMSE <= 1e-5`) or producing one narrower
WRF-anchored blocker.
