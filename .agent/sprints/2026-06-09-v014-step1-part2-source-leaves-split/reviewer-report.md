# Reviewer Report

Decision: accept as a proof-only localization sprint.

The proof is WRF-anchored and falsifies the prior broad alternatives cleanly:
boundary/spec/acoustic code is too late, moist-theta conversion closes to
roundoff, and aggregate JAX physics state-delta is not a valid narrow substitute
for WRF raw source leaves. The important WRF equation checked by the proof is
exact on the nested interior:

`T_TENDF(after_update_phy_ten) == T_TENDF(after_calculate_phy_tend) + active RTH`.

The accepted next boundary is narrow: implement or prove WRF-compatible dry
physics source leaves for active `RTHRATEN` and `RTHBLTEN` before
`_augment_large_step_tendencies`. Do not use inactive WRF leaves as evidence;
the proof correctly records that inactive leaves can contain uninitialized junk
and ranks only active flags for causal interpretation.

Residual risk: the WRF fixture used direct single-rank CPU because `mpirun`
failed on PMIx socket creation in this sandbox. For this step-1 source-leaf
proof that is acceptable, because the emitted d02 truth covered the full domain
and WRF completed successfully.
