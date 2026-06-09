# Memory Patch

Scope:

Project-memory consideration for the v0.14 JAX same-state wrapper attempt
against the green WRF `post after_all_rk_steps pre-halo` surface.

Reviewer Status: no stable memory edit yet.

This sprint found an API blocker rather than a durable numerical lesson. The
current handoff and project plan have been updated so future managers do not
try to compare retained wrfout or post-halo state against the WRF pre-halo
oracle.

Evidence:

- `proofs/v014/jax_after_all_rk_wrapper.json`
- `proofs/v014/jax_after_all_rk_wrapper.md`

Proposed destination:

No stable memory update until the source hook and same-surface comparison
complete. If the hook confirms the pattern is generally useful, add a stable
debugging note that the operational runtime needs explicit capture points for
pre-halo WRF cadence surfaces.
