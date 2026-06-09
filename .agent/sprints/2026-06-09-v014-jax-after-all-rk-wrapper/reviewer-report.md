# Reviewer Report

Decision: accepted as a genuine wrapper/API blocker, not as a model root-cause
claim.

The proof did not take the shortcut of comparing retained wrfout output as if it
were the same surface. It records that retained JAX/GPU h10 remains divergent,
but uses that only as a diagnostic because it is not CPU internal pre-halo
state. The runtime-source inspection is consistent with the code: `_acoustic_scan`
returns only the haloed state, and `_rk_scan_step` also applies halo at the
stage boundary.

Material evidence reviewed:

- `proofs/v014/jax_after_all_rk_wrapper.md`
- `proofs/v014/jax_after_all_rk_wrapper.json`
- `.agent/reviews/2026-06-09-v014-jax-after-all-rk-wrapper.md`

Required follow-up: a minimal, default-off capture hook in
`src/gpuwrf/runtime/operational_mode.py` that exposes the pre-halo state for
proof code without changing normal forecast behavior.
