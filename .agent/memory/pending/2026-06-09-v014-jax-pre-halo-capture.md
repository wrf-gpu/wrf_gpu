# Pending Memory Patch: V0.14 JAX Pre-Halo Capture

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF debugging after the green
WRF `post after_all_rk_steps pre-halo` target was found.

Evidence:

- `proofs/v014/jax_after_all_rk_wrapper.json` proved the public runtime exposed
  only post-halo/post-guard state and could not honestly compare JAX against the
  WRF pre-halo target.
- `proofs/v014/jax_pre_halo_capture.json` proves a default-off private capture
  path now exists in `src/gpuwrf/runtime/operational_mode.py`.
- `tests/test_v014_pre_halo_capture.py` proves disabled normal RK return is still
  `OperationalCarry`, public forecast signatures do not expose capture, and the
  capture path returns the same normal carry plus a finite pre-halo `State`.
- The h10 numerical comparison remains blocked because there is no CPU-loadable
  JAX `OperationalCarry` immediately before `d02` step 6000/h10.

Proposed destination:

After independent review and after the h10 checkpoint/wrapper sprint finishes,
add a concise entry to `.agent/memory/stable/approved-patterns.md`:

- For same-state WRF cadence debugging, add default-off proof hooks only when
  the public runtime exposes the wrong cadence surface. A hook is not a model
  fix; it must preserve normal return types and remain gated behind proof-only
  private APIs until a same-surface JAX-vs-WRF comparison names a mismatch.

Reviewer Status:

Pending. Do not apply to stable memory until the h10 pre-step carry checkpoint
either enables the same-surface comparison or proves a more specific checkpoint
mechanism is required.
