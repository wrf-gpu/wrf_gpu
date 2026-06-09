# Memory Patch

Scope:

Project-memory update for v0.14 JAX pre-halo capture/debugging.

Reviewer Status: pending memory created, stable memory untouched.

Evidence:

- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/jax_pre_halo_capture.md`
- `tests/test_v014_pre_halo_capture.py`

Proposed destination:

`.agent/memory/pending/2026-06-09-v014-jax-pre-halo-capture.md` records the
lesson as pending. Do not apply it to stable memory until the h10 pre-step carry
checkpoint/wrapper sprint confirms whether this debug hook is the durable
pattern or whether a different checkpoint mechanism should be documented.
