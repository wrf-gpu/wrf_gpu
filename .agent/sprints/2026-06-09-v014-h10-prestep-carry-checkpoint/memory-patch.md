# Memory Patch

Scope:

Project-memory consideration for v0.14 h10 same-state checkpointing.

Reviewer Status: no stable memory edit yet.

Evidence:

- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry.md`

Proposed destination:

No stable memory update yet. If the next checkpoint producer succeeds, update
the existing pending pre-halo memory with the exact carry/checkpoint command
that future managers should reuse.
