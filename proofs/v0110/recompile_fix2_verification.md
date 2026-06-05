# v0.11.0 Recompile Fix — Corrected (fix2)

**Verdict: PASS.** All three gates pass. The prior fix (`77700d9`) had the correct
root-cause analysis but a tracing bug; this commit corrects it without any fidelity loss.

## Root cause of the prior failure (`77700d9` TypeError)

`77700d9` device-committed the initial chunk carry to stop a per-chunk JIT recompile.
That commit changed the carry leaves' shardings/avals enough that, during `lax.scan`
tracing of `_advance_chunk`, JAX took its **treedef-mismatch reporting** branch.

That branch (`jax/_src/tree_util.py:762`, `equality_errors_pytreedef`) builds
`tree_unflatten(treedef, [Leaf]*n)` where `Leaf` is a non-array placeholder object
whose type name is literally `Leaf`. Two of our pytrees re-run their **constructors**
in `tree_unflatten`, and those constructors **canonicalise** array children:

- `State.tree_unflatten` -> `State.__init__` -> `jnp.asarray(lu_index, dtype=int32)`
- `DycoreMetrics.tree_unflatten` -> `__post_init__` -> `jnp.asarray(metric, dtype=float64)`

Feeding the `Leaf` placeholder through `jnp.asarray(..., dtype=...)` raised
`TypeError: int()/float() argument ... not 'Leaf'`, aborting tracing **before** chunk-1
hot timing. So the recompile *prevention* logic was fine; the crash was purely in our
non-JAX-contract-compliant `tree_unflatten`s, surfaced only because the committed carry
pushed JAX onto the mismatch-formatting path. (The uncommitted baseline path simply
never entered that branch.)

## Fix

Made `State.tree_unflatten` and `DycoreMetrics.tree_unflatten` the **exact structural
inverse** of `tree_flatten`: `object.__new__(cls)` + `object.__setattr__` to write the
flattened leaves back verbatim, bypassing constructor canonicalisation. The stored
leaves are already canonical (correct dtypes), so the round-trip is the identity
(value-preserving), and it tolerates any placeholder leaf so JAX can format messages
without crashing. The recompile-prevention pieces of `77700d9` (device-committed
initial carry, stable `_StaticHolder(None)` hash) are kept unchanged.

This also closes the separately-tracked `PROOF_TABLE.md` row-11 follow-up
("make `State.tree_unflatten` `.lower()`-safe").

## Gate results

| Gate | Result | Evidence |
|---|---:|---|
| 1. No per-chunk recompile | PASS | `JAX_LOG_COMPILES=1` 3×180-step d02: exactly ONE `_advance_chunk` compile + ONE trace-cache-miss, both in chunk 1. Chunks 2–3 reuse the cached executable. `recompile_fix2_3chunks.log`. |
| 2. Hot chunk ~12 s | PASS | chunk wall (s) = [131.29 cold, **11.82 hot**, **11.85 hot**]; ~65.7 ms/step hot. |
| 3. Bit-identical vs baseline `d054954` | PASS | 1 h d02 forecast, all **66/66** leaves (56 State + 10 M9 diag) match by shape+dtype+sha256. `theta` `0a51eb73…8264a32`, `t2` `1b1085dd…6dafab` on both. |

## Commands

- `JAX_LOG_COMPILES=1 ... python proofs/v0110/recompile_diag.py --chunks 3 --steps 180 --cadence 180` (Gate 1+2)
- `/tmp/bitident_harness.py --hours 1.0 --cadence 180` on the fix branch and on a clean
  `d054954` worktree (Gate 3); full hash diff = bit-identical.

## Proof objects

- `proofs/v0110/recompile_fix2_3chunks.{json,log}`
- `proofs/v0110/recompile_fix2_bitident_fixed.json`
- `proofs/v0110/recompile_fix2_bitident_baseline_d054954.json`
- `proofs/v0110/recompile_fix2_verification.json`
