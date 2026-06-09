# Manager Closeout

## Outcome

The v0.14 memory/FP32 lane is closed on `worker/mythos/v014-memory-fp32`:
one measured-material fix (MYNN BouLac column tiling, -11.53 GiB compiled temp
at target geometry, bit-identical), one bit-identical hygiene fix reclassified
non-material by measurement (transport-velocity reuse, 0.0 GiB — XLA CSE),
FP32 R0 contract landed default-inert with the R1+ blocker quantified, the
exact-branch GPU preflight green on baseline and final tree, and every other
roadmap row closed as measured-defer / non-material / ADR-gated with exact
reasons in the proof object.

## Proof Objects

- `proofs/v014/mythos_memory_fixes_260609.{py,json,md}`
- `proofs/v014/mythos_memory_gpu_suite_260609.{py,json}`
- `proofs/v014/exact_branch_memory_preflight.{json,md}` +
  `_baseline_a32efce3.{json,md}`
- `proofs/v014/fp32_acoustic_static_audit.{py,json}`
- `proofs/v013/moisture_advection_wiring.json` (re-passed)
- `.agent/reviews/2026-06-09-v014-mythos-memory-fixes.md`

## Merge Decision

Merge Decision: PENDING primary manager review (Mythos recommendation:
`MERGE_NOW`; three separated commits: bit-identical memory fixes /
FP32 R0 contract / proofs+docs).

## Scope Changes

None. Acoustic-adjacent rows and FP32 R1/R2 were deliberately not implemented
(exact fault-surface blocker recorded), which the contract authorizes for
rows that are technically unsafe until another correctness gate closes.

## Lessons

- Static duplicate-expression estimates are not VRAM gains until a
  compiled-memory measurement confirms XLA did not already CSE them.
- The RRTMG leading-column tiling pattern generalizes cleanly to per-column
  physics kernels; per-column independence + empirical CPU/GPU bit identity is
  the right acceptance bar.
- Dense `(B, nz, nz)` vectorizations of nested Fortran searches (BouLac) are
  the canonical hidden multi-GiB transient; audit other schemes for the same
  shape signature when profiling reopens memory work.

## Next Sprint

Dynamics frontier (one-step namelist parity freeze + RK1 substage comparators
with the closed init), then FP32 R1 explicit base-state plumbing once that
fault surface is released.
