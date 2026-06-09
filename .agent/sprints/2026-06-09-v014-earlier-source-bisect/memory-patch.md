# Memory Patch

Scope:

Project-memory update for v0.14 grid-parity debugging after the earlier-source
bisection.

Evidence:

- `proofs/v014/earlier_source_bisect.json` verdict is
  `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.
- Initial JAX d02 carry `PB/MUB` matches native `wrfinput_d02`.
- Initial JAX d02 carry `PB/MUB` already differs from CPU-WRF h0/h1/h10 and
  h10 pre-RK truth.
- CPU-WRF `PB/MUB` are stable across h0, h1, h10 wrfout and the h10 pre-RK
  hook on the target patch.
- Worst base field is `MUB`, max_abs `1050.3046875`; `PB` is also wrong with
  max_abs `1047.015625`.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-base-state-split-mismatch.md`.
After the source fix lands and validates, condense into stable memory with the
exact WRF/JAX formula or oracle path.

Reviewer Status:

Pending. Do not promote to stable memory until the source-changing fix sprint
proves the corrected base-state split and records any compatibility caveats.
