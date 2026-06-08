# Memory Patch Proposal

## Scope

Project-memory update for the 2026-06-08 grid-cell attribution result: emitted
static grid, vertical-coordinate, and base-state payload parity must be checked
before spending long GPU time on powered TOST, Switzerland equivalence,
FP32/mixed-precision validation, or speed claims.

## Evidence

- `proofs/v014/grid_cell_envelope.json` and
  `proofs/v014/grid_cell_envelope.md` compare the retained Case 3 wrfouts and
  show 31 non-exact static/grid fields.
- `.agent/reviews/2026-06-08-v014-grid-parity-attribution.md` ranks static
  metric mismatch as the first fix target, with `C2H/C2F` max 95,000 Pa,
  `C4H/C4F` about 26.7 kPa, `RDN` max 161.7, and `HGT` max 228 m.
- `proofs/v014/wind_mass_divergence_probe.json` shows the visible V10 issue is
  coupled to 3D prognostic wind and pressure/mass/geopotential divergence, not a
  pure station or 10 m diagnostic issue.
- The principal explicitly requires "no slob": station-skill evidence must not
  hide direct grid-cell divergence.

## Proposed Destination

`.agent/memory/stable/approved-patterns.md` after independent review approves
the wording.

## Patch

Proposed addition:

- Before long validation campaigns or precision/speed claims, first prove that
  emitted and runtime static grid, vertical-coordinate, and base-state payloads
  match CPU-WRF truth exactly or within a predeclared dtype tolerance. If static
  metric/base parity is red, pause TOST/FP32/speed closure and root-cause that
  mismatch before dynamic operator tuning.

## Reviewer Status

Reviewer Status: pending. Do not apply to stable memory until the static
metric/base parity sprint either fixes the mismatch or proves it is a harmless
writer-only artifact.
