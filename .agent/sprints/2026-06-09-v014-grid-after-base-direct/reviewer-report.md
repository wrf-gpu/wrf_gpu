# Reviewer Report

## Findings

- The sprint met its main contract: current branch includes `7d11be42`, exactly
  one bounded h12 GPU run was launched, and the result was compared CPU-only
  against CPU-WRF d02 wrfouts for h1-h12.
- The proof is correctly scoped as `GRID_SYMPTOM_NOT_CLOSED`. It does not hide
  the remaining V10, pressure, mass, and geopotential residuals.
- Static/base payloads improved materially: exact C/DN/RDN/MAPFAC/lat-lon, near
  exact HGT, and PHB max abs `0.109375`. PB and MUB remain non-exact at roughly
  250 Pa max abs, so base/static is not fully closed.
- Dynamic residuals remain large and coherent: `V10`, `U10`, `PSFC`, `P`, `MU`,
  and `PH` all fail over h1-h12. This matches the independent same-state proof
  that already found post-RK/pre-halo dynamic mismatch.

## Correctness Risks

The comparison has no tolerance manifest, so it is report-only. That is
acceptable for this sprint because the objective was direct symptom assessment,
not a formal pass/fail tolerance gate. The magnitude of remaining dynamic
errors is too large for any practical grid-parity claim.

## Performance Risks

The h12 run wall time was recorded, but peak VRAM was not captured in the proof
JSON. Future GPU run wrappers should include peak VRAM in the committed artifact
instead of relying on external `nvidia-smi` observation.

## Required Fixes

No production code fix is accepted from this sprint. The next debug step should
regenerate same-state carries on current code and instrument one layer earlier
inside final RK pressure-gradient/mass-wind coupling.

Decision:

Accept as a proof artifact only. Grid parity, V10 closure, TOST, Switzerland,
FP32, and release tagging remain blocked by dynamic divergence.
