# Reviewer Report

## Findings

- The sprint stayed inside its write scope: proof helper, JSON, Markdown, and
  review artifact only.
- There is no `src/` diff and no GPU use.
- The contract asked for either a same-state comparison table or an exact
  wrapper-needed verdict. The same-state comparison ran and produced a compact
  table against WRF's `post_after_all_rk_steps_pre_halo` surface.
- The result is not a fix, but it is stronger than a broad h10/grid symptom:
  `U` already disagrees at the named post-RK/pre-halo surface, before later halo
  cadence, wrfout writer mapping, station interpolation, TOST, or Switzerland
  postprocessing can be blamed.
- The proof correctly limits base-field conclusions because the h10 carry
  predates `live_nest_base_source_fix.json`.

## Correctness Risks

The proof covers a selected h10 patch and selected vertical/staggered keys. It
does not prove whole-domain behavior. `PHB` truth is from the green WRF h10
wrfout static field because the post-RK text hook did not emit `PHB`; dynamic
fields use the WRF text surface.

## Performance Risks

None introduced. The new proof script sets `JAX_ENABLE_COMPILATION_CACHE=false`
by default to avoid stale CPU AOT cache warning floods during proof reruns. This
is proof-only and has no production runtime effect.

## Required Fixes

No immediate code fix is required for this sprint. The next implementation or
debug sprint should regenerate the h10 carry on current code, then instrument
one layer earlier inside final RK U/V tendency/acoustic update, mass coupling,
and theta-pressure refresh inputs.

Decision:

Accept with restricted claim: same-state dynamic localization only. Grid parity,
V10 closure, TOST, and Switzerland validation remain open.
