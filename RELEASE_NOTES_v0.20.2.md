# Release notes — v0.20.2

**Type:** output-only / training-data patch (no compile change; default output byte-identical; cache-neutral).

## What's new
- **#122 compact training subset — cloud-validation fields added.** `MINIMAL_TRAINING_SET`
  (opt-in `GPUWRF_TRAINING_OUTPUT_SUBSET`) gains **3 output-only 2D fields** for validating
  cloud forecasts against MSG-satellite cloud truth:
  - **OLR** — TOA outgoing longwave (== cloud-top, satellite-observable; WRF `LWUPT`-derived).
  - **RAINC** — convective precipitation (completes grid-scale `RAINNC` for the 3 km cumulus parent).
  - **SWDNB** — RRTMG instantaneous surface downwelling shortwave.
  Subset is now **39 named variables** (was 36). The fields already exist in the full default output;
  this only adds them to the opt-in subset.

## Invariants (unchanged)
- The **default** full nest output is **byte-identical** to v0.20.1 (the subset is opt-in only).
- **No HLO / compile change** → the jit cache is unaffected (cache-neutral; cross-date #114 and
  cross-region warm-cache behavior unchanged).
- Validated: `tests/test_v0201_training_output_subset.py` (10/10, incl. the default-byte-identity check).

## Everything else
Identical to v0.20.1 (#114 cross-date warm nest cache, #122 compact output, #123 GPU OOM hardening,
S2 B200 I/O tooling). See `RELEASE_NOTES_v0.20.1.md`.
