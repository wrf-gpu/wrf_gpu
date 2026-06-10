# V0.14 Grid-Delta Atlas Gate

Date: 2026-06-09
Owner: manager

## Decision

The v0.14 final validation gate is not station TOST. Before v0.14 can be
tagged, the validation campaign must produce a **Grid-Delta Atlas** comparing
GPU wrfout against CPU-WRF wrfout for every paired case, lead time, grid cell,
and common numeric field in the mandatory 72h field-parity/stability campaign.

The required v0.14 pillars are:

1. **Switzerland/Gotthard 72h field-parity/stability** against CPU-WRF.
2. **Canary L2 d02 72h field-parity/stability** against CPU-WRF.
3. **Grid-cell delta envelope and plots** for model-stability/equivalence
   evidence over all common numeric fields and all cells.

Powered TOST is secondary station sanity evidence only. It can be reported with
the atlas, but it is not a v0.14 tag gate.

## Scope

For each included validation case:

- Pair every GPU `wrfout` frame with the matching CPU-WRF `wrfout` frame by
  domain, valid time, and lead hour.
- Compare every common numeric wrfout field, including static fields, 2-D
  surface fields, 3-D mass fields, staggered `U/V/W`, accumulated precipitation,
  tendencies/diagnostics when present, and all released writer fields.
- Use stagger-aware shape handling; never silently crop except by an explicit
  schema rule recorded in the artifact.
- Record missing fields and non-numeric fields explicitly. Missing common core
  fields are hard failures; non-common optional fields are inventory items, not
  hidden exclusions.
- Compute per-case, per-lead, per-field, and pooled metrics:
  `count`, finite counts, `max_abs`, `RMSE`, `MAE`, `bias`, `p50/p95/p99/p99.9`
  absolute delta, safe relative error where meaningful, correlation, and worst
  index/location.
- Compute stability-over-time summaries: per-field lead-time slope of RMSE and
  bias, maximum lead-to-lead jump, and late-window versus early-window deltas.

## Plots

The gate must generate compact, release-ready plots:

- field x lead heatmaps for RMSE, bias, p99, and max_abs;
- per-field lead-time time series for the mandatory core fields;
- ECDF or violin/ridge plots of absolute deltas for core fields;
- spatial maps for each core field's worst case/lead and for any field that
  breaches a hard threshold;
- a pooled pass/fail dashboard image suitable for README embedding.

Plots must be deterministic and versioned under a repo path such as
`docs/assets/v014/grid_delta_atlas/`. Large raw tables may live under
`proofs/v014/grid_delta_atlas/` or a `/mnt/data` scratch root, but the release
commit must include the compact report, manifest, and selected plots.

## Release Artifacts

Required v0.14 artifacts:

- `proofs/v014/grid_delta_atlas/manifest.json`
- `proofs/v014/grid_delta_atlas/grid_delta_summary.json`
- `proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS.md`
- selected plot PNG/SVG files under `docs/assets/v014/grid_delta_atlas/`
- README section embedding the dashboard plot and linking the full atlas report
- optional TOST report linking to the same atlas if station scoring is run, so
  station scores and full-grid deltas cannot diverge silently

## Gate Semantics

Avoid claiming bitwise or cell identity for a free-running chaotic weather
forecast unless the data actually prove it. The correct claim is:

- **near-equivalence under predeclared field envelopes** for all core fields;
- **stable bounded drift** over lead time;
- **full inventory transparency** for every common numeric field.

Hard-fail before tag:

- any nonfinite GPU field in paired output;
- any missing mandatory core field;
- an unexplained systematic field-family drift in `T/U/V/W/QVAPOR/P/PH/MU`,
  surface winds, `T2`, `PSFC`, or precipitation;
- a grid-delta breach in a predeclared hard field envelope;
- a station TOST result that is interpreted as stronger than the grid-delta
  atlas.

Report-only until thresholds are frozen:

- fields with no predeclared physical tolerance class at atlas-design time.
  They must still be plotted and inventoried; a reviewer must decide before
  release whether any observed residual invalidates the v0.14 claim.

## Implementation Notes

Existing `proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py` already
has a limited cell-level statistics pillar. v0.14 must use the standalone atlas
tooling as the release surface:

- all common numeric fields, not only the old `FIELDS` subset;
- all leads and all paired cases;
- deterministic plots and release docs;
- explicit tolerance manifest and pass/fail dashboard;
- concise top-level output to preserve manager/agent context.

This atlas is run after the short field falsifier and final memory preflight
show that long GPU validation is meaningful.
