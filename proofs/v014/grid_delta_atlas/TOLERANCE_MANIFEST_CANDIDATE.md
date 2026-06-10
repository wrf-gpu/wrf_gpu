# V0.14 Grid-Delta Atlas Tolerance Manifest Candidate

Date: 2026-06-10
Status: `candidate_pre_result_freeze`

## Summary

This candidate freezes only thresholds that already have documented support. The ten hard dynamic fields use the predeclared operational pooled-RMSE limits from `scripts/equivalence_demo.py`, `docs/equivalence-demo.md`, KI-9, and the Switzerland plan. Static grid fields are exact or tight formula checks. Fields without a frozen physical envelope remain report-only.

The machine-readable manifest is `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`. It is compatible with both `scripts/build_grid_delta_atlas.py --tolerance-json` and `scripts/compare_wrfout_grid.py --tolerance-json`.

## Hard Dynamic Fields

| Field | Units | Hard metric | Rationale | Main risk |
| --- | --- | ---: | --- | --- |
| `T2` | K | RMSE <= 1.5 | Existing 2 m temperature operational bar. | Pooled RMSE can hide land/terrain pockets. |
| `U10` | m s-1 | RMSE <= 1.5 | Existing 10 m zonal wind bar. | KI-9 lead-time wind drift can hide in pooled summaries. |
| `V10` | m s-1 | RMSE <= 1.5 | Existing 10 m meridional wind bar. | KI-9 showed this as the dominant surface drift channel. |
| `PSFC` | Pa | RMSE <= 120 | Existing surface-pressure bar, about 0.1 percent of 100 kPa. | Residual can be dynamical, not only diagnostic. |
| `RAINNC` | mm | RMSE <= 1.0 | Existing grid-scale accumulated precipitation bar. | Local maxima still need p99/max and spatial review. |
| `T` | K | RMSE <= 1.5 | Existing 3D perturbation potential temperature bar. | Not an operator parity or bitwise claim. |
| `U` | m s-1 | RMSE <= 1.8 | Existing 3D zonal wind bar. | Requires native stagger-aware shape comparison. |
| `V` | m s-1 | RMSE <= 1.8 | Existing 3D meridional wind bar. | Old failed runs exceed this materially. |
| `W` | m s-1 | RMSE <= 0.30 | Existing small-magnitude vertical velocity bar. | Local acoustic artifacts may require p99/max review. |
| `QVAPOR` | kg kg-1 | RMSE <= 1.0e-3 | Existing 3D water-vapor bar. | Moisture compensation can hide in station metrics. |

All hard fields also require `finite_pair_fraction_min = 1.0`.

## Static Fields

Static geometry, map-factor, rotation, vertical-coordinate, and grid-spacing fields are exact-copy checks with `max_abs = 0.0` when both CPU and GPU emit them. This intentionally catches writer/grid-convention mismatches instead of treating them as forecast tolerance.

`HGT`, `PB`, `PHB`, and `MUB` are tight formula checks with `max_abs <= 0.2` in native units. The basis is `proofs/v014/live_nest_base_source_fix.md`, where the closed target-patch residuals were `HGT=2.4167598553503922e-05 m`, `PB=0.04890023032203317 Pa`, `MUB=0.044447155625675805 Pa`, and `PHB=0.09328280997578986 m2 s-2`.

## Report-Only

`P`, `PH`, `MU`, and `RAINC` are mandatory inventory fields but remain report-only in this manifest because I did not find frozen v0.14 all-cell thresholds for them. The atlas gate still requires reviewer attention for systematic drift in these fields.

Other report-only families: radiation, surface-energy, PBL, land-state, hydrometeor/cloud, snow, and optional diagnostic fields. They must be inventoried and plotted, but a pass/fail threshold would be speculative today.

## Regional Scope

No separate Canary or Switzerland numeric override is proposed. The Switzerland validation plan uses the same ten hard fields and limits. Canary station TOST margins are not grid-cell tolerances and are recorded only as context in the JSON.

## Historical Check

The old v0.12 proof `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json` would still fail under this manifest: `U10`, `V10`, `PSFC`, `T`, `U`, and `V` exceed the hard RMSE limits. This candidate does not widen thresholds to pass known red runs.

## Manager Decision

Recommended decision: `FREEZE_AFTER_REVIEW` for the hard dynamic and static checks. Keep `P`, `PH`, `MU`, `RAINC`, and other untoleranced diagnostics report-only unless a separate independent review freezes additional field envelopes before final scoring.
