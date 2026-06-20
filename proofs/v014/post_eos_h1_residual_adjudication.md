# Post-EOS H1 Residual Adjudication

## Verdict

PROCEED_72H_GATES. The post-EOS h1 falsifier no longer shows the radical theta/EOS pressure-temperature failure. The remaining residuals are bounded enough to start the 72h stability/atlas runs, but this is not a release-green h1 score: the current candidate manifest still fails `PSFC` narrowly and `PB`/`MUB` globally because of static 5-cell-frame spikes. Those are final-scoring risks to carry into the 72h atlas, not evidence of renewed live interior drift.

## Evidence Table

| Field | Metric old -> new | Class | Gate implication |
|---|---:|---|---|
| `PSFC` | RMSE 323.115 -> 124.299; bias -313.780 -> -116.855; p99 198.146; max 261.156 | hard_dynamic_h1_margin_breach | nonblocking_for_start_but_not_release_green_until_72h_scored_passes |
| `MU` | RMSE 121.961 -> 98.291; bias -85.142 -> 85.109; p99 192.134; max 204.205 | critical_report_only_bounded | review_in_72h_atlas_nonblocking_for_start |
| `P` | RMSE 129.754 -> 39.090; bias -85.662 -> -8.202; p99 139.904; max 196.543 | critical_report_only_bounded | review_in_72h_atlas_nonblocking_for_start |
| `PB` | RMSE 4.521 -> 4.521; bias -0.036 -> -0.036; p99 0.107; max 249.883 | known_static_boundary_spike | nonblocking_for_start_but_final_static_exactness_fails_current_manifest |
| `MUB` | RMSE 9.276 -> 9.276; bias -0.119 -> -0.119; p99 18.194; max 250.664 | known_static_boundary_spike | nonblocking_for_start_but_final_static_exactness_fails_current_manifest |
| `PH` | RMSE 85.157 -> 48.069; bias -53.332 -> 25.433; p99 128.426; max 204.046 | critical_report_only_bounded | review_in_72h_atlas_nonblocking_for_start |
| `T` | RMSE 1.457 -> 0.255; bias 0.671 -> 0.082; p99 0.859; max 2.233 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `THM` | RMSE NA -> 0.242; bias NA -> 0.081; p99 0.851; max 2.255 | new_theta_m_diagnostic_green | confirms_post_eos_writer_semantics_at_h1 |
| `U` | RMSE 0.924 -> 0.343; bias -0.072 -> -0.060; p99 1.092; max 3.650 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `V` | RMSE 2.116 -> 1.689; bias -1.661 -> -1.657; p99 2.339; max 4.682 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `W` | RMSE 0.106 -> 0.028; bias -0.001 -> -0.002; p99 0.098; max 0.859 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `QVAPOR` | RMSE 2.008e-04 -> 1.896e-04; bias 3.416e-06 -> -2.093e-06; p99 7.897e-04; max 0.005 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `HFX` | RMSE 38.186 -> 38.568; bias -5.867 -> 10.466; p99 239.081; max 459.069 | physics_diagnostic_report_only_bounded | measure_in_72h_nonblocking_for_start |
| `LH` | RMSE 53.896 -> 27.111; bias 11.997 -> 18.714; p99 105.080; max 306.199 | physics_diagnostic_report_only_bounded | measure_in_72h_nonblocking_for_start |
| `PBLH` | RMSE 78.950 -> 41.407; bias 23.978 -> 11.192; p99 156.228; max 274.550 | physics_diagnostic_report_only_bounded | measure_in_72h_nonblocking_for_start |
| `SWDOWN` | RMSE 55.615 -> 55.673; bias -55.592 -> -55.650; p99 59.214; max 64.088 | radiation_timing_report_only | measure_drift_slope_in_72h_nonblocking_for_start |
| `SWNORM` | RMSE 57.363 -> 57.419; bias -55.561 -> -55.619; p99 110.935; max 203.550 | radiation_timing_report_only | measure_drift_slope_in_72h_nonblocking_for_start |
| `COSZEN` | RMSE 0.055 -> 0.055; bias -0.055 -> -0.055; p99 0.056; max 0.056 | radiation_timing_report_only | measure_drift_slope_in_72h_nonblocking_for_start |
| `GLW` | RMSE 8.486 -> 1.194; bias -6.250 -> 0.128; p99 4.356; max 35.366 | physics_diagnostic_report_only_bounded | measure_in_72h_nonblocking_for_start |
| `T2` | RMSE 2.236 -> 0.386; bias -1.576 -> -0.005; p99 2.321; max 3.126 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `U10` | RMSE 2.124 -> 0.610; bias 0.050 -> -0.298; p99 1.535; max 3.026 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `V10` | RMSE 3.710 -> 1.227; bias -1.282 -> -1.135; p99 2.494; max 3.011 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `RAINNC` | RMSE 0 -> 0; bias 0 -> 0; p99 0; max 0 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `PHB` | RMSE 0.002 -> 0.002; bias 1.151e-06 -> 3.959e-07; p99 0.008; max 0.016 | green_under_current_manifest | hard_manifest_green_at_h1 |
| `HGT` | RMSE 0 -> 0; bias 0 -> 0; p99 0; max 0 | green_under_current_manifest | hard_manifest_green_at_h1 |

All 100 common numeric h1 fields were parsed from the post-fix compare JSON; the full per-field metrics are in the companion JSON under `all_numeric_field_stats`.

## Boundary/Static Spike Analysis

- `PB`: max 249.883 Pa at index [0, 57, 156]; cells >0.2 Pa: 4160 total, 4160 in the 5-cell frame, 0 in the interior. Interior max is 0.0078125 Pa.
- `MUB`: max 250.664 Pa at index [57, 156]; cells >0.2 Pa: 194 total, 194 in the 5-cell frame, 0 in the interior. Interior max is 0.0078125 Pa.
- `PHB`: max 0.015625 m2/s2, below the 0.2 static exactness limit everywhere.
- `P`, `PH`, `PSFC`, and `MU` are live dynamic residuals across the domain, not static-boundary-only artifacts. `P/PH/MU` are report-only in the candidate manifest; `PSFC` is the only hard dynamic h1 breach.

## Radiation Timing Analysis

`COSZEN` h1 bias is -0.055 against a CPU h0->h1 change of 0.183 per hour, implying about -18.043 minutes by local slope. `SWDOWN` gives about -22.488 minutes. This is the same known timing class; it is report-only and should be measured for drift slope over 72h, not fixed before launch.

## Manifest/Tolerance Implication

- Hard h1 manifest-green fields: `T2`, `U10`, `V10`, `RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`, `HGT`, and `PHB`.
- Hard h1 manifest failures if the candidate manifest is applied now: `PSFC` RMSE 124.299 Pa > 120 Pa; `PB` max 249.883 Pa > 0.2 Pa; `MUB` max 250.664 Pa > 0.2 Pa.
- `P`, `PH`, and `MU` are critical report-only fields in the current manifest; no frozen RMSE/max limit exists for them. Do not widen the manifest to bless h1. Run the 72h gate with the manifest and report these as drift diagnostics.
- Starting the 72h gate is compatible with the release-gate start criterion because the short falsifier found bounded, classified residuals rather than nonfinite output, schema failure, or renewed radical field drift. Final release scoring remains stricter than this start decision.

## Exact Next Manager Commands

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_canary_d02_72h_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources}
set +e
scripts/run_gpu_lowprio.sh --cores 0-23 --resource-log-dir "$RUN_ROOT/resources" --resource-label v014_canary_d02_72h --resource-interval 5 -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py --run-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2 --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output --run-id 20260501_18z_l2_72h_20260519T173026Z --hours 72 --output-root "$RUN_ROOT/gpu_output" --proof-dir "$RUN_ROOT/proofs" > "$RUN_ROOT/canary_d02_72h_gpu.log" 2>&1
echo $? > "$RUN_ROOT/canary_d02_72h_gpu.rc"
set -e
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/compare_wrfout_grid.py --cpu-dir <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z --gpu-dir "$RUN_ROOT/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z" --domain d02 --init 2026-05-01T18:00:00+00:00 --min-lead 1 --max-lead 72 --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json --out-json "$RUN_ROOT/canary_d02_72h_grid_compare.json" --out-md "$RUN_ROOT/canary_d02_72h_grid_compare.md"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/build_grid_delta_atlas.py --cpu-dir <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z --gpu-dir "$RUN_ROOT/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z" --case-id canary_d02_20260501_18z --domain d02 --init 2026-05-01T18:00:00+00:00 --min-lead 1 --max-lead 72 --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json --proof-dir "$RUN_ROOT/grid_delta_atlas" --asset-dir "$RUN_ROOT/grid_delta_atlas_assets"
```

## Context-Sparing Handoff

- objective: adjudicate post-EOS h1 residuals for 72h gate start.
- files changed: `proofs/v014/post_eos_h1_residual_adjudication.py`, `.json`, `.md`.
- commands run: JSON validation for old/new compares; manifest-aware h1 comparator to `/tmp`; NetCDF boundary/spin-up inspection; proof script; py_compile/json.tool.
- proof objects produced: this markdown and `proofs/v014/post_eos_h1_residual_adjudication.json`.
- verdict: `PROCEED_72H_GATES`; no launch blocker found.
- hard final-scoring risks: `PSFC`, `PB`, `MUB` under the current candidate manifest.
- report-only nonblockers: `P`, `PH`, `MU`, radiation timing, surface flux/PBL diagnostics.
- boundary result: `PB/MUB` exactness failures are entirely in the 5-cell nest frame, not live interior drift.
- unresolved risk: 72h slope could expose growth in `PSFC/MU/P/PH` or V/V10 despite h1 boundedness.
- next decision needed: manager launches the 72h Canary gate and scores it honestly with the frozen candidate manifest.
