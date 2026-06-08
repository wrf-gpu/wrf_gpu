# Sprint Contract: V0.14 Grid-Cell Parity Attribution

Date: 2026-06-08
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Build the first durable all-comparable-field CPU-WRF-vs-GPU-WRF grid-cell
divergence attribution layer. The immediate goal is not to tune tolerances or
ship a model-code fix; it is to identify where and how the fields diverge so the
next sprint can fix the responsible operator with falsifiable evidence.

## Priority Context

The project priority order is now:

1. Grid-cell parity and root-cause fixes across all written wrfout fields.
2. FP32 acoustic / mixed precision.
3. Remaining memory issues.
4. Powered TOST only after grid fields are no longer radically divergent.

TOST is paused after 3 durable cases. Do not resume it in this sprint.

## Evidence Already Available

- `proofs/v0120/powered_tost_n15/case_*.json`
- `proofs/v0120/powered_tost_n15/pipeline_proofs/20260501_18z_l2_72h_20260519T173026Z/`
- Case 3 retained wrfouts:
  `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z/`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/`
- Current diagnostic:
  `proofs/v014/v10_grid_diagnostics.json`

Current finding: V10 grid RMSE is above 1.5 m/s in 3/3 durable cases. Case 3
also shows U10 RMSE 2.068 m/s, V10 RMSE 2.524 m/s, PSFC RMSE 525 Pa, and T2
RMSE 0.994 K. This is a grid-field divergence problem, not only a station-skill
problem.

## Scope

Allowed:

- Add or refine proof/diagnostic scripts under `proofs/v014/`.
- Add reports under `proofs/v014/` and `.agent/reviews/`.
- Read `src/gpuwrf/io/wrfout_writer.py` and runtime code to enumerate fields.
- Run CPU-only analysis against retained wrfouts.
- Run short GPU probes only if they are explicitly targeted and do not resume the
  full TOST marathon.

Not allowed:

- No model-code fix in `src/` during this sprint.
- No tolerance tuning after seeing results.
- No release/tag decision.
- No powered n=15 TOST resume.
- No FP32 dycore landing.

## Required Work Products

1. All-comparable-field envelope script:
   `proofs/v014/grid_cell_envelope.py`
2. Machine report:
   `proofs/v014/grid_cell_envelope.json`
3. Human report:
   `proofs/v014/grid_cell_envelope.md`
4. Operator-hypothesis report:
   `.agent/reviews/2026-06-08-v014-grid-parity-attribution.md`

## Minimum Field Coverage

Compare every variable written by `src/gpuwrf/io/wrfout_writer.py` that also
exists in CPU-WRF truth and has compatible dimensions. At minimum:

- Surface and diagnostics: `T2`, `Q2`, `U10`, `V10`, `PSFC`, `TSK`, `PBLH`,
  `UST`, `HFX`, `LH`, `SWDOWN`, `GLW`, `RAINC`, `RAINNC`, `RAINSH`.
- 3D dynamics/thermodynamics: `U`, `V`, `W`, `T`, `QVAPOR`, `P`, `PB`, `PH`,
  `PHB`, `MU`, `MUB`.
- Microphysics if present: `QCLOUD`, `QICE`, `QRAIN`, `QSNOW`, `QGRAUP`,
  number concentrations, and `QKE`.
- Static fields should be audited separately for exactness or metadata mismatch,
  not mixed into prognostic RMSE.

If a field is missing or dimension-incompatible, record it explicitly with the
reason.

## Metrics

For each field and lead:

- count, bias, RMSE, MAE, p95_abs, p99_abs, max_abs
- fraction within predeclared diagnostic tolerance, if one exists
- Pearson r where meaningful
- worst lead hours by RMSE
- spatial splits: land/ocean, elevation bins, quadrant, and optional coast band
- cross-field correlations for wind/pressure/temperature errors

The script must preserve the distinction between direct grid-field agreement and
station-observation skill.

## Commands

CPU-only:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 24-31 \
  python proofs/v014/grid_cell_envelope.py
```

Existing V10 diagnostic:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 24-27 \
  python proofs/v014/v10_grid_diagnostics.py
```

## Acceptance Criteria

- The envelope script runs without using GPU.
- The report includes every comparable field, not only T2/U10/V10.
- Missing or incompatible fields are enumerated, not silently skipped.
- Case 3 spatial attribution is included because retained wrfouts exist.
- Case 1/2 aggregate-only limitations are stated honestly.
- The final review ranks the top root-cause hypotheses and proposes the next
  model-code fix sprint with frozen file ownership and proof gates.

## Active Sidecars

- `Peirce` (`019ea948-6d45-78d3-b06a-bc0ad1df40ff`): prior V10 attribution synthesis.
- `Raman` (`019ea948-81c9-7161-b50c-04eaff1eb010`): cell-envelope design.
- `Heisenberg` (`019ea948-ec75-76e0-b708-44aabd02af0b`): FP32 status freeze.

## Closeout

Close with:

- commands run
- proof objects produced
- unresolved risks
- next fix sprint recommendation
- memory-patch recommendation if the grid-first process rule should become stable
  memory
