# Worker Report - M7 Skill Regression RCA Codex

Summary: Implemented the diagnostic-only RCA scripts and produced all contracted proof objects. Verdict: `MULTIPLE_CONTRIBUTORS`.

## Summary

The GPU forecast is already diverged at the first output hour, so this is not a slow 24h accumulation-only regression. Lead-1 surface errors are large: `T2` max abs diff 29.34 K, `U10` 27.77 m/s, `V10` 24.82 m/s, `PSFC` mean diff -315.86 Pa / max 467.80 Pa. Boundary splitting does not implicate lateral BCs: lead-1 `U10`, `V10`, `T`, and `QVAPOR` are interior-concentrated, while `PSFC` is spatially uniform.

Single most likely divergence point for the skill regression: `(e) radiation absence`, expressed through surface/PBL coupling. Quantitative support: at lead 1, `SWDOWN` mean diff -185.36 W/m2 / max 235.21, `GLW` mean diff -331.52 W/m2 / max 349.95, `HFX` max 986.89, `LH` max 360.99, `PBLH` mean -387.13 m / max 756.04. However, the verdict remains `MULTIPLE_CONTRIBUTORS` because `LU_INDEX` also differs at lead 1 (max 14 category mismatch), and `PSFC` has an identical uniform bias in physics-on and physics-off runs.

Fix-sprint recommendation: first run a targeted radiation/surface/PBL sprint that enables or replays WRF-consistent radiation for the 1h case, audits `SWDOWN/GLW/HFX/LH/PBLH/TSK/T2`, and separately fixes the `LU_INDEX` static-field mismatch. Keep boundary application lower priority unless a later run shows boundary concentration.

## Files Changed

- `scripts/m7_rca_hour_by_hour.py`
- `scripts/m7_rca_spatial_maps.py`
- `scripts/m7_rca_physics_bracket.py`
- `tests/test_m7_rca_helpers.py`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/hour_by_hour_deviation.json`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/first_hour_diff.json`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/spatial_deviation_summary.json`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/boundary_vs_interior.json`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/physics_on_off_bracket.json`
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/worker-report.md`

External artifacts, not committed:

- `/tmp/m7_rca_artifacts/lead_01_surface_deviation.nc`
- `/tmp/m7_rca_artifacts/lead_06_surface_deviation.nc`
- `/tmp/m7_rca_artifacts/lead_12_surface_deviation.nc`
- `/tmp/m7_rca_artifacts/lead_24_surface_deviation.nc`
- `/tmp/m7_rca_artifacts/physics_bracket/**`

## Commands Run And Output

`pytest -q tests/test_m7_rca_helpers.py`

```text
...                                                                      [100%]
3 passed in 0.38s
```

`taskset -c 0-3 pytest -q tests/test_m7_rca_helpers.py`

```text
...                                                                      [100%]
3 passed in 0.37s
```

`taskset -c 0-3 python scripts/m7_rca_hour_by_hour.py`

```json
{
  "first_hour_largest_field": "HFX",
  "first_hour_largest_max_abs_diff": 986.8938541412354,
  "first_hour_output": "/tmp/wrf_gpu2_rcacodex/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/first_hour_diff.json",
  "hourly_output": "/tmp/wrf_gpu2_rcacodex/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/hour_by_hour_deviation.json",
  "lead_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
  "pair_count": 24
}
```

`taskset -c 0-3 python scripts/m7_rca_spatial_maps.py`

```json
{
  "artifact_dir": "/tmp/m7_rca_artifacts",
  "boundary_output": "/tmp/wrf_gpu2_rcacodex/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/boundary_vs_interior.json",
  "lead_count": 4,
  "spatial_output": "/tmp/wrf_gpu2_rcacodex/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/spatial_deviation_summary.json"
}
```

`taskset -c 0-3 python scripts/m7_rca_physics_bracket.py`

```json
{
  "artifact_dir": "/tmp/m7_rca_artifacts/physics_bracket",
  "fields_where_physics_off_reduced_max_abs_diff": ["T2", "T", "QVAPOR"],
  "output": "/tmp/wrf_gpu2_rcacodex/.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/physics_on_off_bracket.json",
  "verdict": "PHYSICS_BRACKET_COMPLETE"
}
```

`taskset -c 0-3 python scripts/validate_agentos.py`

```json
{
  "errors": [],
  "ok": true,
  "required_files_checked": 31,
  "skills_checked": 13
}
```

## Proof Objects Produced

- AC1: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/hour_by_hour_deviation.json`
- AC2: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/spatial_deviation_summary.json`
- AC2 maps: `/tmp/m7_rca_artifacts/lead_{01,06,12,24}_surface_deviation.nc`
- AC3: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/first_hour_diff.json`
- AC4: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/boundary_vs_interior.json`
- AC5: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/physics_on_off_bracket.json`
- AC7: `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/worker-report.md`

## Risks

- The physics-off bracket is useful only as a diagnostic bracket: it makes wind fields catastrophically worse, so it should not be interpreted as an acceptable dynamics-only mode.
- The leading radiation/surface/PBL diagnosis does not explain the uniform `PSFC` bias by itself.
- `LU_INDEX` differs despite `LANDMASK` and `HGT` matching, so static-field emission or ingestion needs a separate audit.
- NetCDF map artifacts live under `/tmp/m7_rca_artifacts/` and are not committed.

## Handoff

- objective: empirically bisect the M7 GPU-vs-CPU skill regression using existing wrfouts and short 1h bracket forecasts.
- files changed: listed above; no `src/gpuwrf/**`, governance, goal, tester, reviewer, manager, or memory files modified.
- commands run: focused helper tests, all three contracted diagnostic scripts, and AgentOS validation, with stdout captured above.
- proof objects produced: all AC1-AC5 and AC7 artifacts listed above.
- unresolved risks: radiation/surface/PBL is the leading suspect, but `LU_INDEX` and `PSFC` remain independent contributors.
- next decision needed: authorize a fix sprint focused first on WRF-consistent radiation/surface/PBL replay, with a static `LU_INDEX` audit in the same or immediately following sprint.
