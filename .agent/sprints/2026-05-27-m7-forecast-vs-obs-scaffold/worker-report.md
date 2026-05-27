# Worker Report — M7 Forecast-vs-Observation Verification Scaffold

Summary: SCAFFOLD_READY. Implemented the host-side forecast-vs-observation scaffold for one Gen2 CPU WRF d02 day, including AEMET inventory, bilinear forecast-to-station interpolation, station BIAS/RMSE/MAE scoring, and 1.0 mm / 9x9-cell precipitation FSS. This is scaffold validation only, not a GPU forecast or CPU skill claim.

## Files Changed

- `src/gpuwrf/validation/forecast_vs_obs.py`
- `src/gpuwrf/validation/__init__.py`
- `scripts/m7_forecast_vs_obs_smoke.py`
- `tests/test_m7_forecast_vs_obs.py`
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/aemet_observation_inventory.json`
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/cpu_baseline_scaffold_run.json`
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/usage.md`
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/worker-report.md`

## Commands Run

`taskset -c 0-3 pytest -q tests/test_m7_forecast_vs_obs.py`

stdout/stderr:
```text
.......                                                                  [100%]
7 passed in 0.51s
```

`PYTHONPATH=src taskset -c 0-3 python scripts/m7_forecast_vs_obs_smoke.py`

stdout/stderr:
```text
{'verdict': 'SCAFFOLD_READY', 'inventory_path': '.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/aemet_observation_inventory.json', 'run_path': '.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/cpu_baseline_scaffold_run.json'}
```

`taskset -c 0-3 python scripts/validate_agentos.py`

stdout/stderr:
```json
{
  "errors": [],
  "ok": true,
  "required_files_checked": 31,
  "skills_checked": 13
}
```

## Proof Objects Produced

- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/aemet_observation_inventory.json`: 106 station parquet files, 106 stations, variables present for T2/U10/V10/WIND10/PRECIP, coverage `2006-05-11T00:00:00+00:00` to `2026-05-26T23:00:00+00:00`, bbox lat 27.665278..29.231944 lon -18.115..-13.489167.
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/cpu_baseline_scaffold_run.json`: scored `20260524_18z_l3_24h_20260525T225640Z` over 25 wrfout files, 73 stations, 1820 joined station rows. Station score status OK; precip FSS finite with station-sparse caveat.
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/usage.md`: CLI and API invocation details.

## Risks

- Precipitation FSS is finite but station-projected, with observed grid coverage about 0.0069; use as scaffold sanity until a gridded precipitation observation product is selected.
- The AEMET wind conversion assumes meteorological direction degrees-from, converted to U/V by standard convention.
- Dependencies `pandas` and parquet support are assumed per sprint contract but are not declared in `pyproject.toml`.

## Handoff

Objective: build M7 forecast-vs-observation scaffold and validate it against one Gen2 CPU WRF day.

Proof status: SCAFFOLD_READY; tests and smoke run pass under CPU pinning.

Unresolved risks: station-sparse precip FSS reliability and undeclared pandas/parquet dependency.

Next decision needed: manager/tester should decide whether station-projected FSS is enough for the M7 scaffold gate or whether the next sprint should wire a gridded precip observation source.
