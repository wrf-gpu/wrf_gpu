# M7 Forecast-vs-Observation Scaffold Usage

This scaffold is host-side only. Run it with CPU pinning:

```bash
PYTHONPATH=src taskset -c 0-3 python scripts/m7_forecast_vs_obs_smoke.py
```

Concrete one-run invocation:

```bash
PYTHONPATH=src taskset -c 0-3 python scripts/m7_forecast_vs_obs_smoke.py \
  --run-id 20260524_18z_l3_24h_20260525T225640Z \
  --threshold-mm 1.0 \
  --window-size 9
```

The script writes:

- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/aemet_observation_inventory.json`
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/cpu_baseline_scaffold_run.json`

API example:

```python
from pathlib import Path

from gpuwrf.validation.forecast_vs_obs import (
    compute_station_scores,
    interpolate_to_stations,
    load_aemet_observations,
)

wrfout = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260524_18z_l3_24h_20260525T225640Z/wrfout_d02_2026-05-25_00:00:00")
obs = load_aemet_observations(
    "/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations",
    variables=("T2", "U10", "V10", "WIND10"),
    start_time="2026-05-25T00:00:00Z",
    end_time="2026-05-25T00:00:00Z",
)
stations = obs[["station_id", "lat", "lon", "elev_m"]].drop_duplicates("station_id")
forecast = interpolate_to_stations(
    wrfout,
    stations,
    variables=("T2", "U10", "V10", "WIND10"),
    valid_time="2026-05-25T00:00:00Z",
)
report = compute_station_scores(forecast, obs, variables=("T2", "U10", "V10", "WIND10"))
print(report.to_dict())
```

Precipitation FSS uses `RAINNC + RAINC` accumulation between the first and last WRF snapshots, AEMET precipitation summed over the same window, a 1.0 mm threshold, and a 9x9 grid-cell neighbourhood. Station precipitation is projected to nearest WRF grid cells, so the emitted FSS is a finite scaffold sanity check with a station-sparse caveat.
