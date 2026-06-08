# Canary Existing-Data Stats Scripts

CPU-only utilities for inventorying and re-scoring existing Canary CPU-WRF and GPUWRF outputs.

Run Python commands with the project CPU rule, for example:

```bash
taskset -c 29-31 env JAX_PLATFORMS=cpu python scripts/canary_stats/inventory_existing_runs.py
```

Scripts:

- `inventory_existing_runs.py` scans existing run/output roots and writes JSON, CSV, and a Markdown table.
- `assess_pairability.py` classifies raw CPU/GPU pairability by case, domain, common leads, variables, and grid shape.
- `score_grid_pairs.py` computes paired gridded RMSE/bias/correlation on exact common valid times.
- `score_station_tost_pairs.py` reuses the ADR-029 station complete-pair scorer on existing wrfouts only.
- `make_summary_report.py` creates a compact Markdown/plot summary from inventory and pairability artifacts.
