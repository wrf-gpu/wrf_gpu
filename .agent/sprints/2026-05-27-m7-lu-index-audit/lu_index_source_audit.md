# LU_INDEX Source Audit

Sprint: `2026-05-27-m7-lu-index-audit`

## Finding

`src/gpuwrf/io/land_state.py` loads `LU_INDEX` directly from the selected Gen2 `wrfinput_d02` file through `Gen2Run.load_wrfinput(domain, "LU_INDEX", lazy=False)`. The loader is not reading a different domain, not reading a different time index, and not applying a re-projection step.

For the 20260521 case, Gen2 CPU WRF preserves `LU_INDEX` exactly from `wrfinput_d02` into the first-hour `wrfout_d02`:

- `wrfinput_d02` distribution: `{5: 164, 9: 83, 10: 251, 13: 15, 16: 255, 17: 9726}`
- Gen2 first-hour `wrfout_d02` distribution: `{5: 164, 9: 83, 10: 251, 13: 15, 16: 255, 17: 9726}`
- `wrfinput_d02 == wrfout_d02[lead=1h]`: true

The GPU first-hour `wrfout_d02` distribution is `{2: 768, 17: 9726}`. Every land cell is collapsed to category `2`; every water cell remains category `17`. This is the same pattern produced by the current NetCDF writer fallback:

```python
np.where(landmask > 0.5, 2.0, 17.0)
```

The audited source path is therefore not a cast/rounding defect inside `land_state.py`. It is loss of the categorical static field between the prescribed land-state loader and wrfout emission. `State` does not carry `lu_index` or `ivgtyp`, and `src/gpuwrf/io/wrfout_writer.py` falls back to land/water defaults when those fields are absent.

## Spatial Pattern

Proof objects:

- `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_map.nc`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_summary.json`

Summary:

- max absolute category difference: `14`
- mismatched cells: `768 / 10494`
- land mismatch fraction: `1.0`
- water mismatch fraction: `0.0`
- classification: `CATEGORICAL_COLLAPSE_TO_LAND_WATER_DEFAULT`

This is not a spatial shift, random noise, or MODIS-vs-USGS remapping. It is a deterministic categorical collapse caused by missing `LU_INDEX` in the object passed to the wrfout writer.

## Minimal Fix Proposal

The minimal real fix is to preserve `LU_INDEX` as a static categorical field from `load_prescribed_land_state()` into the forecast output path:

1. Add a `lu_index` surface field to `State` or otherwise pass a static-output sidecar to `write_wrfout_netcdf`.
2. Populate it from `land.lu_index` in `build_replay_case()` and `build_initial_state()`.
3. Keep `wrfout_writer` preferring `LU_INDEX` / `lu_index` before fallback.

Those files are outside this worker contract's allowed edit set (`src/gpuwrf/contracts/**`, `src/gpuwrf/integration/**`, and `src/gpuwrf/io/wrfout_writer.py` are not owned here). Applying the full fix would violate the sprint's file ownership rule.

## Verdict

`BLOCKED`: the source audit and mismatch map are complete, but the required end-to-end fix cannot be applied honestly within the files this sprint permits this worker to modify.
