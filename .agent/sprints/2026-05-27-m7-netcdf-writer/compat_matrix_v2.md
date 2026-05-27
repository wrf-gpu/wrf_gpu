# M7 NetCDF Writer Compatibility Matrix v2

Generated UTC: 2026-05-27T00:50:53+00:00
CPU reference: `/mnt/data/canairy_meteo/runs/wrf_l3/20260525_18z_l3_24h_20260526T221207Z/wrfout_d02_2026-05-25_18:00:00`
Candidate writer output: `/tmp/wrf_gpu2_ncwriter_m7_netcdf_writer/wrfout_d02_2026-05-25_18:00:00`

## Summary

- Minimum variable count: 41
- Downstream-critical missing fields: 0
- AC1 minimum missing fields: 0
- AC1 dimension mismatches: 0
- AC1 dtype mismatches: 0
- AC1 metadata mismatches: 0
- Verdict: PASS

## Minimum Variable Rows

| Variable | Downstream-critical | Reference dims | Candidate dims | Dtype match | Attr match |
|---|---:|---|---|---:|---:|
| `Times` | YES | `['Time', 'DateStrLen']` | `['Time', 'DateStrLen']` | YES | YES |
| `XTIME` | YES | `['Time']` | `['Time']` | YES | YES |
| `XLAT` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `XLONG` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `HGT` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `LANDMASK` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `LU_INDEX` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `U10` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `V10` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `T2` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `Q2` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `PSFC` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `RAINC` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `RAINNC` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `RAINSH` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `SWDOWN` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `GLW` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `PBLH` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `UST` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `HFX` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `LH` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `TSK` | YES | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `CLDFRA` | YES | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `QCLOUD` | YES | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `QICE` | YES | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `QRAIN` | YES | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `XLAT_U` | NO | `['Time', 'south_north', 'west_east_stag']` | `['Time', 'south_north', 'west_east_stag']` | YES | YES |
| `XLONG_U` | NO | `['Time', 'south_north', 'west_east_stag']` | `['Time', 'south_north', 'west_east_stag']` | YES | YES |
| `XLAT_V` | NO | `['Time', 'south_north_stag', 'west_east']` | `['Time', 'south_north_stag', 'west_east']` | YES | YES |
| `XLONG_V` | NO | `['Time', 'south_north_stag', 'west_east']` | `['Time', 'south_north_stag', 'west_east']` | YES | YES |
| `U` | NO | `['Time', 'bottom_top', 'south_north', 'west_east_stag']` | `['Time', 'bottom_top', 'south_north', 'west_east_stag']` | YES | YES |
| `V` | NO | `['Time', 'bottom_top', 'south_north_stag', 'west_east']` | `['Time', 'bottom_top', 'south_north_stag', 'west_east']` | YES | YES |
| `W` | NO | `['Time', 'bottom_top_stag', 'south_north', 'west_east']` | `['Time', 'bottom_top_stag', 'south_north', 'west_east']` | YES | YES |
| `T` | NO | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `QVAPOR` | NO | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `P` | NO | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `PB` | NO | `['Time', 'bottom_top', 'south_north', 'west_east']` | `['Time', 'bottom_top', 'south_north', 'west_east']` | YES | YES |
| `PH` | NO | `['Time', 'bottom_top_stag', 'south_north', 'west_east']` | `['Time', 'bottom_top_stag', 'south_north', 'west_east']` | YES | YES |
| `PHB` | NO | `['Time', 'bottom_top_stag', 'south_north', 'west_east']` | `['Time', 'bottom_top_stag', 'south_north', 'west_east']` | YES | YES |
| `MU` | NO | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
| `MUB` | NO | `['Time', 'south_north', 'west_east']` | `['Time', 'south_north', 'west_east']` | YES | YES |
