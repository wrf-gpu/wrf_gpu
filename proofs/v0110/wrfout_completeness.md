# v0.11.0 wrfout completeness proof

## Result

PASS for the scoped KI-3 closure.

This patch extends the v0.10.0 writer registry from 76 known operational wrfout variables in this base to 85. The nine added variables are the KI-3 gaps:

- Noah-MP snow-layer diagnostics: `TSNO`, `SNICE`, `SNLIQ`
- Noah-MP snow+soil interface diagnostic: `ZSNSO`
- stochastic restart seed arrays: `ISEEDARR_SPPT`, `ISEEDARR_SKEBS`, `ISEEDARRAY_SPP_CONV`, `ISEEDARRAY_SPP_PBL`, `ISEEDARRAY_SPP_LSM`

The writer and restart writer still emit only fields with real sources. They do not fabricate inactive stochastic seed state or land diagnostics when the matching `land_state` / diagnostic/seed source is absent.

## Reference Inventory

CPU-WRF reference file:
`<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d02_2026-05-21_18:00:00`

Reference variable count: 375.

Reference KI-3 dimensions:

- `seed_dim_stag = 8`
- `snow_layers_stag = 3`
- `snso_layers_stag = 7`
- `soil_layers_stag = 4`

Reference KI-3 variable schemas:

| Variable | Reference dims | Reference dtype | gpuwrf dims | gpuwrf dtype |
| --- | --- | --- | --- | --- |
| `TSNO` | `Time,snow_layers_stag,south_north,west_east` | `float32` | same | `f4` |
| `SNICE` | `Time,snow_layers_stag,south_north,west_east` | `float32` | same | `f4` |
| `SNLIQ` | `Time,snow_layers_stag,south_north,west_east` | `float32` | same | `f4` |
| `ZSNSO` | `Time,snso_layers_stag,south_north,west_east` | `float32` | same | `f4` |
| `ISEEDARR_SPPT` | `Time,seed_dim_stag` | `int32` | same | `i4` |
| `ISEEDARR_SKEBS` | `Time,seed_dim_stag` | `int32` | same | `i4` |
| `ISEEDARRAY_SPP_CONV` | `Time,seed_dim_stag` | `int32` | same | `i4` |
| `ISEEDARRAY_SPP_PBL` | `Time,seed_dim_stag` | `int32` | same | `i4` |
| `ISEEDARRAY_SPP_LSM` | `Time,seed_dim_stag` | `int32` | same | `i4` |

## Persistent State Field Order

The exact restart ABI enumerates all 56 persistent `State` fields:

`u`, `v`, `w`, `theta`, `qv`, `p`, `ph`, `mu`, `p_total`, `p_perturbation`, `ph_total`, `ph_perturbation`, `mu_total`, `mu_perturbation`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `Ns`, `Ng`, `qke`, `ustar`, `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc`, `fltv`, `t_skin`, `soil_moisture`, `xland`, `lakemask`, `mavail`, `roughness_m`, `rain_acc`, `snow_acc`, `graupel_acc`, `ice_acc`, `u_bdy`, `v_bdy`, `theta_bdy`, `qv_bdy`, `ph_bdy`, `mu_bdy`, `w_bdy`, `p_bdy`, `pb_bdy`, `phb_bdy`, `mub_bdy`, `lu_index`, `Nc`, `Nn`, `rainc_acc`.

The exact restart carry also persists the promoted operational scratch/radiation fields:

`t_2ave`, `ww`, `mudf`, `muave`, `muts`, `ph_tend`, `u_save`, `v_save`, `w_save`, `t_save`, `ph_save`, `mu_save`, `ww_save`, `rthraten`.

Optional exact carry groups are manifest-driven and fail-closed:

`noahmp_land`, `noahmp_rad`, `cumulus_carry`, `noahclassic_land`, `noahclassic_rad`.

WRF stochastic seed arrays are now WRF-named optional `wrfrst` variables, guarded by `GPUWRF_STOCHASTIC_SEED_VARIABLE_ORDER`, and read back through `read_wrfrst_stochastic_seeds()`. A CPU structural proof wrote all five seed arrays into `wrfrst`, read them back, and confirmed exact integer equality for:

`ISEEDARR_SPPT`, `ISEEDARR_SKEBS`, `ISEEDARRAY_SPP_CONV`, `ISEEDARRAY_SPP_PBL`, `ISEEDARRAY_SPP_LSM`.

## Commands

- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 pytest -q tests/test_v0110_wrfrst_netcdf.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 python scripts/v0110_restart_proof.py --output /tmp/v0110_restart_seed_structural.json --skip-forecast`
- `taskset -c 0-27 python - <<'PY' ... netCDF4 reference inventory ...`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 python - <<'PY' ... writer registry inventory ...`

## Remaining Gaps

This is not a claim that gpuwrf now writes all 375 CPU-WRF wrfout variables. It closes the Tier-1 item #2 KI-3 dimensions and variables, preserves WRF-compatible schemas for the added fields, and documents the exact persistent restart ABI for future fused-kernel/state work.
