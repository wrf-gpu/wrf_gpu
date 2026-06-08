# V0.14 Base-State Writer Attribution

Generated UTC: `2026-06-08T23:20:26.547870+00:00`

CPU-only NetCDF probe. No GPU, no model execution, no source edits.

## Verdict

- No runtime input mismatch: CPU `wrfinput_d02` and GPU run-root `wrfinput_d02` are exact for all six target fields.
- No additional base-state source fix is required before same-state dynamic localization.
- Proceed with documented exclusions: treat `PHB`/`HGT` as CPU output-convention fields, `XLAT`/`XLONG` as writer-fallback fields, and `PB`/`MUB` as one-hour forecast state-split symptoms.

## Classifications

| Field | Classification | Key evidence |
| --- | --- | --- |
| `PHB` | `cpu_output_convention` | CPU wrfout h0/h1 PHB are static but differ from wrfinput; fresh GPU h1 follows wrfinput to fp32 writer roundoff (max_abs 0.015625). |
| `MUB` | `forecast_step_change` | CPU and GPU wrfinputs are exact and CPU base field is h0-h1 static, but fresh GPU h1 differs from its native wrfinput. The writer reconstructs this base field from evolved total-minus-perturbation state at output time, so this is a one-hour state-split symptom rather than a static input mismatch. |
| `PB` | `forecast_step_change` | CPU and GPU wrfinputs are exact and CPU base field is h0-h1 static, but fresh GPU h1 differs from its native wrfinput. The writer reconstructs this base field from evolved total-minus-perturbation state at output time, so this is a one-hour state-split symptom rather than a static input mismatch. |
| `HGT` | `cpu_output_convention` | GPU h1 terrain is byte-exact to the run-root wrfinput; CPU wrfout h0/h1 are byte-exact to each other but differ from wrfinput. |
| `XLAT` | `writer_fallback` | CPU wrfinput/wrfout lat-lon are exact, but fresh GPU h1 is byte-exact to the writer projection fallback generated from GPU writer attrs, not to wrfinput. |
| `XLONG` | `writer_fallback` | CPU wrfinput/wrfout lat-lon are exact, but fresh GPU h1 is byte-exact to the writer projection fallback generated from GPU writer attrs, not to wrfinput. |

## Exact Files

- `cpu_wrfinput`: `/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z/wrfinput_d02`
- `gpu_native_wrfinput`: `/tmp/v0120_merged_run_root/20260501_18z_l2_72h_20260519T173026Z/wrfinput_d02`
- `cpu_wrfout_h0`: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_18:00:00`
- `cpu_wrfout_h1`: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_19:00:00`
- `gpu_wrfout_h1`: `/tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_19:00:00`

Full comparison tables, finite coverage, p99/max/worst-cell statistics, writer-fallback tests, and derived state-split totals are in `proofs/v014/base_state_writer_attribution.json`.
