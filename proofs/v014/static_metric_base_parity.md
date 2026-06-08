# V0.14 Static Metric/Base-State Parity

Generated UTC: `2026-06-08T22:48:18.772856+00:00`

CPU-only probe over Case 3 d02 retained artifacts. No GPU run was launched.

## Verdict

- Source touched: yes. `src/gpuwrf/integration/d02_replay.py` now attaches the loaded WRF `DycoreMetrics` to `GridSpec.metrics` before the case is handed to runtime/writer code.
- Pre-fix vertical C/DN/RDN and MAPFAC mismatches are writer-only: raw `GridSpec.metrics` used the analytic flat fallback, while `case.metrics`/namelist metrics loaded WRF values. The current patched synthetic writer payload matches wrfinput for those fields; the retained GPU h1 artifact still shows the old writer payload.
- `XLAT/XLONG` remain a writer payload fallback issue, separate from metric plumbing: the runtime State lacks lat/lon arrays, so the writer emits projection-derived coordinates that match the retained GPU h1 artifact.
- `HGT` mismatch is CPU wrfinput-vs-CPU-wrfout terrain convention: retained GPU/current writer follows wrfinput, while CPU wrfout h0/h1 differ from wrfinput.
- `PB/PHB/MUB` are not caused by flat metrics. `PHB` is dominated by CPU wrfinput-vs-CPU-wrfout convention; `PB/MUB` also have a retained-GPU-h1 component that cannot be split between forecast-step state drift and h1 writer base-field reconstruction without a retained GPU h0.

## Layer Counts

- `cpu_wrfinput_vs_cpu_wrfout_h0`: DIFF=4, EXACT=45
- `cpu_wrfout_h0_vs_cpu_wrfout_h1`: EXACT=49
- `cpu_h1_vs_gpu_h1`: DIFF=33, EXACT=16
- `wrfinput_vs_loaded_namelist_metrics`: EXACT=33, MISSING=11, WITHIN_TOL=5
- `wrfinput_vs_raw_grid_metrics_without_attach`: DIFF=29, EXACT=4, MISSING=11, WITHIN_TOL=5
- `wrfinput_vs_current_grid_metrics`: EXACT=33, MISSING=11, WITHIN_TOL=5
- `wrfinput_vs_prefix_synthetic_writer_payload`: DIFF=29, EXACT=20
- `prefix_synthetic_writer_payload_vs_retained_gpu_h1`: DIFF=3, EXACT=46
- `wrfinput_vs_current_synthetic_writer_payload`: DIFF=6, EXACT=43
- `current_synthetic_writer_payload_vs_retained_gpu_h1`: DIFF=26, EXACT=23

## Key Field Origins

- `C2H`: `writer_payload_grid_metrics_prefix_fixed_current_source` (high); Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces the retained GPU h1 writer payload, while the patched GridSpec.metrics and current synthetic writer payload match wrfinput. Runtime namelist metrics also match wrfinput, so this is an emitted-static-field bug in the retained artifact, not forecast-step dynamics.
- `C2F`: `writer_payload_grid_metrics_prefix_fixed_current_source` (high); Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces the retained GPU h1 writer payload, while the patched GridSpec.metrics and current synthetic writer payload match wrfinput. Runtime namelist metrics also match wrfinput, so this is an emitted-static-field bug in the retained artifact, not forecast-step dynamics.
- `C4H`: `writer_payload_grid_metrics_prefix_fixed_current_source` (high); Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces the retained GPU h1 writer payload, while the patched GridSpec.metrics and current synthetic writer payload match wrfinput. Runtime namelist metrics also match wrfinput, so this is an emitted-static-field bug in the retained artifact, not forecast-step dynamics.
- `C4F`: `writer_payload_grid_metrics_prefix_fixed_current_source` (high); Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces the retained GPU h1 writer payload, while the patched GridSpec.metrics and current synthetic writer payload match wrfinput. Runtime namelist metrics also match wrfinput, so this is an emitted-static-field bug in the retained artifact, not forecast-step dynamics.
- `RDN`: `writer_payload_grid_metrics_prefix_fixed_current_source` (high); Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces the retained GPU h1 writer payload, while the patched GridSpec.metrics and current synthetic writer payload match wrfinput. Runtime namelist metrics also match wrfinput, so this is an emitted-static-field bug in the retained artifact, not forecast-step dynamics.
- `HGT`: `cpu_wrfinput_vs_cpu_wrfout` (high); GPU/writer payload follows wrfinput, while CPU wrfout differs from CPU wrfinput and is stable from h0 to h1.
- `XLAT`: `writer_payload_latlon_fallback` (high); XLAT/XLONG are not DycoreMetrics fields. The synthetic writer payload differs from wrfinput and matches retained GPU h1, indicating the writer projection fallback was emitted because the runtime State does not carry lat/lon arrays.
- `XLONG`: `writer_payload_latlon_fallback` (high); XLAT/XLONG are not DycoreMetrics fields. The synthetic writer payload differs from wrfinput and matches retained GPU h1, indicating the writer projection fallback was emitted because the runtime State does not carry lat/lon arrays.
- `MAPFAC_M`: `writer_payload_grid_metrics_prefix_fixed_current_source` (high); Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces the retained GPU h1 writer payload, while the patched GridSpec.metrics and current synthetic writer payload match wrfinput. Runtime namelist metrics also match wrfinput, so this is an emitted-static-field bug in the retained artifact, not forecast-step dynamics.
- `PB`: `cpu_wrfinput_vs_cpu_wrfout_plus_retained_gpu_h1_forecast_or_writer_reconstruction` (medium); CPU wrfout h0/h1 differ from wrfinput, but retained GPU h1 also differs from a zero-step writer reconstruction. With no retained GPU h0 frame, the GPU-side component cannot be split between forecast-step state drift and h1 writer base-field reconstruction.
- `PHB`: `cpu_wrfinput_vs_cpu_wrfout` (high); Current writer reconstruction follows wrfinput, retained GPU h1 is also near wrfinput, and CPU wrfout differs from wrfinput while staying stable from h0 to h1.
- `MUB`: `cpu_wrfinput_vs_cpu_wrfout_plus_retained_gpu_h1_forecast_or_writer_reconstruction` (medium); CPU wrfout h0/h1 differ from wrfinput, but retained GPU h1 also differs from a zero-step writer reconstruction. With no retained GPU h0 frame, the GPU-side component cannot be split between forecast-step state drift and h1 writer base-field reconstruction.

## Worst CPU h1 vs GPU h1 Differences

| field | max abs | RMSE | bias | origin |
| --- | ---: | ---: | ---: | --- |
| `C2H` | 95000 | 70062.4 | 30445.5 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C2F` | 95000 | 69896.2 | 30714.7 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C4F` | 26782.8 | 13942 | 10372.3 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C4H` | 26740.1 | 14080.2 | 10608.1 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `PHB` | 2237.94 | 45.3525 | 0.640949 | `cpu_wrfinput_vs_cpu_wrfout` |
| `MUB` | 1115.21 | 58.7687 | -3.18983 | `cpu_wrfinput_vs_cpu_wrfout_plus_retained_gpu_h1_forecast_or_writer_reconstruction` |
| `PB` | 1111.71 | 28.6425 | -0.960326 | `cpu_wrfinput_vs_cpu_wrfout_plus_retained_gpu_h1_forecast_or_writer_reconstruction` |
| `HGT` | 228.129 | 8.46292 | 0.182259 | `cpu_wrfinput_vs_cpu_wrfout` |
| `RDN` | 161.674 | 24.3732 | 3.67441 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C1H` | 1.02545 | 0.482756 | 0.2668 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C1F` | 1.01942 | 0.476954 | 0.262026 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C3F` | 0.281924 | 0.146758 | -0.109182 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `C3H` | 0.281475 | 0.148213 | -0.111664 | `writer_payload_grid_metrics_prefix_fixed_current_source` |
| `XLAT_U` | 0.0276241 | 0.0100346 | -0.00716086 | `writer_payload_latlon_fallback` |
| `XLAT_V` | 0.0273514 | 0.009919 | -0.00707389 | `writer_payload_latlon_fallback` |
| `XLAT` | 0.0273228 | 0.00991665 | -0.00707291 | `writer_payload_latlon_fallback` |

## Runtime/Writer Split

- Pre-fix witness: `Gen2Run.grid(...).as_grid_spec()` constructs `GridSpec` before loaded metrics are available; `GridSpec.__post_init__` fills missing metrics with `DycoreMetrics.flat`.
- Current source path: `build_replay_case` loads `DycoreMetrics` from the same metrics source and replaces `grid.metrics` with that payload before `State.zeros` and before writer bundles are built.
- Runtime dynamics: nested pipeline still passes `metrics=case.metrics` into `OperationalNamelist.from_grid`; this proof found no evidence that dynamics consumed flat vertical metrics.
- The nested writer calls `write_wrfout_netcdf(state, grid, namelist, ...)`, and `_add_grid_coordinate_fields` reads `grid.metrics`, not `namelist.metrics`.
- The retained GPU h1 wrfout predates this local source patch, so it is used as a stale-artifact witness, not as a post-fix output claim.

## Limits

- No retained GPU h0 wrfout exists, so PB/PHB/MUB h1 differences cannot be split into post-step state drift versus h1 writer reconstruction with only retained artifacts.
- build_replay_case itself cannot run under JAX_PLATFORMS=cpu because State.zeros requires a visible GPU; this proof reconstructs the relevant GridSpec and loaded DycoreMetrics CPU-only instead.
- No fresh GPU writer smoke was launched. Retained GPU h1 predates the source patch and cannot demonstrate grid-cell-envelope improvement without a new GPU/writer artifact.
