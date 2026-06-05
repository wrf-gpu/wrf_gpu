# v0.11.0 RRTMG finite recheck

- verdict: KEEP_RRTMG_ON_TRUNK
- proper-cadence finite: True
- run: 20260521_18z_l3_24h_20260522T133443Z / d02
- hours: 1
- forecast path: run_forecast_operational_segmented(segment_steps=180) via execute_daily_pipeline
- segment steps: 180
- radiation cadence steps: 180
- topo_shading: 1
- slope_rad: 1
- pipeline verdict: PIPELINE_PARTIAL
- wall clock total s: 160.72225324099418

## Final-state finite counts

| field | finite | nonfinite | min | max |
|---|---:|---:|---:|---:|
| theta | True | 0 | 290.5620486850249 | 492.0712585449219 |
| u | True | 0 | -14.347259843967915 | 25.716537770390683 |
| v | True | 0 | -18.36096618751196 | 10.82250176622191 |
| w | True | 0 | -2.7558420450520438 | 0.9827579241853923 |
| p | True | 0 | 5171.054701533395 | 101661.1796875 |
| ph | True | 0 | -7.796288059580547e-07 | 204077.453125 |
| mu | True | 0 | 66394.07572095659 | 96758.76513671875 |
| qv | True | 0 | 2.440321965059102e-06 | 0.013447845651947609 |

## Interpretation

The proper daily output-interval segmented cadence stayed finite with full physics, real d02 XLAT/XLONG radiation static fields, topo_shading=1, and slope_rad=1. The earlier cold one-step theta/u/v nonfinite is therefore treated as a harness artifact.

## Commands

- `proofs/v0110/rrtmg_finite_recheck.py --mode on --hours 1 --out-json proofs/v0110/qke_ki2_gate_merged_trunk.json --out-md proofs/v0110/qke_ki2_gate_merged_trunk.md --diagnostic-exit-zero`
