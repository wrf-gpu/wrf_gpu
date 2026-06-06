# v0.11.0 d03 1 km validation

## Verdict

**D03_1KM_STABLE**. The donation-safe gated-fp32 d03 Tenerife replay reached **24 h** with `all_finite=true`; no qke non-finite recurrence was observed. Full-fp64 fallback was **not run** because the gated-fp32 lane did not go non-finite.

## Run

- branch: `worker/gpt/v0110-val-d031km`
- head: `3d5c6ae (HEAD -> worker/gpt/v0110-val-d031km, worker/opus/v0110-integration) v0.12.0 plan FINAL: Scope A locked (common+fail-closed) + GPT-review gaps folded + difficulty-rated sprint table`
- run_id: `20260521_18z_l3_24h_20260522T133443Z`
- domain/grid: d03 Tenerife 1 km, mass grid 75x93, 44 vertical levels
- command wrapper: `/tmp/wrf_gpu_run_lowprio.sh`
- dt/acoustic: 3.0 s / 10 substeps
- force_fp64: `False` (gated-fp32 mode)
- wrfouts: 24/24
- wall clock: 1654.060 s total, 68.919 s per forecast hour

## Finite/Stability

- lead reached: 24 h, final valid time `2026-05-22T18:00:00+00:00`
- all finite: `True`
- qke: dtype `float64`, nonfinite_count `0`, min `1e-05`, max `124.52927886424234`
- qke cold-start seed: `True`

## Skill vs Corpus d03 Truth

| field | final RMSE | threshold | final within | mean RMSE | max RMSE | beats persistence all leads |
|---|---:|---:|---|---:|---:|---|
| T2 | 1.61456 K | 3.0 | True | 1.32034 | 1.97982 | True |
| U10 | 5.13282 m s-1 | 7.5 | True | 4.00168 | 5.37119 | False |
| V10 | 6.6252 m s-1 | 7.5 | True | 4.85051 | 6.8102 | False |

RAINNC was scored informationally only: final RMSE `0.684518 mm`; no threshold.

## Notes

- Initial gated-fp32 attempt was blocked before forecast by JAX buffer donation aliasing: `JaxRuntimeError: INVALID_ARGUMENT: Attempt to donate the same buffer twice in Execute() (flattened argument 8, replica 0, partition 0, first use: 5). Toy example for this bug: `f(donate(a), donate(a))`.`
- `scripts/d03_replay.py` now uses a validation-harness donation-safe forecast wrapper that clones State leaves before the donated JIT call; model numerics are unchanged.
- The raw pipeline verdict is `PIPELINE_PARTIAL` because station scoring was not requested; gridded corpus validation status is `PASS`.
- Full-fp64 fallback was not run because the gated-fp32 validation stayed finite for the full target lead.

## Proof Objects

- `proofs/v0110/d031km_v0110.json`
- `proofs/v0110/d03_summary_d031km_gated_fp32_clone.json`
- `proofs/v0110/pipeline_run_d03_d031km_gated_fp32_clone.json`
- `proofs/v0110/d03_validation_d031km_gated_fp32_clone.json`
- `proofs/v0110/d031km_gated_fp32_clone.log`
