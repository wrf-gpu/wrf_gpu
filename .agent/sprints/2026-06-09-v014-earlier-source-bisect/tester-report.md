# Tester Report

## Tests Added Or Run

- `python -m py_compile proofs/v014/earlier_source_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/earlier_source_bisect.py`
- `python -m json.tool proofs/v014/earlier_source_bisect.json >/tmp/earlier_source_bisect.manager.validated.json`

## Results

Decision: pass for an evidence sprint.

The CPU command regenerated JSON/Markdown/review from the compact GPU replay
artifact and preserved `BASE_STATE_SPLIT_DEFINITION_MISMATCH`. JSON validation
passed.

Key checks:

- Initial native carry `PB/MUB` match native `wrfinput_d02`: `true`.
- Initial native carry `PB/MUB` match CPU-WRF h0/static base truth: `false`.
- Worst initial base field is `MUB`, max_abs `1050.3046875`.
- The same `MUB` max_abs remains present through d02 step 5997.

## Fixtures Used

- Native L2 run root under `/tmp/v0120_merged_run_root`.
- CPU-WRF hourly backfill outputs under
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/`.
- CPU-WRF h10 pre-RK hook truth from `proofs/v014/pre_rk_input_boundary.json`.

## Gaps

The proof names the source surface, not the source-code formula. The next sprint
must patch or derive WRF's post-initialization base-state split and prove it
against the same initial-carry and h10 pre-RK checks.

## Decision

Accept. This is sufficient to move from evidence bisection to a narrow
source-changing fix sprint.
