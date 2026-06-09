# Tester Report

## Tests Added Or Run

The proof script `proofs/v014/live_nest_base_source_fix.py` was rerun by the
manager after correcting the verdict language. It validates the patched native
init path against CPU-WRF h0/h10 as oracles and records target-patch plus
whole-domain stats.

Commands:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/nested_pipeline.py src/gpuwrf/nesting/interp.py src/gpuwrf/nesting/boundary_construction.py proofs/v014/live_nest_base_source_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_source_fix.py`
- `python -m json.tool proofs/v014/live_nest_base_source_fix.json >/tmp/live_nest_base_source_fix.manager.validated.json`

## Results

Verdict is
`LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`.

Fixed target-patch deltas vs CPU-WRF h0:

- HGT: `2.4167598553503922e-05` m
- PB: `0.04890023032203317` Pa
- MUB: `0.044447155625675805` Pa
- PHB: `0.09328280997578986`

Fixed target-patch total-state deltas also improve:

- P_TOTAL: `1080.4921875` -> `33.43062101097894` Pa
- MU_TOTAL: `1038.0496826171875` -> `12.299452038438176` Pa
- PH_TOTAL: `878.0291748046875` -> `0.09377109122578986`

JSON validates. No GPU was used. No TOST or Switzerland validation was run.

## Fixtures Used

- Native inputs under `/tmp/v0120_merged_run_root/20260501_18z_l2_72h_20260519T173026Z`
- CPU-WRF truth under `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

## Gaps

No init-override falsifier or direct V10/grid-field proof was run on this patch.
No station TOST or Switzerland demo validation was run. Dynamic P/MU residuals
remain a live suspect.

Decision:

Accept the proof only for base-state source correctness. Do not treat it as
grid-parity or V10 closure.
