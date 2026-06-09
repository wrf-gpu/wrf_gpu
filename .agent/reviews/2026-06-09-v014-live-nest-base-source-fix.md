# Review: V0.14 Live-Nest Base Source Fix

Verdict: `LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`.

## Findings

- Source patch is narrow: `d02_replay.py` adds parent-aware live-nest base init; `nested_pipeline.py` passes parent cases explicitly.
- Base fields validate against CPU-WRF h0 within the predeclared formula tolerance, not by reading h0 in production.
- Worst fixed h0 target-patch deltas: HGT `2.4167598553503922e-05`, PB `0.04890023032203317`, MUB `0.044447155625675805`, PHB `0.09328280997578986`.
- Total-state target-patch max deltas after the fix: P_TOTAL `33.43062101097894`, MU_TOTAL `12.299452038438176`, PH_TOTAL `0.09377109122578986`.
- This is not accepted as a V10/grid-parity closer: no init-override falsifier or direct grid-field proof has been run.
- Dynamic P/MU perturbation differences remain visible and remain a live suspect for the interior-wide V10 divergence.
- TOST remains paused until the grid-field symptom is directly improved and re-gated.

## Commands

```bash
python -m py_compile \
  src/gpuwrf/integration/d02_replay.py \
  src/gpuwrf/integration/nested_pipeline.py \
  src/gpuwrf/nesting/interp.py \
  src/gpuwrf/nesting/boundary_construction.py \
  proofs/v014/live_nest_base_source_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/live_nest_base_source_fix.py
python -m json.tool proofs/v014/live_nest_base_source_fix.json \
  >/tmp/live_nest_base_source_fix.validated.json
```
