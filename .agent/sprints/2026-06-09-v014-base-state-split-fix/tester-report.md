# Tester Report

## Tests Added Or Run

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/base_state_split_fix.py`
- `python -m json.tool proofs/v014/base_state_split_fix.json >/tmp/base_state_split_fix.manager.validated.json`

## Results

Decision: pass for a blocked proof sprint.

The proof is CPU-only, regenerates JSON/Markdown/review, and validates. It
records no `src/gpuwrf/integration/d02_replay.py` diff.

Key results:

- Validation-only WRF h0-HGT formula matches CPU-WRF h0 base fields tightly:
  `PB` patch max `0.04889917548280209` Pa, `MUB` patch max
  `0.044447155625675805` Pa.
- Simplified local bilinear+blend reconstruction is rejected: `PB` patch max
  `796.2565574348409` Pa, `MUB` patch max `798.7609739865584` Pa.

## Fixtures Used

- Native `wrfinput_d01/d02`.
- CPU-WRF h0/h1/h10 backfill outputs.
- Existing pre-RK and earlier-source proof artifacts.
- WRF source copies under `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF` and
  `/mnt/data/wrf_gpu2/v014_same_state_wrf/WRF`.

## Gaps

The exact post-blend/pre-start-domain WRF fields are not yet captured as a
portable oracle. That is the next sprint.

## Decision

Accept blocked proof. Do not patch `build_replay_case` until the WRF live-nest
initialization oracle or native port is available.
