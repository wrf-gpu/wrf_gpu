# Worker Report

## Summary

Summary:

Verdict: `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.

The worker proved the bad d02 base-state leaves are present at initial native
load / carry construction, before replay-time drift is needed. The initial JAX
child carry matches the native `wrfinput_d02` `PB/MUB` split on the target
patch, but CPU-WRF h0, h1, h10 wrfout, and the h10 pre-RK hook share a stable
different `PB/MUB` split.

## Files Changed

- `proofs/v014/earlier_source_bisect.py`
- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`

## Commands Run

- `python -m py_compile proofs/v014/earlier_source_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/earlier_source_bisect.py`
- `python -m json.tool proofs/v014/earlier_source_bisect.json >/tmp/earlier_source_bisect.validated.json`
- `WRFGPU2_EARLIER_SOURCE_BISECT_ALLOW_GPU=1 WRFGPU2_EARLIER_SOURCE_BISECT_FORCE_REPLAY=1 JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src python proofs/v014/earlier_source_bisect.py`

## Proof Objects

- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`
- `/mnt/data/wrf_gpu2/v014_earlier_source_bisect/earlier_source_bisect.live_replay_compact.json`

## Risks

- Dynamic `T/P/MU` at d02 step 5997 still lack exact same-step CPU-WRF internal
  truth; this sprint classifies `PB/MUB` as static/base leaves.
- The proof is patch-local to the existing h10 target patch, not a full-grid
  validation campaign.
- Live replay still requires GPU because `_load_domains` reaches
  `State.zeros`.

## Handoff

Open a source-changing fix sprint for
`src/gpuwrf/integration/d02_replay.py::build_replay_case` native child
base-state split construction. The fix must reproduce WRF's post-initialization
`PB/MUB` split or explicitly justify an accepted h0 base-state oracle path.
