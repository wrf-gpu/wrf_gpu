# Review: V0.14 Earlier-Source Bisection

verdict: `BASE_STATE_SPLIT_DEFINITION_MISMATCH`

objective: bisect whether the bad h10 d02 OperationalCarry source is native load/initial carry or an earlier replay segment before completed step 5997.

files changed:
- `proofs/v014/earlier_source_bisect.py`
- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`

commands run:
- `python -m py_compile proofs/v014/earlier_source_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/earlier_source_bisect.py`
- `python -m json.tool proofs/v014/earlier_source_bisect.json >/tmp/earlier_source_bisect.validated.json`
- `WRFGPU2_EARLIER_SOURCE_BISECT_ALLOW_GPU=1 WRFGPU2_EARLIER_SOURCE_BISECT_FORCE_REPLAY=1 JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src python proofs/v014/earlier_source_bisect.py`

proof objects produced:
- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`
- `/mnt/data/wrf_gpu2/v014_earlier_source_bisect/earlier_source_bisect.live_replay_compact.json`

unresolved risks:
- Dynamic T/P/MU at d02 step 5997 still lack same-step CPU-WRF internal truth; only PB/MUB are classified there as static base fields.
- The replay required a targeted GPU run because State.zeros/_load_domains is not CPU-capable in this branch.
- The conclusion is patch-local to the existing h10 target patch, not a full-grid validation campaign.

next decision needed: Open a source-changing fix sprint for src/gpuwrf/integration/d02_replay.py::build_replay_case native child base-state split construction; reproduce WRF's post-initialization PB/MUB split or load an accepted h0 base-state oracle before replay.
