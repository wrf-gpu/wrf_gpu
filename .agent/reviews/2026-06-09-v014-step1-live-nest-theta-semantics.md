# Review: V0.14 Step-1 Live-Nest Theta Semantics

Verdict: `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.

objective: prove whether WRF live-nest `T_STATE`/theta semantics after terrain/base blending close the accepted WRF pre-call residual.

files changed:
- `proofs/v014/step1_live_nest_theta_semantics.py`
- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_live_nest_theta_semantics.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`

commands run:
- `python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_semantics.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json >/tmp/step1_live_nest_theta_semantics.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_live_nest_theta_semantics.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_live_nest_theta_semantics.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth` reused, not rebuilt

unresolved risks:
- WRF theta_m conversion plus adjust_tempqv reduces T_STATE max_abs by about three orders of magnitude but leaves a millikelvin-scale residual above the prior 1e-3 K material gate.
- No production source patch was made under the sprint hard constraint because the proof-local candidate did not fully close T_STATE.
- Accepted WRF pre-call QVAPOR truth is absent from the named truth schema, so QVAPOR closure is report-only against wrfout H0.

next decision needed: Add or reuse an accepted WRF pre-call QVAPOR/savepoint truth and isolate the remaining T_STATE millikelvin residual before patching production.
