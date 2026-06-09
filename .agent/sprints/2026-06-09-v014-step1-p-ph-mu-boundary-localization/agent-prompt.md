You are Opus 4.8 xhigh, correctness/debug worker for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`
Sprint: `.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization`

Read first:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization/sprint-contract.md`
5. `proofs/v014/step1_live_nest_theta_qv_wiring.md`
6. Only the prior proof/source files needed for the task.

Objective:

Localize or narrowly fix the remaining d02 Step-1 strict same-input divergence
after production live-nest theta/QV initialization closure. Current baseline:
first divergent schema field `T`; largest residual `P` max_abs
`974.9820434775493`, RMSE `135.98147360593399`, worst Fortran
`i=1,j=30,k=1`, boundary band true; `PH/MU/W/U` remain material.

Rules:

- CPU only unless manager explicitly authorizes a short GPU check later.
- No TOST, no Switzerland, no FP32 source work, no memory source work, no
  Hermes/Telegram.
- No broad rewrite. No timestep-loop host/device transfer. No CPU-WRF runtime
  dependency.
- You may create disposable WRF scratch instrumentation only under
  `/mnt/data/wrf_gpu2/v014_step1_p_ph_mu_boundary_localization/**`; commit only
  a diff if used.
- Production source edits are optional only if you prove an exact narrow bug and
  stay inside the contract write scope. If you edit source, rerun the Step-1
  proof and report before/after top residuals plus performance implications.

Method expectation:

Use the fastest rigorous wall-clock path. Prefer a focused Step-1
boundary/substage comparator reusing existing truth surfaces and prior
instrumentation over slow free-running forecasts. Explicitly distinguish
boundary-package construction, boundary application, RK tendency/source,
small-step/acoustic, pressure refresh, and comparison/schema issues. Avoid
JAX-vs-JAX-only conclusions.

Required outputs:

- `proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`
- optional `proofs/v014/step1_p_ph_mu_boundary_localization_wrf_patch.diff`

Validation commands at minimum:

```bash
python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_p_ph_mu_boundary_localization.py
python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json \
  >/tmp/step1_p_ph_mu_boundary_localization.validated.json
git diff -- src/gpuwrf
```

Final report must include:

- objective
- files changed
- commands run
- proof objects produced
- unresolved risks
- next decision needed, if any
- one final verdict from the contract

When done, notify the manager pane exactly with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'OPUS STEP1_P_PH_MU_BOUNDARY_LOCALIZATION DONE - see proofs/v014/step1_p_ph_mu_boundary_localization.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
