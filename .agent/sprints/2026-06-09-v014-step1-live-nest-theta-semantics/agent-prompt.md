You are GPT-5.5 xhigh, debugging/fix worker for wrf_gpu2 v0.14 grid-parity.

Read, in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/sprints/2026-06-09-v014-step1-live-nest-theta-semantics/sprint-contract.md
5. proofs/v014/step1_jax_loader_tstate.md
6. proofs/v014/step1_jax_loader_tstate.json
7. proofs/v014/step1_jax_loader_tstate.py
8. src/gpuwrf/integration/d02_replay.py
9. /home/enric/src/wrf_pristine/WRF/share/mediation_integrate.F around live-nest `adjust_tempqv`
10. /home/enric/src/wrf_pristine/WRF/dyn_em/nest_init_utils.F `adjust_tempqv`

Objective:
Prove and, if proven, port WRF live-nest `T_STATE`/theta semantics after
terrain/base blending. The trigger proof says JAX updates PB/PHB/MUB but
carries raw wrfinput theta unchanged:
`STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

Hard constraints:
- Verify `git log -1` and that commit `7ae33eda` is an ancestor.
- CPU only: `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work.
- No Hermes/Telegram.
- Use scratch only under `/mnt/data/wrf_gpu2/v014_step1_live_nest_theta_semantics/**`.
- No source patch unless a proof-local candidate formula closes WRF pre-call
  `T_STATE` residual.
- If source is edited, rerun the full proof chain named in the contract.
- Do not touch unrelated untracked files.

Fastest rigorous method:
Transcribe WRF `adjust_tempqv` proof-locally first. Inputs should mirror WRF:
raw `MUB` as `save_mub`, live-nest recomputed `MUB` as `mub`, raw perturbation
`P` as `pp`, raw `T` as `th`, raw `QVAPOR` as `qv`, and real `c3h/c4h/p_top`.
Compare raw/current/candidate `T_STATE`, `QVAPOR`, and continuity fields against
the accepted WRF pre-call truth. Resolve `use_theta_m` explicitly.

Deliverables:
- `proofs/v014/step1_live_nest_theta_semantics.py`
- `proofs/v014/step1_live_nest_theta_semantics.json`
- `proofs/v014/step1_live_nest_theta_semantics.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`
- optional targeted source patch only if proven

Validation:
Run at minimum:
`python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py`
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_semantics.py`
`python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json >/tmp/step1_live_nest_theta_semantics.validated.json`
`git diff -- src/gpuwrf`

If source is edited, run all proof-chain commands in the sprint contract.

Final output:
Write artifacts, then notify manager pane exactly with delayed repeated Enter:
`tmux send-keys -t 0:2 'GPT STEP1_LIVE_NEST_THETA_SEMANTICS DONE - see proofs/v014/step1_live_nest_theta_semantics.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`
