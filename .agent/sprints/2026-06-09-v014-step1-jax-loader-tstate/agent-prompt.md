You are GPT-5.5 xhigh, debugging worker for wrf_gpu2 v0.14 grid-parity.

Read, in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/sprints/2026-06-09-v014-step1-jax-loader-tstate/sprint-contract.md
5. proofs/v014/step1_pre_part1_handoff.md
6. proofs/v014/step1_pre_part1_handoff.json
7. proofs/v014/step1_pre_part1_handoff.py
8. proofs/v014/step1_live_nest_init_rerun.py
9. src/gpuwrf/integration/d02_replay.py
10. src/gpuwrf/nesting/boundary_construction.py
11. src/gpuwrf/runtime/operational_state.py

Objective:
Split the JAX live-nest Step-1 loader/carry construction for `T_STATE` against
the accepted WRF solve_em pre-`first_rk_step_part1` truth.

Previous accepted proof says:
`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`. WRF `T_STATE` is unchanged at
the solve_em call-site, WRF `T_STATE` maps to `State.theta - 300 K`, and WRF
pre-call `T_STATE` vs raw JAX live-nest state has max_abs
5.490173101425171, RMSE 1.9175184863907806.

Hard constraints:
- Verify `git log -1` and that commit `99df65e0` is an ancestor.
- CPU only: `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work.
- No Hermes/Telegram.
- Use scratch only under `/mnt/data/wrf_gpu2/v014_step1_jax_loader_tstate/**`.
- Production `src/gpuwrf/**` is read-only unless you prove one exact, narrow,
  performance-compatible loader bug. If you edit source, rerun the extra proof
  chain named in the contract.
- Do not touch unrelated untracked files.

Fastest rigorous method:
Do not rebuild WRF. Reuse the accepted WRF pre-call truth from
`/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`. Reconstruct the
JAX Step-1 input stages and compare:
`raw_child_state`, `live_child_state`, `boundary_packaged_state`,
`initial_carry_state`, and `haloed_step_entry_state`.

For `T_STATE`, include full-domain and interior-vs-boundary-band metrics. Also
report `State.theta` full-theta comparisons so the semantic check stays explicit.

Deliverables:
- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

Validation:
Run at minimum:
`python -m py_compile proofs/v014/step1_jax_loader_tstate.py`
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_loader_tstate.py`
`python -m json.tool proofs/v014/step1_jax_loader_tstate.json >/tmp/step1_jax_loader_tstate.validated.json`
`git diff -- src/gpuwrf`

Final output:
Write the proof/review artifacts, then notify the manager pane exactly with
delayed repeated Enter:
`tmux send-keys -t 0:2 'GPT STEP1_JAX_LOADER_TSTATE DONE - see proofs/v014/step1_jax_loader_tstate.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`
