You are GPT-5.5 xhigh, debugging worker for wrf_gpu2 v0.14 grid-parity.

Read, in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/skills/building-wrf-oracles/SKILL.md
5. .agent/sprints/2026-06-09-v014-step1-part1-physics-state-mutation/sprint-contract.md

Objective:
Split the first material Step-1 T mismatch inside WRF `first_rk_step_part1`.
Previous proof `proofs/v014/step1_rk1_source_boundary.json` says:
`STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.
The WRF `after_first_rk_step_part1` `T_STATE` field differs from both JAX
operational carry and `_physics_step_forcing.state` by max_abs about 5.49 K.

Hard constraints:
- Verify `git log -1` and that commit `c18795af` is an ancestor.
- CPU only: use `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work.
- No Hermes/Telegram.
- Production `src/gpuwrf/**` is read-only unless you prove one exact, narrow,
  performance-compatible bug. If you make a production fix, rerun the extra
  gates in the contract.
- Use scratch only under `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/**`.
- Do not touch unrelated untracked files.

Fastest rigorous method:
Extend the existing Step-1 source-boundary truth/comparator. Instrument the
scratch WRF `module_first_rk_step_part1.F` at internal call boundaries such as
entry, after `init_zero_tendency`, `phy_prep`, radiation, surface, PBL,
cumulus, shallowcu, SCM, FDDA, and exit. Emit enough full d02 fields to identify
the first `T_STATE` mutation or prove input already diverged. Compare to the
matching JAX `_physics_step_forcing` / adapter / carry surfaces, not to unrelated
time boundaries.

Deliverables:
- `proofs/v014/step1_part1_physics_state_mutation.py`
- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation.md`
- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`
- optional `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`

Validation:
Run at minimum:
`python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py`
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part1_physics_state_mutation.py`
`python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json >/tmp/step1_part1_physics_state_mutation.validated.json`
`git diff -- src/gpuwrf`

Final output:
Write the proof/review artifacts, then notify the manager pane exactly with
delayed repeated Enter:
`tmux send-keys -t 0:2 'GPT STEP1_PART1_PHYSICS_STATE_MUTATION DONE - see proofs/v014/step1_part1_physics_state_mutation.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`
