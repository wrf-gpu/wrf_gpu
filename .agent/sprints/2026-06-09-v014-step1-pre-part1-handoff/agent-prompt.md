You are GPT-5.5 xhigh, debugging worker for wrf_gpu2 v0.14 grid-parity.

Read, in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/skills/building-wrf-oracles/SKILL.md
5. .agent/sprints/2026-06-09-v014-step1-pre-part1-handoff/sprint-contract.md

Objective:
Move one boundary upstream from WRF `first_rk_step_part1` and classify why
`T_STATE` already diverges at part1 entry. Previous proof
`proofs/v014/step1_part1_physics_state_mutation.json` says:
`STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`; WRF part1 entry vs JAX live-nest
step-entry has max_abs 5.490173101425171, while WRF internal part1 T_STATE delta
from entry is exactly 0.0.

Hard constraints:
- Verify `git log -1` and that commit `588686d6` is an ancestor.
- CPU only: use `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work.
- No Hermes/Telegram.
- Production `src/gpuwrf/**` is read-only unless you prove one exact, narrow,
  performance-compatible bug. If you make a production fix, rerun the extra
  gates in the contract.
- Use scratch only under `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/**`.
- Do not touch unrelated untracked files.

Fastest rigorous method:
Extend the existing Step-1 savepoint/comparator. Emit WRF solve_em call-site
surfaces immediately before `CALL first_rk_step_part1` and enough upstream
surfaces to determine whether WRF changed `grid%t_2` before the call. Compare
to matching JAX live-nest loader/carry/state surfaces. Explicitly verify
full-theta vs perturbation-theta semantics; do not treat that as implicit.

Deliverables:
- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`
- optional `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`

Validation:
Run at minimum:
`python -m py_compile proofs/v014/step1_pre_part1_handoff.py`
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_pre_part1_handoff.py`
`python -m json.tool proofs/v014/step1_pre_part1_handoff.json >/tmp/step1_pre_part1_handoff.validated.json`
`git diff -- src/gpuwrf`

Final output:
Write the proof/review artifacts, then notify the manager pane exactly with
delayed repeated Enter:
`tmux send-keys -t 0:2 'GPT STEP1_PRE_PART1_HANDOFF DONE - see proofs/v014/step1_pre_part1_handoff.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`
