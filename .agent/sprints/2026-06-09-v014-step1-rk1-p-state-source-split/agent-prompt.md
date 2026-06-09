You are GPT-5.5 xhigh in tmux, assigned a proof-first v0.14 grid-parity debug
sprint for `/home/enric/src/wrf_gpu2`.

Read and obey:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-09-v014-step1-rk1-p-state-source-split/sprint-contract.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Objective:

Localize or fix the post-Mythos Step-1 RK1 stage-entry material `P_STATE`
divergence. The fresh comparator now says the first material T/P-family
mismatch is `P_STATE` at
`after_rk_addtend_before_small_step_prep`, RK1, with huge `PH_TEND/RW_TEND`
family residuals. Do not jump into acoustic substeps until this earlier
boundary is closed.

Important method:

Do not only test the manager's current hypothesis. If it is wrong, rank likely
alternatives, run cheap falsifiers, and return the best next exact boundary.

Deliver:

- `proofs/v014/step1_rk1_p_state_source_split.py`
- `proofs/v014/step1_rk1_p_state_source_split.json`
- `proofs/v014/step1_rk1_p_state_source_split.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

Rules:

- CPU-only.
- No GPU.
- No TOST/Switzerland.
- No memory/FP32 source work; Mythos owns that in tmux `0:1`.
- No Hermes/Telegram.
- Production source edit only if exact, narrow, and proven by before/after proof.

Completion marker:

`GPT STEP1_RK1_P_STATE_SOURCE_SPLIT DONE - see proofs/v014/step1_rk1_p_state_source_split.md`
