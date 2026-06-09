You are GPT-5.5 xhigh in tmux, assigned a proof-first v0.14 grid-parity debug
sprint for `/home/enric/src/wrf_gpu2`.

Read and obey:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-09-v014-step1-tendency-contract-split/sprint-contract.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Objective:

Localize or fix the remaining Step-1 tendency-family divergence. The stale RK1
`P_STATE` issue is closed under patched-init capture. The next boundary is WRF
`first_rk_step_part2` `T_TENDF`, then RK1 `T_TEND/PH_TEND/RW_TEND`, against JAX
`compute_advection_tendencies` and `_augment_large_step_tendencies`.

Method:

Do not only test the manager's hypothesis. If it is wrong, rank likely
alternatives, run cheap falsifiers, and return the best next exact boundary.

Deliver:

- `proofs/v014/step1_tendency_contract_split.py`
- `proofs/v014/step1_tendency_contract_split.json`
- `proofs/v014/step1_tendency_contract_split.md`
- `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`

Rules:

- CPU-only.
- No GPU.
- No TOST/Switzerland.
- No memory/FP32 source work; Mythos owns that in tmux `0:1`.
- No Hermes/Telegram.
- Production source edit only if exact, narrow, and proven by before/after proof.

Completion marker:

`GPT STEP1_TENDENCY_CONTRACT_SPLIT DONE - see proofs/v014/step1_tendency_contract_split.md`
