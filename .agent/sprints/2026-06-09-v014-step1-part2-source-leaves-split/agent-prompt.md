You are GPT-5.5 xhigh in tmux, assigned a proof-first v0.14 grid-parity debug
sprint for `/home/enric/src/wrf_gpu2`.

Read and obey:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-09-v014-step1-part2-source-leaves-split/sprint-contract.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Objective:

Split or fix the remaining Step-1 `T_TENDF` source-leaf divergence inside WRF
`first_rk_step_part2`.

Current accepted boundary:

- patched-init P/MU/W/PH frontiers are closed;
- first material field is full-domain `T_TENDF` at WRF
  `after_first_rk_step_part2`;
- `T_TENDF` vs current JAX dry source: max_abs `2457.5830078125`, RMSE
  `21.20870100357482`;
- source-save pre-addtend `T_TENDF` is also divergent;
- `rad_rk_tendf=1` does not move the boundary;
- boundary/spec/acoustic code is too late for the first failure.

Method:

Use the fastest rigorous CPU-only path. Emit or consume WRF truth inside
`first_rk_step_part2` after `calculate_phy_tend`, after `update_phy_ten`, and
after `conv_t_tendf_to_moist`. Include raw `RTH*TEN`, `T_HIST_SRC`, and adjacent
source/save leaves. Compare against the current JAX dry physics/source bundle
under the patched-init capture from `step1_tendency_contract_split.py`.

Do not only test the manager hypothesis. If it is wrong, rank likely
alternatives, run cheap falsifiers, and return the next exact boundary. If a
narrow source fix is obvious, implement it only with before/after proof.

Deliver:

- `proofs/v014/step1_part2_source_leaves_split.py`
- `proofs/v014/step1_part2_source_leaves_split.json`
- `proofs/v014/step1_part2_source_leaves_split.md`
- `.agent/reviews/2026-06-09-v014-step1-part2-source-leaves-split.md`
- if WRF patching is used:
  `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`

Rules:

- CPU-only; no GPU.
- No TOST or Switzerland.
- No FP32/memory source work.
- No Hermes/Telegram.
- Do not use Fable/Mythos.
- Keep final markdown/review concise.

Completion marker:

`GPT STEP1_PART2_SOURCE_LEAVES_SPLIT DONE - see proofs/v014/step1_part2_source_leaves_split.md`
