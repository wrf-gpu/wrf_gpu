You are GPT-5.5 xhigh in tmux, assigned the v0.14 Step-1 source-fidelity closure sprint for `/home/enric/src/wrf_gpu2`.

Read and obey:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-10-v014-step1-source-fidelity-closure/sprint-contract.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/step1_part2_source_leaves_split.md`
- `proofs/v014/step1_dry_source_leaf_fix.md`

Objective:

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining Step-1 `T_TENDF` source-fidelity gap.

Current accepted facts:

- WRF `update_phy_ten` closes exactly as `T_TENDF = pre + active RTH`.
- WRF `conv_t_tendf_to_moist` closes to roundoff and equals `after_first_rk_step_part2`.
- Patched JAX dry `T_TENDF` is active but too small: max_abs `260.83156991819124`.
- WRF active `RTHBLTEN` max_abs is `2522.90576171875`.
- Final WRF after-conv vs patched JAX dry residual remains max_abs `2457.575215120763`, RMSE `21.445918959761645`.
- Forcing radiation only moves max_abs to `2454.161554535577`, so held `RTHRATEN` is secondary.
- WRF `conv_t_tendf_to_moist` contributes max_abs `224.50967407226562`.

Method:

Use the fastest rigorous CPU-only path and think like a senior debugger, not a narrow instruction follower. You may fix all three blockers if the evidence remains local:

1. Split MYNN PBL adapter/kernel inputs and outputs against WRF `RTHBLTEN/RQVBLTEN`; explain and fix why the JAX source is about 10x too weak.
2. Seed/refresh held `RTHRATEN` at Step 1 if necessary, but rank it correctly as secondary unless new proof says otherwise.
3. Implement WRF `conv_t_tendf_to_moist` / `QV_TEND` before `DryPhysicsTendencies.t_tendf`.
4. Rerun the strict Step-1 proof. If it does not close, return exactly one narrower WRF-anchored blocker and the fastest next proof/fix route.

Rules:

- CPU-only. No GPU, no TOST, no Switzerland.
- No Hermes/Telegram.
- Do not use Fable/Mythos.
- Keep output compact.
- No production CPU-WRF dependency, no timestep-loop host/device transfer, no broad dycore rewrite.
- Preserve `rad_rk_tendf=0` behavior.
- If you edit production source, keep the diff narrow and performance-compatible.

Deliver:

- `proofs/v014/step1_source_fidelity_closure.py`
- `proofs/v014/step1_source_fidelity_closure.json`
- `proofs/v014/step1_source_fidelity_closure.md`
- `.agent/reviews/2026-06-10-v014-step1-source-fidelity-closure.md`
- updated reused proof artifacts if needed
- focused tests if production code changes

Completion marker:

`GPT SOURCE_FIDELITY_CLOSURE DONE - see proofs/v014/step1_source_fidelity_closure.md`
