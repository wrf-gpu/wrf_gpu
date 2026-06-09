You are GPT-5.5 xhigh in tmux, assigned the v0.14 dry source-leaf implementation sprint for `/home/enric/src/wrf_gpu2`.

Read and obey:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-10-v014-dry-source-leaf-fix/sprint-contract.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/step1_part2_source_leaves_split.md`

Objective:

Implement or conclusively block true WRF dry physics source leaves for active `RTHRATEN` and `RTHBLTEN` before `_augment_large_step_tendencies`, then prove the Step-1 `T_TENDF` residual collapses.

Current accepted proof:

- WRF `update_phy_ten`: `T_TENDF == pre + active RTH`, nested-interior max_abs `0.0`.
- WRF `conv_t_tendf_to_moist`: closes to roundoff, max_abs `0.00016236981809925055`.
- Current patched-init JAX dry `T_TENDF`: max_abs `2457.5830078125`, RMSE `21.674279301376934`.
- Active raw leaves: `RTHRATEN`, `RTHBLTEN`; dominant active raw leaf: `RTHBLTEN`.
- Aggregate post-physics state delta is falsified as a narrow substitute.

Method:

Use the fastest rigorous CPU-only path. Start with `src/gpuwrf/runtime/operational_mode.py`, `DryPhysicsTendencies`, the radiation held `rthraten`, and the active PBL adapter(s). If a direct implementation works, land it with focused tests and proof. If it does not, do not stop at "failed"; rank the exact blocker(s), identify the next source boundary, and produce a concise proof-backed handoff.

Rules:

- CPU-only. No GPU, no TOST, no Switzerland.
- No Hermes/Telegram.
- Do not use Fable/Mythos.
- Keep output compact.
- No production CPU-WRF dependency, no timestep-loop host/device transfer, no broad dycore rewrite.
- If you edit production source, keep the diff narrow and performance-compatible.

Deliver:

- `proofs/v014/step1_dry_source_leaf_fix.py`
- `proofs/v014/step1_dry_source_leaf_fix.json`
- `proofs/v014/step1_dry_source_leaf_fix.md`
- `.agent/reviews/2026-06-10-v014-dry-source-leaf-fix.md`
- updated `proofs/v014/step1_part2_source_leaves_split.{py,json,md}` if reused post-fix
- focused tests if production code changes

Completion marker:

`GPT DRY_SOURCE_LEAF_FIX DONE - see proofs/v014/step1_dry_source_leaf_fix.md`
