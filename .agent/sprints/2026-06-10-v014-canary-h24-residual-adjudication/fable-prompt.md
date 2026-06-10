You are Fable high, end-to-end adjudicator for wrf_gpu2 v0.14 Canary h24
field-residuals. This is not a micro-search task. Decide whether the current
h24 residuals block the v0.14 validation sequence, are bounded enough to let the
current 72h run continue and then launch Switzerland if h72 stays finite, or
require a code-fix task.

Read:
- `/home/enric/src/wrf_gpu2/PROJECT_CONSTITUTION.md`
- `/home/enric/src/wrf_gpu2/AGENTS.md`
- `/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-10-v014-canary-h24-residual-adjudication/sprint-contract.md`
- `/home/enric/src/wrf_gpu2/.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- `/home/enric/src/wrf_gpu2/.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/noahmp_nested_gpu_h4_validation.md`

Primary artifacts:
- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/canary_d02_h24_intermediate_grid_compare.json`
- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/canary_d02_h24_intermediate_grid_compare.md`
- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- CPU truth `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

Rules:
- No GPU.
- No source edits.
- CPU-only probes are okay: `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- Ignore `/home/enric/src/canairy_waves`.
- Keep the report compact and manager-actionable.

Write exactly:
`.agent/reviews/2026-06-10-v014-canary-h24-residual-adjudication-fable.md`

Include:
1. Verdict paragraph: `PROCEED_72H`, `PROCEED_BOUNDED_WITH_FOLLOWUP`,
   `BLOCK_72H_AFTER_CURRENT_RUN`, or `STOP_AND_FIX_NOW`.
2. Evidence table for `MUB/PB`, `QNRAIN`, `LH/HFX/PBLH`, `SWDOWN/SWNORM/SWDNB`,
   `PSFC/P/MU/PH`, and core fields `T/QVAPOR/U/V/U10/V10/T2/TSK`.
3. Explicit yes/no: can Switzerland GPU start if h72 Canary remains finite and
   no new worse signal appears?
4. Exact h72 checks the manager should inspect.
5. If fix needed, exact whole-task follow-up prompt outline.

Completion marker:
`FABLE CANARY_H24_RESIDUAL_ADJUDICATION DONE - see .agent/reviews/2026-06-10-v014-canary-h24-residual-adjudication-fable.md`
