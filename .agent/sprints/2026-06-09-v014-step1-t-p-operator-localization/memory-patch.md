# Memory Patch Proposal

## Scope

Project memory update for the v0.14 Step-1 T/P operator-localization sprint.

## Evidence

- `proofs/v014/step1_t_p_operator_localization.json` records verdict
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.
- The proof consumed 168 WRF substage truth files emitted by disposable,
  env-gated scratch WRF instrumentation under
  `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`.
- The first strict and first material T/P-family mismatch is `T_STATE` at
  `after_rk_addtend_before_small_step_prep`, RK1.
- The largest material residual at that boundary is `PH_TEND` max_abs
  `794096.1875`; `RW_TEND`, `PH_TENDF`, `T_TEND`, and `T_TENDF` are also large.
- RK1 `after_small_step_prep_calc_p_rho` work arrays match for `T_WORK` and
  `P_WORK` with max_abs `0.0`.
- Manager validation reran the CPU proof, JSON validation, Python compilation,
  and confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-t-p-operator-localization.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that v0.14 grid-parity debugging has localized the remaining Step-1
T/P-family mismatch before `small_step_prep`/acoustic. The next target is WRF
`first_rk_step_part1/part2` and JAX `_physics_step_forcing` / dry `*_tendf`
construction, not acoustic or final pressure refresh.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
