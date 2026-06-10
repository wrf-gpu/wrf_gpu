# Sprint Contract: V0.14 Canary h24 Residual Adjudication

Date: 2026-06-10
Owner: manager
Assignee: Fable high, fresh tmux window
Status: DISPATCH

## Objective

Adjudicate the current Canary L2 d02 72h GPU run at h24. Determine whether the
h24 field-comparison residuals are a v0.14 release blocker, a known/bounded
class to carry into the final 72h Atlas, or a local fixable bug. If a safe local
fix is obvious, describe the exact patch and proof gate; do not interrupt the
running GPU validation.

This is an end-to-end decision task, not a micro-hypothesis prompt.

## Inputs

- Current run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
- h24 compare:
  `canary_d02_h24_intermediate_grid_compare.{json,md,log}`
- h4 accepted land gate:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_noahmp_lu16fix_h4_20260610T212056Z`
- frozen-land baseline:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_moistcqw_20260610T171818Z`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- Current docs:
  `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
  `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
  `proofs/v014/noahmp_nested_gpu_h4_validation.md`
  `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`

## Known h24 facts

- GPU forecast is still running and has passed h24 output generation.
- h24 compare command returned `rc=0`, but comparator verdict is `FAIL`.
- Hard tolerance failures are currently only static/base-state fields `MUB` and
  `PB`, a previously known class.
- Dynamic residuals needing adjudication include:
  `QNRAIN` rare max outlier (`p99_abs=0`, max large), `LH`, `HFX`, `PBLH`,
  `SWDOWN/SWNORM/SWDNB`, and pressure drift summaries.
- Core weather fields with manifest tolerances such as `U10`, `V10`, `T2`,
  `T`, `QVAPOR`, and `PSFC` appear finite and mostly within candidate
  tolerance classes, but verify this from JSON.

## Constraints

- Do not use GPU.
- Do not modify source code or running validation roots.
- You may run CPU-only analysis scripts/probes with
  `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- Keep output compact and manager-actionable.
- If you find a code bug, provide exact file/function/patch plan and proof
  gate; do not apply it in this sprint unless it is documentation-only.
- Ignore all `/home/enric/src/canairy_waves` artifacts; different project.

## Required Output

Write:

`.agent/reviews/2026-06-10-v014-canary-h24-residual-adjudication-fable.md`

Required structure:

1. Verdict paragraph: `PROCEED_72H`, `PROCEED_BOUNDED_WITH_FOLLOWUP`,
   `BLOCK_72H_AFTER_CURRENT_RUN`, or `STOP_AND_FIX_NOW`.
2. Evidence table: field/class, h24 signal, likely cause class, release impact,
   recommended action.
3. Explicit yes/no: can Switzerland GPU start if h72 Canary remains finite and
   no new worse signal appears?
4. Exact next proof: what the manager should inspect at h72.
5. If fix needed: exact whole-task Fable/Mythos follow-up prompt outline.

Completion marker:

`FABLE CANARY_H24_RESIDUAL_ADJUDICATION DONE - see .agent/reviews/2026-06-10-v014-canary-h24-residual-adjudication-fable.md`
