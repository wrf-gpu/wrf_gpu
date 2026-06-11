# Sprint Contract: v0.14 GPT Stage-3 / Specified-LBC Verifier

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh, tmux
Status: PREPARED; dispatch only after the active Fable xhigh
`stage3-wrapper-cadence` sprint prints its DONE marker or otherwise stops.

## Objective

Independently verify the Fable stage-3/wrapper-cadence fix for the remaining
Switzerland/Gotthard h36->h37 dry-mass/PSFC blocker.

This is a scarcity-control follow-up: do **not** send another Fable/Mythos xhigh
run reflexively. GPT must first review the Fable diff, reproduce or inspect the
proofs, and decide whether the patch is acceptable, locally fixable, or still
blocked.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/sprints/2026-06-11-v014-fable-stage3-wrapper-cadence/sprint-contract.md`
- Fable output report:
  `.agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`
- Fable proof JSON:
  `proofs/v014/switzerland_stage3_wrapper_cadence.json`
- Existing acoustic proofs:
  `proofs/v014/switzerland_acoustic_continuation.json`
  `proofs/v014/switzerland_acoustic_substep_blocker.json`

## Required Work

1. Inspect the Fable diff and classify each source change:
   WRF-faithful required change, proof-only support, optional speed/cleanup, or
   risky/unproven.
2. Verify that the change implements the intended WRF-specified boundary cadence
   rather than just tuning the failing case:
   - relax-bdy dry tendency staging,
   - stage/end-of-step specified boundary application,
   - per-substep specified LBC handling,
   - tangential/normal wind work targets,
   - interaction with non-specified/idealized domains.
3. Re-run or validate the focused proof artifacts if feasible without stealing a
   long GPU slot. Prefer CPU/static checks and existing JSON first. Use GPU only
   through the project lock if the proof contract requires it.
4. Judge the h36->h37 gate:
   - accepted if residual/excess outflux materially collapses without clamps,
     masks, host/device loop transfers, or tolerance relaxation;
   - rejected if the gate worsens, only shifts error, or lacks WRF evidence;
   - bounded if the residual is small enough and independently justified.
5. If the diff is close but has a small local defect, GPT may implement a
   minimal non-destructive fix on its own worker branch. Do not rewrite the
   entire Fable patch.

## Output

Write:

`proofs/v014/gpt_stage3_wrapper_verifier.md`

Include:

- one-sentence verdict: `ACCEPT`, `REJECT`, `LOCAL_FIX_PROPOSED`, or
  `NEED_FABLE_HIGH_AFTER_GPT`;
- compact table of Fable claims vs evidence;
- source-risk table by file/function;
- gate-result comparison against old/hypso/rhs_ph baselines;
- exact commands run and artifacts checked;
- next action for the manager, max 8 bullets.

Print exactly:

`GPT STAGE3_WRAPPER_VERIFIER DONE - see proofs/v014/gpt_stage3_wrapper_verifier.md`

## Constraints

- Do not use `ask-hermes`, Telegram, or human notification commands.
- Do not touch `/home/enric/src/canairy_waves`.
- Do not dispatch Fable/Mythos.
- Do not start long 72h GPU gates.
- Keep output context-sparing.
