# Sprint Contract: v0.14 GPT Stage-3 / Specified-LBC Verifier

Date: 2026-06-11
Owner: manager
Assignee: GPT-5.5 xhigh, tmux
Status: READY; Fable xhigh `stage3-wrapper-cadence` is complete at
`a5f282521090c4b1e3d1d4618295db09d49cdc17` in
`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`.

## Objective

Independently verify the Fable stage-3/wrapper-cadence result for the remaining
Switzerland/Gotthard h36->h37 dry-mass/PSFC blocker, then continue the residual
debug lane only if the next proof loop is clear and short.

This is a scarcity-control follow-up: do **not** send another Fable/Mythos xhigh
run reflexively. GPT must first review the Fable diff, reproduce or inspect the
proofs, and decide whether the patch is acceptable as a WRF-faithfulness
boundary fix, should be rejected, should be split/parked, or needs a small local
correction. Fable's own report says the boundary lane is **not** the final
venting fix; verify that conclusion rather than assuming the blocker is closed.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `.agent/sprints/2026-06-11-v014-fable-stage3-wrapper-cadence/sprint-contract.md`
- Fable output report:
  `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/.agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`
- Fable proof JSON:
  `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_stage3_wrapper_cadence.json`
- Existing acoustic proofs:
  `proofs/v014/switzerland_acoustic_continuation.json`
  `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_acoustic_substep_blocker.json`

Fable verdict to verify:

- WRF specified-domain boundary cadence and specified advection-order
  degradation materially improve the boundary band against WRF native dumps
  (often 5-20x).
- Those changes are flag-gated/default-off and therefore do not affect current
  production runtime unless explicitly enabled.
- The h36->h37 hourly gate does **not** close; excess outflux worsens as the
  boundary becomes more WRF-faithful, implying the old boundary error was partly
  compensating an interior mass sink.
- The next likely root is an interior hydrostatic `phi`-sink / pressure-rise
  pair produced in the acoustic `w/phi` machinery, with ranked suspects:
  `advance_w` phi-update terms, `rw_tend`/`ph_tend` consumption inside the
  implicit solve, then post-stage `calc_p_rho` / grid-p refresh.

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
4. Judge the h36->h37 gate exactly:
   - `BOUNDARY_FIX_ACCEPT_BUT_BLOCKER_OPEN` if the band fix is WRF-proven but
     the venting gate remains open;
   - `REJECT` if the band proof or implementation is not WRF-faithful;
   - `LOCAL_FIX_PROPOSED` only if a small correction is safe and evidenced;
   - do not call the blocker closed unless residual and excess outflux
     materially collapse without clamps, masks, host/device loop transfers, or
     tolerance relaxation.
5. If the diff is close but has a small local defect, GPT may implement a
   minimal non-destructive fix on its own worker branch. Do not rewrite the
   entire Fable patch.
6. If the Fable conclusion survives review and no small band defect is found,
   produce the direct next debug artifact for the interior lane:
   - either a concrete `advance_w` / `phi` term-discriminator plan with exact
     source functions, dump points, commands, and acceptance thresholds;
   - or, if feasible within a short CPU/static proof loop, implement a focused
     proof script under `proofs/v014/` that compares the pre/post-`advance_w`
     `phi` work-array terms for stage 1 from the h36 bit-identical input state.
   Prefer WRF-native/line-ported evidence. Avoid JAX-vs-JAX self-acceptance.

## Output

Write:

`proofs/v014/gpt_stage3_wrapper_verifier.md`

Include:

- one-sentence verdict: `BOUNDARY_FIX_ACCEPT_BUT_BLOCKER_OPEN`, `REJECT`,
  `LOCAL_FIX_PROPOSED`, or `NEED_FABLE_HIGH_AFTER_GPT`;
- compact table of Fable claims vs evidence;
- source-risk table by file/function;
- gate-result comparison against old/hypso/rhs_ph baselines;
- explicit statement on runtime risk: default-off cost, expected enabled cost,
  and any hot-path concerns;
- next interior `advance_w` / `phi` discriminator artifact or exact plan;
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
