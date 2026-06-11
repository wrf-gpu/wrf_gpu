You are Fable xhigh continuing v0.14 Switzerland/Gotthard correctness work.

Work in your current worktree:

`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

First sync/rebase/merge as needed from the manager branch if you need the latest
merged state, but do not discard your work without evidence. The manager branch
has now merged your accepted subfixes and pushed:

- `3d0b439c`: real-case `hypsometric_opt=2` LOG-form HPG diagnostics.
- `79b0c22e`: real-case `rhs_ph`, edge-faithful specified-domain stage omega,
  and WRF dycore constants.
- `82f6b703`: merge into manager branch.
- `17c856c9`: roadmap/handoff update.

Read and follow the sprint contract:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-fable-stage3-wrapper-cadence/sprint-contract.md`

Whole endpoint:

Close the remaining h36->h37 dry-mass/PSFC blocker, now narrowed to
stage-3/end-of-step wrapper cadence plus residual lateral-band amplifier.
Build the smallest diagnostic proving where the remaining jump first appears,
compare against WRF-native or WRF-line-ported evidence, implement the smallest
WRF-faithful fix if found, and prove it with h36->h37 short gate plus focused
tests. If not closed, produce an exact no-fix handoff.

Do not start performance work. Do not use ask-hermes/Telegram. Do not touch
`/home/enric/src/canairy_waves`.

End stdout exactly:

`FABLE STAGE3_WRAPPER_CADENCE DONE - see .agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`
