You are GPT-5.5 xhigh, independent verifier/debugger for wrf_gpu2 v0.14.

Work in a fresh tmux/codex worker from the main repo:

`/home/enric/src/wrf_gpu2`

Read and follow:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-gpt-stage3-wrapper-verifier/sprint-contract.md`

Fable xhigh has completed commit:

`a5f282521090c4b1e3d1d4618295db09d49cdc17`

Fable worktree:

`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

Endpoint:

Independently verify the just-finished Fable stage-3/wrapper-cadence result for
the Switzerland/Gotthard h36->h37 dry-mass/PSFC blocker. Review the Fable diff,
proof report, and proof JSON from the Fable worktree. Fable claims the boundary
cadence/advection changes are WRF-faithful and useful, but they do NOT close the
venting blocker; they instead falsify the boundary-band lane and point to an
interior acoustic `advance_w` / `phi` sink.

Decide whether the Fable patch should be accepted as a boundary
WRF-faithfulness fix, rejected, locally fixed by GPT, or split/parked. Also
produce the direct next interior `advance_w`/`phi` discriminator artifact or
exact plan if the Fable conclusion survives review. Do not call Fable. Do not
run long GPU gates. Use GPU only through the project lock if a short focused
gate is required and safe.

Write:

`/home/enric/src/wrf_gpu2/proofs/v014/gpt_stage3_wrapper_verifier.md`

End stdout exactly:

`GPT STAGE3_WRAPPER_VERIFIER DONE - see proofs/v014/gpt_stage3_wrapper_verifier.md`
