You are GPT-5.5 xhigh, independent verifier/debugger for wrf_gpu2 v0.14.

Work in a fresh tmux/codex worker from the main repo:

`/home/enric/src/wrf_gpu2`

Read and follow:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-gpt-stage3-wrapper-verifier/sprint-contract.md`

Endpoint:

Independently verify the just-finished Fable stage-3/wrapper-cadence fix for
the Switzerland/Gotthard h36->h37 dry-mass/PSFC blocker. Review the Fable diff,
proof report, and proof JSON. Decide whether the patch should be accepted,
rejected, locally fixed by GPT, or escalated to Fable high later. Do not call
Fable. Do not run long GPU gates. Use GPU only through the project lock if a
short focused gate is required and safe.

Write:

`/home/enric/src/wrf_gpu2/proofs/v014/gpt_stage3_wrapper_verifier.md`

End stdout exactly:

`GPT STAGE3_WRAPPER_VERIFIER DONE - see proofs/v014/gpt_stage3_wrapper_verifier.md`
