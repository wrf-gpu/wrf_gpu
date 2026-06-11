You are GPT-5.5 xhigh, independent verifier for a Fable acoustic-substep candidate fix in wrf_gpu2.

Work in:

`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

Read and follow:

`.agent/sprints/2026-06-11-v014-gpt-acoustic-substep-verifier/sprint-contract.md`

Updated manager instruction: you have autonomy to do small, reversible, non-destructive work if it helps finish this. You may write verifier helpers, proof summaries, and proposed patch files. If a tiny source edit is clearly needed and Fable's running chain is finished, you may make it; otherwise do not overwrite Fable's active core edits, and record the change as a separate patch proposal for manager/Fable. Do not interact with Fable. Do not run ask-hermes/Telegram/human notifications. Do not touch `/home/enric/src/canairy_waves`.

Your job: independently test/review and, if safely possible, minimally advance the current uncommitted candidate diff from Fable. It appears to implement WRF constants (`cp=1004.5`, `g=9.81`), fresh stage omega (`ww`) threading, work-delta `u/v` feed into `advance_w` surface BC, WRF-location `w_damping`, and `diff_opt/km_opt` namelist threading. There are already running/finished 2h Switzerland gates under `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`.

Use existing logs first:

- `/tmp/acoustic_gate_forecast.log`
- `/tmp/acoustic_chain2.log`
- `proofs/v014/switzerland_acoustic_substep_blocker.json`
- `proofs/v014/switzerland_acoustic_substep_blocker.py`

If GPU work is needed, use `scripts/run_gpu_lowprio.sh` only and do not collide with the existing chain/lock.

Write your report to:

`.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`

Your verdict must be one of:

- `ACCEPT_FOR_MANAGER_GATE`
- `REJECT`
- `NEED_FABLE_AFTER_RESET`

Also state whether the fix materially collapses the h36->h37/h38 Switzerland residual, whether it has performance/architecture risks, and the exact commands/proofs used.

End stdout with exactly:

`GPT ACOUSTIC_SUBSTEP_VERIFIER DONE - see .agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`
