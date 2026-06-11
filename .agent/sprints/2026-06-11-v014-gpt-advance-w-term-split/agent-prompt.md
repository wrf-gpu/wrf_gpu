You are GPT-5.5 xhigh, independent debugger for wrf_gpu2 v0.14.

Work in a dedicated worker git worktree:

`/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-advance-w-term-split`

Read and follow:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-gpt-advance-w-term-split/sprint-contract.md`

Important framing:

The current evidence says the remaining Switzerland/Gotthard h36->h37
dry-mass / PSFC blocker first appears in the single RK1 acoustic substep from
WRF call `21601` to `21602`. Prior sprints have ruled out boundary cadence,
specified stage omega, real-case `rhs_ph`, final wrapper pressure refresh, and
the known lower-boundary surface-`w` feed deviation. The manager's leading
boundary is now `advance_mu_t` outputs consumed by `advance_w` versus internals
of `advance_w_wrf()`.

Do not treat that as truth. Treat it as the current evidence boundary. Build
your own hypothesis ledger, test it, and solve the assigned blocker end to end
if possible. If the evidence points elsewhere, reject the manager boundary and
prove the better one.

Endpoint:

Find and fix the remaining first `phi/p` creator if a local WRF-faithful fix is
provable; otherwise return a stronger exact term/state-boundary narrowing and
next proof loop. You may implement minimal source fixes on your worker branch if
the evidence is strong. Do not run long 72h GPU gates. Do not call Fable/Mythos.
Do not use Hermes/Telegram.

Write:

`/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md`

and proof artifacts under:

`/home/enric/src/wrf_gpu2/proofs/v014/`

End stdout exactly:

`GPT ADVANCE_W_TERM_SPLIT DONE - see .agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md`
