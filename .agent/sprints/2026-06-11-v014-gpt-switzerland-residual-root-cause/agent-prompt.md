You are GPT-5.5 xhigh, independent debugger for wrf_gpu2 v0.14.

Work in a dedicated worker git worktree:

`/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-switzerland-residual-root-cause`

Read and follow:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-gpt-switzerland-residual-root-cause/sprint-contract.md`

Important framing:

The manager has a current leading candidate around an interior acoustic
`phi/p` mechanism, possibly near `advance_w`, `rw_tend`/`ph_tend`, or pressure
refresh. Do not treat that as truth. Treat it as one candidate from the current
evidence. Build your own ranked hypothesis ledger from the reports, source, WRF
anchors, and proof artifacts, then follow the fastest rigorous proof path. If
the evidence points elsewhere, reject the manager hypothesis and pursue the
better root.

Fable and the manager currently think the boundary lane is no longer the main
driver and that an interior hydrostatic `phi/p` mechanism is the best clue. Use
that as context, but you own the diagnosis. Solve it end to end if you can.
Return to the manager only when the contract stop criteria are met: fixed,
specific local fix proposed, stronger no-fix narrowing, prior hypothesis
rejected with better evidence, or method limit requiring new dumps/long GPU/
scarce model escalation.

Endpoint:

Find and fix the remaining Switzerland/Gotthard h36->h37 dry-mass/PSFC blocker
if a local WRF-faithful fix is provable; otherwise produce a stronger exact
narrowing and next proof loop. You may implement minimal source fixes on your
worker branch if the evidence is strong. Do not run long 72h GPU gates. Do not
call Fable/Mythos. Do not use Hermes/Telegram.

Write:

`/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`

and any proof artifacts under:

`/home/enric/src/wrf_gpu2/proofs/v014/`

End stdout exactly:

`GPT SWITZERLAND_RESIDUAL_ROOT_CAUSE DONE - see .agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`
