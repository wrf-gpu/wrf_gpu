# Changelog

## 0.2.4 (2026-06-09, principal-directed)

- Added the Mythos heavy-problem lane: send extremely hard v0.14 problems to
  Mythos in tmux `0:1` as whole endpoint-defined assignments while the manager
  retains contracts, locks, review, gates, and merge control.
- Added the Mythos context-refresh ritual: before each new Mythos sprint after a
  completion or context-risk point, send `/compact`, wait about two minutes, then
  send the full assignment with Enter.

## 0.2.3 (2026-06-09, principal-directed)

- Added long-roadmap drift prevention: after every 15 closed active-milestone
  sprints, dispatch an Opus 4.8 xhigh management review to critique roadmap
  direction, proof chain, sprint sizing, parallelization, and next steps.
- Added a compact reusable Opus management-review prompt and output contract.
- Added the v0.14 goal-change gate: the milestone goal may change only when an
  Opus 4.8 xhigh management review explicitly agrees that the current goal is
  technically impossible or no longer the smartest useful target under the
  latest evidence.

## 0.2.2 (2026-06-09, principal-confirmed)

- Tightened the cross-model debug cadence: after two focused GPT/debug sprints
  still leave a complex correctness fix unproven, escalate to one Opus xhigh
  critic/debugger to review method, hypotheses, evidence, performance impact,
  and the candidate bug before the manager draws the next conclusion.

## 0.2.1 (2026-06-09, principal-authorized)

- Added the cross-model debug cadence: after two inconclusive GPT/debug sprints
  on the same complex correctness bug, dispatch one Opus xhigh critic/debugger
  before committing to the next conclusion.
- Added the tmux delayed repeated-Enter completion pattern for worker done
  markers into the manager pane.
- Added tmux session hygiene before dispatch: close completed/no-longer-needed
  worker windows before launching new agents.

## 0.2.0 (2026-06-01, principal-authorized)

- Added **Agent dispatch mechanics** section: effort-tiers (Opus-max core / Opus-xhigh+GPT-xhigh debug-write-review / parallel), one-GPU-job-at-a-time, Opus in-process workers.
- **GPT-5.5 launched as an INTERACTIVE codex TUI in tmux** (not headless `codex exec`) so the principal can attach + interject (principal directive 2026-06-01).
- Captured the hard-won liveness rule (transcript-lag ≠ death; verify by PID) and the hibernation/CUDA-context + detached-run + commit-per-proof discipline.

## 0.1.0

- Initial sprint management workflow.
