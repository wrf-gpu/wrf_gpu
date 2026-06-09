# Changelog

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
