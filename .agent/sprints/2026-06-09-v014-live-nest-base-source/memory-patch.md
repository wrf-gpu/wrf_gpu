# Memory Patch Proposal

## Scope

Project memory update for the v0.14 live-nest base source sprint and tmux worker
notification reliability.

## Evidence

- `proofs/v014/live_nest_base_source_fix.json` verdict is
  `LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`.
- Native live-nest base fields now match CPU-WRF h0 as validation oracle to
  formula-level residuals, with no CPU-WRF h0 production input.
- `.agent/reviews/2026-06-09-v014-debug-method-critic.md` requires an
  init-override or direct grid-field proof before this base fix may be treated
  as V10/grid closure.
- The principal observed that single-Enter tmux notifications can remain staged
  and require a manual Enter. Future tmux worker prompts should request delayed
  repeated Enter presses after the done marker.
- The principal accepted the Opus critique as useful and directed the manager
  cadence: after two inconclusive GPT/debug sprints on one complex problem,
  dispatch one Opus xhigh critic/debugger to challenge method, conclusions, and
  hypotheses before committing to the next conclusion.

## Proposed Destination

Create:

- `.agent/memory/pending/2026-06-09-v014-live-nest-base-source.md`
- `.agent/memory/pending/2026-06-09-cross-model-debug-cadence.md`

Future stable skill update candidate for `.agent/skills/managing-sprints`:
document worker completion as:
`tmux send-keys -t 0:2 '<DONE MARKER>' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`.
Also document the cross-model debug cadence described above.

## Patch

Record that live-nest base source parity is a scoped fix, not grid-parity
closure. Record that tmux worker notifications need delayed multi-Enter in this
environment. Record the two-GPT-then-Opus escalation cadence for complex
correctness bugs.

## Reviewer Status:

Pending. Promote after the next direct grid-field/falsifier sprint confirms the
claim separation and the tmux notification pattern works without manual Enter.
