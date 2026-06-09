# Sprint Contract: V0.14 Dynamic Root-Cause Opus Critic

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

After two focused GPT debug sprints failed to prove closure on the same
v0.14 grid-divergence problem, run one Opus xhigh critic/debugger to challenge
the evidence chain, root-cause hypotheses, and next-step plan.

The critic should decide what the next highest-yield localization or fix sprint
should be, without editing production code.

## Trigger Evidence

- `proofs/v014/same_state_momentum_mass.json`:
  `JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`
- `proofs/v014/grid_after_live_nest_base.json`:
  `GRID_SYMPTOM_NOT_CLOSED`
- `proofs/v014/live_nest_base_source_fix.json`:
  `LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`
- `.agent/reviews/2026-06-09-v014-debug-method-critic.md`
- `.agent/skills/managing-sprints/SKILL.md` cross-model debug cadence

## Non-Goals

- No production `src/` edits.
- No GPU.
- No TOST.
- No Switzerland validation.
- No FP32 implementation.
- No memory optimization implementation.
- No Hermes or Telegram.

## Write Scope

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- Optional compact machine-readable summary:
  `proofs/v014/dynamic_root_cause_opus_critic.json`

## Required Work

1. Read the trigger evidence above and the relevant compact proof Markdown.
2. Inspect source code only as needed to evaluate hypotheses, especially:
   - `src/gpuwrf/dynamics/`
   - `src/gpuwrf/runtime/operational_mode.py`
   - relevant integration/carry code touched by the live-nest base-source fix
3. Challenge the current manager conclusion:
   - Is final RK pressure-gradient/mass-wind/theta-pressure coupling really the
     next best target?
   - Could the evidence still indicate stale carry/init, boundary exchange,
     stagger mapping, writer, or reference-state issues?
   - Are there hidden performance risks in likely fixes?
4. Produce a concise ranked hypothesis table with:
   - hypothesis
   - evidence for/against
   - cheapest falsifier
   - exact files/functions to inspect or instrument next
   - expected proof object
5. Recommend exactly one next sprint as the highest-yield action and one backup
   if it fails.
6. State whether any Opus-discovered issue justifies production code edits now
   or whether a proof sprint must precede edits.

## Validation

The deliverable must be a self-contained Markdown handoff with:

- objective
- files changed
- commands run
- proof objects produced
- unresolved risks
- next decision needed

If JSON is written, it must validate with:

```bash
python -m json.tool proofs/v014/dynamic_root_cause_opus_critic.json >/tmp/dynamic_root_cause_opus_critic.validated.json
```

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'OPUS DYNAMIC_ROOT_CAUSE_CRITIC DONE - see .agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
