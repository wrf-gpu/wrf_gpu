# AI operator routing

If you are Claude helping an END USER run a forecast with `wrf_gpu` (not
developing the port), stop and follow `AI_OPERATOR.md` or the
`.claude/skills/run-wrf-gpu` skill. The rest of this file is the port's internal
development protocol.

# Claude Project Instructions

@AGENTS.md

Claude-specific notes:

- The manager role should use the strongest Opus-class model available.
- Use subagents only for isolated work with frozen interfaces and clear file ownership.
- Keep context clean. Load sprint contracts and relevant skills before deep references.
- Use skills from `.agent/skills`; `.claude/skills` is only a mirror/index location.
- Do not auto-accept destructive actions.
- Do not update memory or skills without the patch and review protocol.
