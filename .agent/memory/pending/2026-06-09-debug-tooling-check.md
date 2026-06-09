# Debug Tooling Check

Date: 2026-06-09

For complex v0.14 runtime/debug paths, the manager must ask at every planning
step: "Are we using the right tool and method, and is this the fastest rigorous
wall-clock path?" Slow runtime reproduction can waste many sprints.

If a focused harness, savepoint emitter, comparator, schema freezer, or
visualization can turn repeated ad hoc debugging into a fast falsifiable proof
loop, spending one agent sprint on that tool is cheap. It can also be cheaper to
send one worker in parallel or serially to prove/refute a key hypothesis than to
continue step-by-step through slow runs.

For kernel/runtime-level debug, prefer expert methods that minimize steps and
false assumptions: isolate state boundaries, freeze schemas, create minimal
reproducers/savepoints, compare exact oracles, and parallelize independent
hypothesis tests without colliding on GPU or source ownership.

This rule is now recorded in `.agent/skills/managing-sprints/SKILL.md` and in
the reusable Opus management-review prompt.
