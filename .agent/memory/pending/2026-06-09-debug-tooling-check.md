# Debug Tooling Check

Date: 2026-06-09

For complex v0.14 runtime/debug paths, the manager must periodically ask:
"Are we using the right tool?" Slow runtime reproduction can waste many sprints.

If a focused harness, savepoint emitter, comparator, schema freezer, or
visualization can turn repeated ad hoc debugging into a fast falsifiable proof
loop, spending one agent sprint on that tool is cheap and should be considered
before continuing another runtime-chasing ladder.

This rule is now recorded in `.agent/skills/managing-sprints/SKILL.md` and in
the reusable Opus management-review prompt.
