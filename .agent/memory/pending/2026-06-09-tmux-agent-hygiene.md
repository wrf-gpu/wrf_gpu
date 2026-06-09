# Pending Memory: tmux Agent Hygiene

Status: pending promotion after the next multi-agent dispatch.

Lesson:

- Before launching new tmux agents, close completed/no-longer-needed worker
  windows from prior sprints so the shared tmux session remains readable.
- Do not close active workers, the manager pane, or principal-owned panes.

Evidence:

- Principal directive on 2026-06-09 during v0.14 grid-debug management.
- `.agent/skills/managing-sprints/SKILL.md`
