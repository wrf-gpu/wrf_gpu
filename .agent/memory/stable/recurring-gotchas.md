# Recurring Gotchas

- Isolated GPU kernel speedups do not imply end-to-end forecast speedup.
- Bitwise equality is a debug tool, not the default validation target.
- Hidden host/device transfers can erase all kernel gains.
- Agent-written memory can drift unless reviewed like code.
- Backend preference from research notes is evidence, not a decision.
