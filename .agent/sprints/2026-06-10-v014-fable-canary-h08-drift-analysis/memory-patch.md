# Memory Patch

Reviewer Status: ACCEPT_MEMORY_ANCHOR_ALREADY_WRITTEN.

This sprint is a correctness fix, not a memory optimization. It changes root
boundary cadence and terminal wrfbdy leaf construction. It does not introduce a
new resident memory contract, dynamic timestep-loop host transfers, broad array
materialization, or an FP32/memory claim.

Fable wrote a compact external memory anchor for the root cause in the project
Claude memory. The manager roadmap was also updated in
`.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md` and
`.agent/decisions/V0140-RELEASE-CHECKLIST.md`.

The memory/FP32 lane remains unchanged: mixed FP32 stays behind the fp64
field-parity frontier, and exact-branch memory preflight should be rerun before
the final release candidate if more source changes land.
