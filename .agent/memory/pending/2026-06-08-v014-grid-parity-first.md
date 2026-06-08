# Memory Patch Proposal

## Scope

Project-memory update for the 2026-06-08 manager goal shift: grid-cell
CPU-WRF-vs-GPU-WRF parity now precedes powered TOST, FP32, memory cleanup, and
scheme-long-tail work.

## Evidence

- The principal explicitly reprioritized the work: first find and fix why cells
  diverge across all values; second FP32; third remaining memory; fourth TOST
  only after cell fields are no longer radically divergent.
- Case 3 of powered TOST completed and the watcher stopped the run before Case 4.
- `proofs/v014/v10_grid_diagnostics.json` shows V10 grid RMSE above 1.5 m/s in
  3/3 durable powered-TOST cases, while station V10 is outside the tight ADR-029
  margin in only 1/3 cases. This proves station TOST alone is not sufficient as
  the next decision point.
- `docs/KNOWN_ISSUES.md` and `docs/equivalence-demo.md` already identify KI-9 as
  lead-time wind/mass divergence, but not with a closed operator root cause.

## Proposed Destination

`.agent/memory/stable/approved-patterns.md` and/or the manager runbook after
independent review approves the wording.

## Patch

Proposed addition:

- When direct CPU-WRF-vs-GPU-WRF grid-cell diagnostics show broad field
  divergence, station-observation TOST becomes a final gate rather than the next
  GPU campaign. First localize and either fix or explicitly root-cause the
  cell-level field mismatch across all comparable wrfout fields; only then spend
  long GPU time on powered TOST.

Proposed addition:

- Keep a compact manager handoff decision file for any priority inversion that
  changes the current release gate. It must record the priority order, current
  proof objects, running agents, stopped/running GPU jobs, and the next manager
  actions so auto-compaction or manager replacement cannot revert to stale
  roadmap wording.

## Reviewer Status

Reviewer Status: pending. Do not apply to stable memory until the sidecar
grid-envelope and prior-attribution reports are reviewed.
