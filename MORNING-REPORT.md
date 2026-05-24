# Morning Report — 2026-05-24

**Status:** M6 is active, not closed. The project has moved past the old warm-bubble amplitude target and is now executing the critic-ratified HYBRID close path: diagnose real replay, clean up sourced operator terms, then run Tier-3 and Tier-4 gates. ADR-023 and ADR-024 remain **PROPOSED**.

## Milestone Ledger

| Milestone | Status | Current read |
|---|---|---|
| M0 | Closed | AgentOS/bootstrap complete. |
| M1 | Closed | WRF oracle and fixture foundation complete. |
| M2 | Closed | Backend decision path completed earlier; current implementation path is JAX/XLA. |
| M3 | Closed | GPU state/grid skeleton and transfer discipline established. |
| M4 | Closed | Minimal dycore work completed enough to feed M6.x. |
| M5 | Closed | First physics-suite work completed enough to feed coupled M6. |
| M6 | Active | Dycore stabilization and coupled short-forecast validation. |
| M7 | Prologue done | S0a operational/data prologue complete; implementation waits on M6 close. |
| M8 | Pending | Public/forkable release not started. |

## M6 Dissection

Done:
- S1 diagnostic foundation: 12 diagnostic sidecars plus `.agent/decisions/source_mining_operator_table.md`.
- S2/S2.1 attempted unchanged ADR-023 d02 baseline: both fell back to synthetic data because the real replay probe timed out with zero stdout/stderr.
- S3-narrow stabilizer cleanup: scanner moved from 28 to 20 experiment-backed findings and from 8 to 37 source-backed findings; `_mu_continuity_increment` is still deferred.
- Gate policy: ADR-024 makes warm-bubble an operator-sanity diagnostic. Current honest verdict is `FAIL_PHYSICAL_BOUNDS`, not an amplitude miss.

In flight:
- S2.2 d02 replay hang debug: find and fix why `scripts/m6_d02_boundary_replay_1h.py --duration-s 1` hangs.
- S4-prep Tier-3 convergence infrastructure: build the idealized dt-convergence runner and schema so S4 can start quickly.
- This doc refresh sprint.

Queued:
- S2.1-redo real baseline after S2.2 fixes replay startup.
- S3-real source-backed mu/metric cleanup using real baseline deltas.
- S4 Tier-3 controlled convergence.
- S5 6h/24h Tier-4 Gen2/observation comparator.
- S6 closeout or explicit architecture blocker.

## HYBRID Plan Position

The close-strategy critic returned `HYBRID`: diagnostics first, baseline before operator changes, Tier-3 before Tier-4, and separate ADR-024 gate-policy acceptance from ADR-023 architecture acceptance. The project is between HYBRID S2 and S3: S1 is done, S2 is blocked on infrastructure, S3-narrow handled bounded provenance cleanup, and S3-real must wait for real replay evidence unless the manager deliberately accepts a weaker path.

## Parallel Intel

ADR-021 strip result: the carry-expansion path is not a clean fallback. Removing the warm-bubble clamps and harness aids produced `FAIL_FINITENESS` at step 2, with theta perturbations around +/-22,000 K and signed vertical velocities around +/-1.6e8 m/s at step 1. ADR-021 is branch evidence only unless a new sourced stabilization plan is reviewed.

Gen2 baseline result: `data/fixtures/gen2_baseline/rmse_summary.csv` now gives real d02 forecast-to-forecast anchors from 17 same-grid Gen2 pairs. Spatial-mean RMSE at 24 h is T2 0.628 K, U10 1.456 m/s, V10 1.591 m/s; at 72 h it is T2 0.255 K, U10 0.888 m/s, V10 0.870 m/s. These are Gen2 consistency anchors, not observation-error proof.

## Open Questions

1. Should S3-real be barred until S2.2 produces a real d02 baseline, or may it proceed with an explicit "no real before/after baseline" caveat?
2. Should ADR-024 be promoted independently once review accepts the gate policy, while ADR-023 remains proposed until mu/stabilizer evidence passes?
3. If S2.2 finds the replay hang is environmental rather than code, should the manager run the real baseline on a different machine or define a smaller real-data replay target?

## Time To M6 Close

Best estimate: 3-6 focused sprints after S2.2 unblocks the real d02 replay. In wall time, that is roughly 1-3 days if the hang fix is small and S3-real localizes the mu/stabilizer issue; longer if S2.2 exposes a JAX/GPU infrastructure defect or S4 Tier-3 fails structurally.
