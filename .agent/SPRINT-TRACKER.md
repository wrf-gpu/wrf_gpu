# Sprint Tracker — Live Dashboard

Manager-maintained live state. Last refreshed by doc-refresh worker on 2026-05-24.

## Currently In Flight

| Sprint | Purpose | Current state |
|---|---|---|
| `2026-05-24-m6x-s2dot2-d02-replay-hang-debug` | Find and fix why `scripts/m6_d02_boundary_replay_1h.py --duration-s 1` hangs with zero stdout/stderr. | Dispatched on `worker/gpt/m6x-s2dot2-d02-replay-hang-debug`; report files still templates in this worktree. |
| `2026-05-24-m6x-s4prep-tier3-convergence-infra` | Build Tier-3 idealized dt-convergence runner/schema so S4 can run quickly after S2.2/S3-real. | Dispatched on `worker/gpt/m6x-s4prep-tier3-convergence-infra`; report files still templates in this worktree. |
| `2026-05-24-m6x-doc-refresh` | Refresh repo-level docs after the M6.x pivot chain. | This sprint, branch `worker/gpt/m6x-doc-refresh`. |

## Recently Completed In Last 24h

| Sprint | Outcome | Evidence |
|---|---|---|
| `m6x-warm-bubble-gate-strategy-critic` | `CHANGE-THE-GATE`: old `[5, 10] m/s` warm-bubble amplitude target is unsourced for the current pure-small-step harness. | `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md` |
| `m6x-warm-bubble-gate-redesign` | ADR-024 PROPOSED; warm-bubble now reports operator sanity. Current main verdict is `FAIL_PHYSICAL_BOUNDS` from mu perturbation exceeding the 50 kPa bound. | `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` |
| `m6x-warm-bubble-failure-diagnostic` | Opus `MIXED`: confirmed pressure-diagnose wiring bug plus architectural/stabilization gap. | `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md` |
| `m6x-pressure-diagnose-wiring-fix` | Fixed the pressure overwrite bug; did not solve the mu/stabilizer issue. | `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/worker-report.md` |
| `m6x-adr021-clamp-strip-honest-test` | ADR-021 clamp-free path failed catastrophically: `FAIL_FINITENESS` at step 2 after enormous theta and vertical-velocity growth. | `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/worker-report.md` |
| `m6x-gen2-rmse-baseline-characterization` | Produced Gen2 d02 24h/72h RMSE anchors from 17 same-grid forecast-to-forecast pairs. | `data/fixtures/gen2_baseline/rmse_summary.csv` |
| `m6x-close-strategy-plan-critic` | `HYBRID`: diagnostics first, real baseline before operator changes, Tier-3 before Tier-4, explicit closeout/blocker. | `.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md` |
| `m6x-s1-diagnostic-foundation` | Built 12 diagnostic sidecars and the source-mining operator table. | `.agent/sprints/2026-05-23-m6x-s1-diagnostic-foundation/worker-report.md` |
| `m6x-s2-d02-baseline-instrumented` | Partial/blocker: real replay probe timed out after 120s; only synthetic fallback sidecars were produced. | `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/worker-report.md` |
| `m6x-s2dot1-d02-baseline-real-rerun` | Blocker persisted: real replay probe timed out after 1800s with zero stdout/stderr. | `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/worker-report.md` |
| `m6x-s3narrow-stabilizer-source-cleanup` | PASS: stabilizer scanner moved 28→20 experiment-backed and 8→37 source-backed; `_mu_continuity_increment` deferred. | `.agent/sprints/2026-05-23-m6x-s3narrow-stabilizer-source-cleanup/worker-report.md` |

## Queue

| Next | Sprint | Gate |
|---|---|---|
| 1 | S2.1-redo real d02 baseline | Wait for S2.2 to make `--duration-s 1` and short replay runs complete. |
| 2 | S3-real source-backed mu/metric cleanup | Needs real baseline deltas unless manager explicitly accepts weaker evidence. |
| 3 | S4 Tier-3 controlled convergence | Needs S4-prep infrastructure and post-S3 operator state. |
| 4 | S5 6h/24h Tier-4 comparator | Needs Tier-3 pass; Gen2 RMSE anchors exist for T2/U10/V10. |
| 5 | S6 closeout or architecture blocker | Promote only evidence-backed decisions; otherwise write an explicit blocker. |

## Current Technical Read

ADR-023 is the active proposed architecture on main, but not accepted. ADR-021 is not a clean fallback after clamp stripping. ADR-024 changes the warm-bubble gate to operator sanity; M6 close is Tier-3 plus initial Tier-4 consistency, conservation/bounds, and clean transfer audit.

The immediate blocker is infrastructure, not an operator verdict: the real d02 replay produces no output before timeout, so the project still lacks a real unchanged-operator d02 baseline.
