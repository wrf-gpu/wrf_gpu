# V0.15 Roadmap Draft

Date: 2026-06-10 WEST
Owner: manager
Status: DRAFT; do not start until v0.14 grid-parity and validation gates close.

## Goal

v0.15 should move the project from "v0.14 validated, grid-close, memory-safe"
to a more complete, faster, and more efficient WRF v4-faithful GPU rewrite.
The release goal is not a marketing feature bundle. It is: reduce remaining
compute and memory waste, advance mixed precision only where WRF-faithful, and
close deferred completeness gaps without weakening validation.

## Entry Criteria

Do not open v0.15 implementation sprints until all are true:

- v0.14 Step-1/grid-cell blocker is closed or explicitly bounded with proof.
- v0.14 exact-branch memory preflight is green on the final candidate.
- Switzerland/Gotthard validation has run or has an explicit manager decision.
- Powered TOST is running or complete, paired with the Grid-Delta Atlas.
- Any new v0.15 tolerance envelope is frozen before scoring, not after.

## Immediate Prep

Once TOST and Switzerland compute are started and Fable/Mythos is free, run the
prepared read-only efficiency review:

- sprint: `.agent/sprints/2026-06-10-v015-fable-kernel-efficiency-review/`
- endpoint: ranked v0.15 action list over every major kernel/module
- output: `.agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`
- no source edits, no GPU interruption, no micro-prompts

The review must score each candidate by expected gain and complexity. Complexity
means changed files/contracts, required proof gates, numerical risk, GPU
performance risk, and possibility that the gain will not materialize.

## Draft Priority Table

| Priority | Lane | Why | Current status | v0.15 target |
|---:|---|---|---|---|
| P0 | Fable kernel efficiency review | Avoid drifting into low-gain optimizations; produce one ranked compute+memory action list. | Prepared only; not started because Fable is on v0.14 Step-1. | Static/proof-backed review of dycore, radiation, physics, coupling, writer, validation runners; gain/complexity table. |
| P1 | FP32 acoustic R1-R8 | Highest possible memory/perf gain after correctness; probes say feasible in principle, but fp64 dynamics frontier must be closed first. | R0 default-inert contract landed; ADR-031 draft; R1/R2 blocked by fp64 P/PH/MU/Step-1 divergence. | Opt-in `mixed_perturb_fp32` acoustic mode with fp64-default bit identity, explicit base-state plumbing, perturbation-authoritative loop, CPU/GPU gates, transfer audit. |
| P1 | v0.14 validation hardening carry-over | Final validation output should be reusable, not a one-off. | Grid-Delta Atlas tool exists; tolerance manifest still review/freeze step; TOST and Switzerland pending post-fix. | Freeze tolerance manifest, final README dashboard, reusable validation manifest runner, deterministic exclusion log. |
| P2 | Moisture limiter / active moisture-advection memory and semantics | Measured +1.90 GiB when active; also tied to WRF-cadence fidelity. | Deferred until active moisture path is validation target. | Per-species limiter workspace reduction with positivity/conservation and WRF cadence proof. |
| P2 | Acoustic scan carry split / evolving-only carry | Static memory gain around 1.56 GiB; may combine with FP32 acoustic. | Deferred; prior split attempts reverted; same fault surface as fp64 dynamics. | Co-design with FP32 or separate dycore-memory sprint after fp64 proof frontier closes. |
| P2 | Post-physics sparse/donated merge | Static 1.3-2.6 GiB possible; could improve liveness and compile pressure. | Non-material in current preflight after MYNN tiling; deferred. | Measure after v0.14 final branch; implement only if profiler shows material peak or compile benefit. |
| P2 | PBL/surface bottom-only prep reuse | 0.3-0.8 GiB estimate and correctness side benefit. | Deferred because surface/PBL contracts are semantically delicate. | WRF surface-driver-to-PBL-driver contract proof, then avoid duplicated full-column prep where safe. |
| P2 | Non-radiation column tiling generalization | RRTMG/MYNN tiling produced large wins; other column physics may still have whole-batch peaks. | MYNN BouLac fixed by leading-column tiling; other schemes measurement-first. | Pick measured offender, one scheme at a time, tile-vs-untiled exactness + GPU VRAM suite. |
| P3 | RRTMG tile-size tuning and full-forecast profiling | Current tile is correct; runtime/peak tradeoff unknown. | Correctness fixed; performance not fully tuned. | Full-forecast profiler artifacts, tile-size sweep, transfer audit. |
| P3 | State alias / total-perturb-base schema reduction | Could simplify memory liveness but high ABI risk. | ADR-required, deferred. | ADR plus restart/wrfout/boundary compatibility proof before any source changes. |
| P3 | Real multi-GPU sharding | Needed for true scalable GPU rewrite; fake mesh only today. | Fake-mesh bit identity done; real hardware throughput projected. | Real multi-GPU proof on available hardware: halo correctness, partition invariance, throughput/per-watt artifacts. |
| P3 | AOT/precompile for production grids | Reduces wall-clock and operator friction, not numerical risk. | Runtime code has AOT scaffold; no release-grade artifact. | Precompile/cold-start benchmark for Canary and Switzerland grids, cache integrity docs. |
| P4 | Full wrfout coverage | Gatekeeper completeness; v0.13 writer still focused subset. | KI-3: 104-variable focused writer vs CPU-WRF 375 variables. | Expand writer inventory or formal fail-closed field policy with exact mapping and tests. |
| P4 | Physics long-tail | Completeness toward WRF v4; large schemes remain fail-closed/ref-only. | Many schemes carried: CAM/NUWRF/GFDL radiation, RUC LSM, Grell-3D/KSAS/New-Tiedtke kernels, Shin-Hong/QNSE/UW/GBM/TEMF, huge microphysics. | Select by operational value and proof tractability after v0.15 efficiency review, not by breadth alone. |

## Deferred From v0.14 Into v0.15

- FP32 acoustic R1+ and any mixed-precision validation campaign.
- Acoustic carry split, pad/mask helper cleanup, and state alias reduction.
- Moisture limiter workspace rewrite and active moisture-advection cadence
  hardening if v0.14 does not close it.
- Post-physics merge and PBL/surface bottom-only prep optimizations.
- Broad non-radiation column tiling beyond MYNN.
- Real multi-GPU throughput measurement.
- Full wrfout 375-variable coverage plan.
- Larger physics-scheme long-tail.

## Required Evidence Style

- Compute claims require profiler or wall-clock artifacts on real workloads.
- Memory claims require peak VRAM or compiled-memory artifacts plus transfer
  audit.
- Precision claims require fp64-default bit identity and WRF/analytic gates for
  the mixed mode.
- Completeness claims require fail-closed catalog updates and source/oracle
  proofs, not silent menu expansion.
- Long validation must always include both station TOST and Grid-Delta Atlas;
  station-only claims are not acceptable.

## First Three v0.15 Sprints, If v0.14 Gates Are Green

1. **V015-FABLE-EFF-0: kernel efficiency review.** Read-only, complete ranked
   action list. No source edits.
2. **V015-FP32-R1: explicit base-state plumbing.** Only if fp64 Step-1/dynamics
   frontier is closed. Gate: `fp64_default` bit identity and cancellation probe
   improvement.
3. **V015-VAL-0: frozen validation envelope.** Convert v0.14 Atlas/TOST output
   into a reusable tolerance/exclusion/plot release package, with independent
   review before any new scoring run.
