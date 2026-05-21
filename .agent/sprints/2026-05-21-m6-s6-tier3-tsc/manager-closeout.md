# M6-S6 Manager Closeout — Tier-3 TSC Drift Envelope: ACCEPT-AS-SCAFFOLD-PARTIAL

**Sprint**: M6-S6 Tier-3 TSC1.0 drift envelope + F-min-1/2 oracle infrastructure
**Status**: **CLOSED — Opus ACCEPT-AS-SCAFFOLD-PARTIAL**
**Date**: 2026-05-21 ~23:05
**Manager**: Claude Opus 4.7 (1M-context)

## Headline

Worker delivered 5/8 ACs substantively, 2/8 PARTIAL with REAL oracle infrastructure (F-min-1 Thompson side-channel + F-min-2 wrfbdy decoder both not stubs), 1/8 (AC3 d02 drift) honestly BLOCKED on CUDA OOM with no fudge. Opus reviewer binding: **CLOSE M6-S6 as scaffold + partial closures; DEFER AC3 to post-M6.x; PROPAGATE F-min-S8-1/2/3 to M6-S8.**

## Key findings (Opus §R-1..R-16)

| ID | AC | Finding |
|---|---|---|
| R-1..R-2 | AC1+AC2 | Reduced TSC dt-refinement 18/9/4.5s at +6/+12/+24h leads with raw pairwise max envelope. Analytic-fixture CPU reference (no l2-vs-l3 config noise per critic amendment). |
| R-3 | AC3 | CUDA_ERROR_OUT_OF_MEMORY on d02 segment. status=BLOCKED honestly carried; gate fails. NO fudge. |
| R-7 | F-min-1 | Thompson side-channel oracle is REAL infrastructure but PARTIAL substantive closure of M6-S4 R-10 — exposes end-state delta, not internal source/sink rates. Load-bearing in coupled context (catches non-Thompson water imbalance); the isolated-Thompson probe demo is one step removed. |
| R-9 | F-min-2 | wrfbdy decoder REAL, exercises against actual `/mnt/data/canairy_meteo/runs/wrf_l3/.../wrfbdy_d01`. Native `wrfbdy_d02` architecturally doesn't exist (nested WRF derives d02 BCs from d01 runtime feedback). R-11 closes substantively via two independent oracles. |
| R-14 | — | Inherited dycore cap (`dycore_dt_s=min(dt_s,1.0)`) honestly documented. Envelope semantics will need re-validation under M6.x-uncapped dycore. |

## d02 OOM analysis (Opus diagnosis)

8 GiB OOM on a 32 GB device — NOT genuine RTX 5090 saturation. d02 working set <1 GB; suspect XLA compilation buffer accumulation across reduced-TSC + d02 phases in same Python process (unique `(grid_shape, dt, n_acoustic, radiation_cadence_steps)` recompiles).

**Lowest-risk fix**: process-isolated `--phase {reduced,d02,oracles}` runner. Defer to post-M6.x sprint to avoid running d02 drift against about-to-be-replaced dycore.

## M6-S4 tautology closure status

| Binding follow-up | Status |
|---|---|
| R-10 / F-min-1 water-budget oracle | **PARTIAL** — REAL infra; substantive load-bearing path to be exercised in M6-S8 (state_next post-advection/boundary) |
| R-11 / F-min-2 boundary-flux oracle | **PARTIAL (d01) + SUBSTANTIVE (d02 via replay zarr)** — two independent oracles; add d02-boundary-zone direct GPU-vs-Gen2 wrfout comparison in M6-S8 |
| R-12 / F-min-3 dycore-cap-lift | **DEFERRED to M6.x** — in flight |

## Manager binding decisions

1. ✓ Close M6-S6 (scaffold + partial closures honest).
2. ✓ Inherit **three F-min-S8-* follow-ups** into M6-S8:
   - **F-min-S8-1**: call `water_budget_residual` with Thompson side-channel oracle in coupled operational d02 forecast (substantively closes R-10 for d02)
   - **F-min-S8-2**: add d02-boundary-zone direct GPU-vs-Gen2-wrfout comparison (substantively closes R-11 for d02)
   - **F-min-S8-3**: when post-M6.x d02 drift lands, cross-check operational RMSE against the re-measured (uncapped) TSC envelope
3. ✓ Queue post-M6.x d02-drift follow-up sprint (codex, 6-12h, process-isolated runner).
4. **M6-S8 dispatch is UNBLOCKED** by M6-S6 close. Will dispatch after M6.x lands (M6-S8 should run against uncapped dycore, not the current capped placeholder).
5. ✓ Maintain mandatory M6-S8 disclaimer language (per M6-S5 Opus §5): "throughput established; forecast capability conditional on M6.x close".

## Open at this point

- **M6.x WRF-canonical dycore completion** (codex, 16-32h, in flight; worker actively writing acoustic CFL tests in /tmp/wrf_gpu2_m6x)
- **M6-S8** queued, dispatches after M6.x Opus accept
- **Post-M6.x d02-drift follow-up** queued (6-12h codex)
- **M7 dispatch** BLOCKED pending M6.x + M6-S8 close

## Reviewer signature

Claude Opus 4.7 xhigh, fresh-context post-reboot re-dispatch, independent of M6 manager.
File: `/tmp/wrf_gpu2_m6s6/.agent/sprints/2026-05-21-m6-s6-tier3-tsc/reviewer-report.md` (181 lines)

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 23:05
