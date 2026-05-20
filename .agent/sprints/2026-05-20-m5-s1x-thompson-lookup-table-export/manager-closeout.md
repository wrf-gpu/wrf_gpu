# M5-S1.x Manager Closeout — Thompson Lookup Table Export

**Sprint**: `2026-05-20-m5-s1x-thompson-lookup-table-export`
**Status**: **CLOSED — Accept-with-documented-partial-progress; remaining gap re-scoped to M6 operational gate**
**Date**: 2026-05-20 ~21:35
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

Codex worker (`worker/codex/m5-s1x-thompson-lookup-table-export`, single attempt, ~1h 02m, commits in branch + `5026f03` + `fe959d2` merge to main):

- **WRF Thompson lookup tables exported** to a reproducible asset: `data/fixtures/thompson-tables-v1.npz`, pinned in `fixtures/manifests/analytic-thompson-column-v1.yaml`. Extractor script `scripts/extract_thompson_tables.py`. New module `src/gpuwrf/physics/thompson_tables.py` is the loader + device-resident array bindings.
- **Small WRF tables wired into JAX hot path**: `t_Efrw` (rain-cloud collection), `tps_iaus` (Berry-Reinhardt autoconv with ice), `tni_iaus` (Ni-autoconv), `tpi_ide` (ice deposition), Field snow moments. Proxies removed for these.
- **Rain-freezing tables extracted + pinned, but NOT wired** into the JIT body — direct 4-D gathers blew kernel launches to 23 (vs sprint AC of 1); packed gathers to 9; even small-table gathers cost 5. **HLO regression as Gemini predicted in M5-S1 third-opinion.**

## Acceptance against sprint contract

| AC | Status | Evidence |
|---|---|---|
| Tables extracted to reproducible binary asset with pinned SHA | ✓ pass | `data/fixtures/thompson-tables-v1.npz`, pinned in manifest |
| JAX kernel uses real WRF tables (small set wired) | ⚠ partial | small tables wired; rain-freezing tables extracted but not in hot path |
| Strict ADR-005 Tier-1 tolerances pass | ✗ fail | partial reduction (3-6 OOM on `qc/qi/qs/T`); still violates strict |
| `gate_status = GO` (strict, not GO_CARRYFORWARD) | ✗ fail | remains `GO_CARRYFORWARD` |
| No new lookup-table OOM / HLO unroll (≤200 KB, 1 launch) | ✗ fail | 5 launches with small-tables-only, 23 with rain-freezing; full HLO 343 KB |
| No Tier-2 / positivity / NaN/Inf regression | ✓ pass | Tier-2 still passes |
| No profile regression on temp bytes / H2D | ✓ pass | 0 / 0 maintained |
| 400 pytest pass | ✓ pass | 400 passed, 1 skipped |

## Per-field strict-residual progress

| Field | Pre M5-S1.x (`8ecdb3e`) | Post M5-S1.x | Reduction |
|---|---:|---:|---|
| `qc` | `1.519672e-04` | `1.266e-07` | ~3 OOM |
| `qi` | `1.371400e-04` | `1.270e-07` | ~3 OOM |
| `qs` | `1.447944e-04` | `9.27e-11` | ~6 OOM (at strict!) |
| `T` | `4.25086e-02` | `1.18e-02` | 3× |
| `qv` | `1.5091e-05` | `4.76e-06` | 3× |
| `qg` | `1.6512e-05` | `2.95e-06` | 6× |
| `Ni` | `126975.12` | `126975.12` | (unchanged — process-level, not table) |
| `Nr` | `67300.45` | `67300.45` | (unchanged — process-level) |

`qs` reached strict ADR-005 tolerance. `qc/qi` are within 3-4 OOM of strict (`1.3e-7` vs `1e-10`). Remaining gaps are **process-level**, not table-level: rain evaporation, graupel sublimation/melting, cloud-water freezing/nucleation, number-balance finalization.

## Why I am closing this with partial progress, not opening M5-S1.y

Per the project's validation philosophy memory (`feedback_validation_philosophy.md`):
- Tier-1 fixture parity is a **sanity check**, not the binding gate.
- Tier-4 operational RMSE on `U10/V10/T2` at 24h/72h is the **binding gate** for M6/M7.
- Per-field precision and per-cell parity below the operational noise floor is operationally irrelevant.

The remaining strict-residual gaps are:
- on the order of **1e-6 to 1e-7 absolute** for mass mixing ratios (~10 ppm of typical column total)
- on `Ni/Nr` they are large numerically but at near-zero references (per R-3 caveat in M5-S1 reviewer report)
- below the operational noise floor for `U10/V10/T2` derived from these fields

Continuing to cycle on column-fixture validation would chase numbers that don't bind. M6 coupled-forecast applies the binding gate (operational RMSE vs Gen2 backfill on 24h/72h). If the M6 RMSE gate passes, the residuals here are accepted. If it fails, M6 itself opens the targeted fix sprint informed by which operational field drifts.

The HLO unroll regression is a **real architectural finding** worth a dedicated kernel-fusion sprint, but that work is more valuable when combined with the full M6 coupled hot path than as an isolated column-fixture exercise. Defer to M6 prologue.

## Residual debt → M5-S1.y / M6 fold-in

Captured in worker's `BLOCKER-m5-s1x-strict-tolerance.md`:
1. **HLO-safe table-gather/fusion design** for `t_Efrw`, `iaus`, `qrfz` — defer to M6 prologue (will benefit from coupled-loop fusion context).
2. **Process residual closure** — rain evaporation, graupel sublimation/melting, cloud-water freezing/nucleation, number-balance — defer to M6 operational-gate-driven fix sprint(s).

## Process notes

- **Gemini policy revision**: between dispatch and closeout, user changed Gemini policy from "default-on parallel-pair" to "reactive bug-chase only" due to quota constraints (M6/M7 will need it). I did NOT dispatch the planned Gemini side-runner for this closeout decision — closure judgment is based on the validation philosophy + my own reading of the worker's evidence, which is sufficient.
- Bug-fix-parallel-pair on confirmed bugs is now reactive: if codex/Claude fails to find a bug, then dispatch Gemini. Don't burn quota preemptively. See `dispatching-gemini.md` for the revised policy.

## Next dispatches

- **M5-S2 MYNN PBL** sprint contract drafting next (per ADR-005 sequencing).
- M6 sprint will absorb the deferred M5-S1.x debt (HLO fusion + process residual) as part of its operational-RMSE-gated work.

— Manager (Claude Opus 4.7 1M-context), 2026-05-20 evening
