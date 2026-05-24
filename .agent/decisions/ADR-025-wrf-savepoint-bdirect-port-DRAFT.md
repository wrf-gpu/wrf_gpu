# ADR-025 — WRF Small-Step Savepoint Harness + B-Direct Bottom-Up Port (DRAFT)

**Status**: **DRAFT** — to be filled in during sprint M6B0 and reviewed at M6B0 close.
**Date**: 2026-05-24
**Author**: Manager (Claude Opus 4.7, 1M-context)
**Triggered by**: External deep-consultation response (2026-05-24); HYBRID exit-rule firing on S2.1-redo real d02 baseline; `NO-BUG-LOCALIZED` verdict from `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md`.
**Supersedes**: ADR-023 (now SUPERSEDED-PROVISIONAL).

## Decision (provisional shape; to be finalized in M6B0)

The M6 dycore close path is **B-direct with savepoint-first discipline**:

1. **Build a CPU WRF `module_small_step_em` savepoint harness** that emits per-operator inputs and outputs for the exact Canary d02 case (column → 16×16 patch → full d02 progression).
2. **Build a JAX-side savepoint comparator** that ingests the WRF HDF5/NetCDF dumps and reports per-operator delta against the JAX implementation. Must fail loudly on a deliberate-perturbation negative test.
3. **Port the WRF small-step bottom-up** under per-operator parity gates (sanitizer-off; no caps; no clamps; no tanh): coefficient construction → tridiagonal solve → scratch state (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, save fields) → acoustic recurrence → full dycore step → coupled step.
4. **Only after the 10-step coupled parity gate passes** may the project re-run the 1h Canary d02 RMSE comparison (M6b) and the 6h/24h Gen2 consistency (M6c).
5. **No optimization sprints until M6B5 passes.** Performance is gated by correctness.

## Why this supersedes ADR-023's thesis

ADR-023 attempted to keep the acoustic carry minimal (no expansion to WRF small-step scratch families) and relied on a conservative tridiagonal column solver patterned after MPAS/SCREAM/ICON4Py. The thesis was scientifically defensible but failed real Gen2 d02 evidence at 1h: 136.9 K T2 RMSE, 17 B sanitized nonfinites, θ at sanitizer caps. The subsequent operator bug-hunt ruled out single-operator-bug fixes via seven sanitizer-bypass A/B toggles.

The external consultation's correction:

> "Stop trying to make a WRF-like dycore stable from the outside. Rebuild the WRF small-step from the inside, under savepoint parity, then optimize."

> "WRF compatibility cannot be validated only at 1 hour or 24 hours. At this point it must be validated at the acoustic substep level."

The replacement architecture is to **use WRF itself as the numerical compiler**: extract per-operator savepoints, reproduce them in JAX, then port the small-step bottom-up under hard parity.

## What this ADR does not yet decide (resolved during M6B0)

- The exact WRF Fortran instrumentation strategy (in-tree patch vs wrapper layer vs preprocessor macros).
- The savepoint file format (HDF5 vs NetCDF vs Serialbox vs custom NPY bundle).
- The savepoint schema field list (must include: run-ID, WRF version/commit, namelist, domain, map factors, vertical grid, timestep, RK stage, acoustic substep, stagger, units, variable provenance).
- The tolerance ladder per Tier (Tier-1 fixture parity tolerances for each operator class).
- Whether a small-domain "golden slice" precedes the full d02 (default: yes).

These are decided in the M6B0 worker-report and reviewer-report, then folded back into this ADR before it moves from DRAFT to PROPOSED.

## Acceptance criteria for ADR-025 to move DRAFT → PROPOSED

- M6B0 produces a working savepoint extractor patch against the Canairy CPU-WRF build (`/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`).
- The savepoint schema is reviewed by a critic and frozen.
- The JAX comparator passes a deliberate-perturbation negative test (i.e., it correctly detects an injected error and refuses to declare parity).
- At least one operator-level parity proof is demonstrated (coefficient construction or tridiagonal solve), sanitizer-off, on one column AND one 16×16 d02 patch.

## Acceptance criteria for ADR-025 to move PROPOSED → ACCEPTED

- M6B0 through M6B6 sprints completed; each operator has a per-step parity proof.
- 10-step sanitizer-off replay produces no nonfinites and matches WRF savepoints within frozen tolerances.
- External WRF-expert human review of the savepoint schema (or Codex critical-review substitute if expert unobtainable) completed and signed off.

## Risk register

- **WRF Fortran patch authorization**: extraction touches Canairy's production WRF build. M6B0 must isolate the instrumented build under a separate path and not regress the operational Gen2 path. Hard requirement, not soft.
- **Comparator false-positive**: if the JAX comparator silently passes due to overly loose tolerances, the entire B-direct ladder collapses. Mandatory deliberate-perturbation negative test at M6B0.
- **Savepoint storage cost**: full d02 savepoints across 10 steps × ~12 operators × all 3D fields could be tens of GB. M6B0 must include a storage estimate and a small-slice progression path.
- **WRF instrumentation overhead**: if savepoint extraction slows CPU WRF beyond practicality, the harness limits to a small-slice sub-domain.

## Open questions for M6B0

1. Does the Canairy WRF build have Serialbox available, or must the harness be hand-rolled?
2. Which `module_small_step_em` entry/exit points are the canonical operator boundaries (suggested initial set: `calc_coef_w` entry/exit; advance_mu_t entry/exit; advance_w entry/exit; pressure restoration entry/exit; one full acoustic substep boundary; one RK stage boundary)?
3. What stagger metadata must accompany every field (u/v/w/mass-point/eta-half/eta-full)?
4. What are the per-field parity tolerances at Tier-1 (fp64 ULP-scale vs relaxed for accumulation operations)?
5. Should the M6B0 deliverable include a golden small-domain (e.g. 32×32×40) before the full d02, or run directly on a Canary d02 slice?
