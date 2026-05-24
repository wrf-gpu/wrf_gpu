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

## M6B0 decisions recorded by worker

- **WRF instrumentation strategy**: isolated wrapper plus reviewable `module_small_step_em.F` hook anchors under `external/wrf_savepoint_patch/`. The protected operational binary at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` is never overwritten; `proof_instrumented_build.txt` records matching before/after SHA-256 values. M6B1 should replace the wrapper with a true relinked Fortran hook build if reviewer requires direct `wrf.exe` emission rather than the isolated extractor path.
- **Savepoint file format**: `npz-bundle-v1`, written by `src/gpuwrf/validation/savepoint_io.py`. This was chosen over HDF5/NetCDF for the first harness because it is dependency-light, easy to checksum, and sufficient for column/patch fixtures; the schema keeps `file_format` explicit so a future HDF5 backend can coexist.
- **Frozen schema field list**: `run_id`, `wrf_version`, `wrf_commit`, `namelist_hash`, `source_path`, `domain_index`, `tier`, `operator`, `boundary`, `dt_seconds`, `rk_stage_index`, `acoustic_substep_index`, `map_factors`, `vertical_grid`, `variables`, `schema_version`, `file_format`, `sanitizer_mode`, `created_utc`, and `notes`. Per-variable metadata is `name`, `dtype`, `shape`, `stagger`, `units`, `provenance`, and `role`.
- **Tolerance ladder, coefficient construction**: `cofrz=1e-12`, `cofwr=1e-12`, `cofwz=1e-11`, `coftz=1e-11`, `cofwt=1e-12`, `rdzw=1e-12`, `tri_a=1e-11`, `tri_b=1e-11`, `tri_c=1e-11` absolute max-delta. Rationale is recorded in `proof_comparator_tolerance_ladder.txt`; M6B1 must revisit these against true Fortran `calc_coef_w` dumps.
- **Golden-domain progression**: yes. The approved progression is one Canary d02 column, then a 16x16 Canary d02 patch, then full d02 only when storage and hook coverage justify it. M6B0 defers Tier-3 full-domain savepoints and records the linear storage estimate in `proof_savepoint_storage_estimate.txt`.

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

1. **Resolved**: no Serialbox dependency is assumed for M6B0; harness is hand-rolled NPZ.
2. **Resolved for M6B0**: canonical boundaries are `coefficient_construction`, `mu_muts_muave_ww_start`, `mu_muts_muave_ww_end`, `t_2ave_update`, `ph_tend_accumulation`, `advance_w_entry`, `advance_w_exit`, `pressure_geopotential_restoration`, `acoustic_substep_start`, `acoustic_substep_end`, and `rk_stage_end`.
3. **Resolved**: allowed stagger metadata values are `mass`, `u`, `v`, `w`, `eta-half`, `eta-full`, and `scalar`.
4. **Resolved for first operator**: coefficient-construction tolerances are listed above and in `proof_comparator_tolerance_ladder.txt`.
5. **Resolved**: column and 16x16 d02 slices precede full d02; full-domain Tier-3 is deferred from M6B0.
