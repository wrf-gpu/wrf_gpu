# ADR-025 — WRF Small-Step Savepoint Harness + B-Direct Bottom-Up Port (PROPOSED)

**Status**: **PROPOSED** — M6B0-R resolved the file-format, instrumentation, CPU-path, tolerance-ladder, and golden-slice ordering decisions. Acceptance still requires M6B0-R review.
**Date**: 2026-05-24
**Author**: Manager (Claude Opus 4.7, 1M-context)
**Triggered by**: External deep-consultation response (2026-05-24); HYBRID exit-rule firing on S2.1-redo real d02 baseline; `NO-BUG-LOCALIZED` verdict from `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md`.
**Supersedes**: ADR-023 (now SUPERSEDED-PROVISIONAL).

## Decision

The M6 dycore close path is **B-direct with savepoint-first discipline**:

1. **Build a CPU WRF `module_small_step_em` savepoint harness** that emits per-operator inputs and outputs for the exact Canary d02 case (column → 16×16 patch → full d02 progression).
2. **Build a JAX-side savepoint comparator** that ingests WRF HDF5 dumps and reports per-operator delta against the JAX implementation. Must fail loudly on a deliberate-perturbation negative test.
3. **Port the WRF small-step bottom-up** under per-operator parity gates (sanitizer-off; no caps; no clamps; no tanh): coefficient construction → tridiagonal solve → scratch state (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, save fields) → acoustic recurrence → full dycore step → coupled step.
4. **Only after the 10-step coupled parity gate passes** may the project re-run the 1h Canary d02 RMSE comparison (M6b) and the 6h/24h Gen2 consistency (M6c).
5. **No optimization sprints until M6B5 passes.** Performance is gated by correctness.

## Why this supersedes ADR-023's thesis

ADR-023 attempted to keep the acoustic carry minimal (no expansion to WRF small-step scratch families) and relied on a conservative tridiagonal column solver patterned after MPAS/SCREAM/ICON4Py. The thesis was scientifically defensible but failed real Gen2 d02 evidence at 1h: 136.9 K T2 RMSE, 17 B sanitized nonfinites, θ at sanitizer caps. The subsequent operator bug-hunt ruled out single-operator-bug fixes via seven sanitizer-bypass A/B toggles.

The external consultation's correction:

> "Stop trying to make a WRF-like dycore stable from the outside. Rebuild the WRF small-step from the inside, under savepoint parity, then optimize."

> "WRF compatibility cannot be validated only at 1 hour or 24 hours. At this point it must be validated at the acoustic substep level."

The replacement architecture is to **use WRF itself as the numerical compiler**: extract per-operator savepoints, reproduce them in JAX, then port the small-step bottom-up under hard parity.

## M6B0-R decisions recorded by worker

- **WRF instrumentation strategy**: Fortran wrapper module plus `#ifdef WRF_SAVEPOINT` call-site patch artifacts under `external/wrf_savepoint_patch/`. The protected operational binary at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` is never overwritten; pre/post SHA-256 checks are mandatory in `external/wrf_savepoint_patch/build.sh`. The current M6B0-R executable is a CPU savepoint emission shim compiled through the WRF GPU environment and HDF5 toolchain; the patch artifacts define the direct relink path for reviewer follow-up.
- **Savepoint file format**: `hdf5-savepoint-v1`, written by `src/gpuwrf/validation/savepoint_io.py` through `h5py`. Each savepoint stores metadata as canonical JSON, field arrays under `/fields`, gzip/shuffle compression where chunkable, and a SHA-256 payload digest for tamper detection.
- **Frozen schema field list**: `run_id`, `wrf_version`, `wrf_commit`, `namelist_hash`, `source_path`, `domain_index`, `tier`, `operator`, `boundary`, `dt_seconds`, `rk_stage_index`, `acoustic_substep_index`, `map_factors`, `vertical_grid`, `variables`, `schema_version`, `file_format`, `sanitizer_mode`, `created_utc`, and `notes`. Per-variable metadata is `name`, `dtype`, `shape`, `stagger`, `units`, `provenance`, and `role`.
- **Boundary registry**: `calc_coef_w_pre/post`, `small_step_prep_post`, `advance_mu_t_pre/post`, `advance_uv_post`, `advance_w_rhs_ready`, `advance_w_raw_w`, `advance_w_tridiag_fwd`, `advance_w_tridiag_back`, `advance_w_rayleigh`, `advance_w_ph_final`, `calc_p_rho_post`, `small_step_finish_post`, `acoustic_substep_boundary`, and `rk_stage_boundary`. Boundary/microphysics savepoints remain reserved for M6B6+.
- **Tolerance ladder**: machine-readable JSON at `src/gpuwrf/validation/tolerance_ladder.json`, including units, dtype, abs/rel/ULP thresholds, accumulation exceptions, and a rule that deliberate perturbations must be at least 10x the active pass tolerance. Comparator decisions are generated from this ladder.
- **Golden-domain progression**: mandatory ordering is column, 16x16 patch, fixed golden small-domain slice, then any future full d02 stretch. M6B0-R pins the golden run ID and records terrain/map-factor metadata and storage bytes.
- **CPU vs GPU operator path**: M6B0-R parity is CPU-path only against `module_small_step_em`; `external/wrf_savepoint_patch/namelist.savepoint` documents the fallback flags. GPU companion operators are deferred to M6B6+.

## Acceptance criteria for ADR-025 to move PROPOSED → REVIEWED

- M6B0-R review confirms whether the Fortran emission shim plus patch artifacts are sufficient, or whether direct relinked `wrf.exe` emission must be completed before M6B1.
- The HDF5 schema and tolerance ladder receive critic sign-off.
- The JAX comparator passes the deliberate-perturbation and tamper-detection negative tests.
- The `calc_coef_w` parity proof is reviewed honestly as either `PASS` or `PARITY-DEFECT-LOCALIZED`, sanitizer-off, on column, 16x16, and golden tiers.

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

1. **Resolved**: no Serialbox dependency is assumed for M6B0-R; harness is hand-rolled HDF5.
2. **Resolved for M6B0-R**: canonical boundaries are listed in the boundary registry above.
3. **Resolved**: allowed stagger metadata values are `mass`, `u`, `v`, `w`, `eta-half`, `eta-full`, and `scalar`.
4. **Resolved for first operator**: coefficient-construction tolerances are listed above and in `proof_comparator_tolerance_ladder.txt`.
5. **Resolved**: column and 16x16 d02 slices precede the fixed golden small-domain slice; full d02 is deferred.
6. **Resolved**: CPU path is mandatory for first parity; GPU path is deferred.
