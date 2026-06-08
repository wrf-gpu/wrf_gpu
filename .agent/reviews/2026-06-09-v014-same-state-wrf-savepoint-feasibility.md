# V0.14 Same-State WRF Savepoint Feasibility Review

Date: 2026-06-09
Agent: codex
Mode: CPU-only sidecar inspection

## Objective

Determine the fastest reliable path to generate CPU-WRF source-derived term savepoints for the selected h8-h14 dynamic divergence case, using existing WRF source/build trees and existing scripts/tests where possible.

## Files Changed

- `proofs/v014/same_state_wrf_savepoint_feasibility.json`
- `proofs/v014/same_state_wrf_savepoint_feasibility.md`
- `.agent/reviews/2026-06-09-v014-same-state-wrf-savepoint-feasibility.md`

No WRF source files were modified. No repository `src/` files were modified.

## Commands Run

- Read project instructions: `PROJECT_CONSTITUTION.md`, `AGENTS.md`.
- Read V0.14 handoff and plan: `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`, `proofs/v014/same_state_tendency_localization_plan.md`.
- Read local project skills for physics validation and WRF oracle construction.
- Inspected existing savepoint schema, IO, scripts, tests, and old external WRF savepoint patch scaffolding.
- Inspected WRF trees under:
  - `/home/enric/src/wrf_pristine/WRF`
  - `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF`
  - `/home/enric/src/wrf_ideal_f7i/WRF`
  - stale old paths referenced by the external patch
- Inspected Case 3 launch/history directories, namelist, rsl logs, and NetCDF metadata.
- Used ripgrep/source inspection to identify candidate WRF routines and line numbers.

## Proof Objects Produced

- `proofs/v014/same_state_wrf_savepoint_feasibility.json`
- `proofs/v014/same_state_wrf_savepoint_feasibility.md`

The proof objects identify:

- Exact WRF source/build candidates.
- Build availability and provenance risks.
- Candidate routines/files to instrument.
- Minimal patch strategy.
- Expected savepoint artifact schema.
- Risks and mitigations.
- Next implementation sprint contract outline.

## Findings

The fastest practical path is a disposable instrumented copy of `/home/enric/src/wrf_pristine/WRF`. It has a present CPU `main/wrf.exe` and `main/real.exe`, but the tree is dirty and appears to be a serial GNU build, so it should not be patched in place and should not be treated as identical to the historical 28-rank dmpar truth run without provenance checks.

The original Case 3 rsl provenance points at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF`, which exists, but no active build/executable was found there. It is useful for provenance and source comparison, not for the fastest first savepoint run.

Existing repo savepoint scripts/tests are useful as schema/comparison scaffolding, but none currently produce source-derived WRF dycore term truth for the V0.14 h10 same-state problem. The old `external/wrf_savepoint_patch` is not directly usable because hooks are empty and build paths are stale.

## Unresolved Risks

- No Case 3 restart files were found, so h10 must be reached by forward integration from `2026-05-01_18:00:00` unless the next sprint creates a restart.
- The quickest available executable is serial, while the original run used 28 MPI tasks; a dmpar rebuild in the disposable copy may be needed for walltime and closer execution order.
- Step mapping for nested d02 h10 is expected near d02 step 6000, but must be proven with a marker dump before full term emission.
- MPI/tile-safe patch emission requires rank/tile owner metadata or post-merge filtering.
- Current validation schema operator names do not cover all V0.14 term groups, so the implementation should use a proof-local schema unless a separate `src` schema-extension sprint is approved.

## Next Decision Needed

Approve the implementation sprint to create a disposable WRF instrumentation tree from `/home/enric/src/wrf_pristine/WRF`, add env-gated `solve_em.F` boundary hooks first, run h10 marker validation, then emit selected-cell h10 source-derived term savepoints.
