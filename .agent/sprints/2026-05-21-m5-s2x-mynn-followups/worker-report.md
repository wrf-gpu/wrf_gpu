# M5-S2.x Worker Report - MYNN Follow-Ups

## Objective

Close the M6 prologue MYNN follow-ups without re-implementing MYNN: add a WRF-harness-backed independent mean-field budget probe, resolve the flux-Richardson radicand guard, define the M6-S3 surface-layer coupling interface, and keep the accounting honest.

## Files Changed

- `scripts/wrf_mynn_harness.f90`: WRF object-linked harness now appends raw `mynn_tendencies` outputs `du`, `dv`, `dth`, and `dqv` as columns 19-22 (`scripts/wrf_mynn_harness.f90:121`, `scripts/wrf_mynn_harness.f90:125`).
- `scripts/m5_generate_mynn_fixture.py`: fixture parser and manifest schema now carry `output_du`, `output_dv`, `output_dtheta`, and `output_dqv` with units/tolerances (`scripts/m5_generate_mynn_fixture.py:137`, `scripts/m5_generate_mynn_fixture.py:155`, `scripts/m5_generate_mynn_fixture.py:243`, `scripts/m5_generate_mynn_fixture.py:266`).
- `src/gpuwrf/validation/tier2_mynn.py`: added the independent WRF tendency comparison and writer for `artifacts/m5/tier2_mynn_independent_budget.json` (`src/gpuwrf/validation/tier2_mynn.py:20`, `src/gpuwrf/validation/tier2_mynn.py:110`, `src/gpuwrf/validation/tier2_mynn.py:138`, `src/gpuwrf/validation/tier2_mynn.py:193`).
- `src/gpuwrf/physics/mynn_pbl.py`: removed the JAX-only radicand clamp by routing level-2 flux Richardson through an unguarded WRF-faithful helper (`src/gpuwrf/physics/mynn_pbl.py:175`, `src/gpuwrf/physics/mynn_pbl.py:206`).
- `src/gpuwrf/physics/mynn_surface_stub.py`: added the `surface_layer(state) -> SurfaceFluxes` M6-S3 hook and expanded the typed flux contract with `rhosfc` and `fltv` (`src/gpuwrf/physics/mynn_surface_stub.py:27`, `src/gpuwrf/physics/mynn_surface_stub.py:41`, `src/gpuwrf/physics/mynn_surface_stub.py:76`).
- `scripts/m5_run_mynn.py` and `scripts/m5_gate_mynn.py`: validation now writes/requires the independent WRF budget artifact.
- `tests/test_m5_mynn_tier2.py`, `tests/test_m5_mynn_radicand.py`, and `tests/test_m5_mynn_gate.py`: added AC1/AC2 coverage and gate assertion.
- `.agent/decisions/ADR-008-mynn-jax-implementation.md`: amended with the radicand decision and surface-layer interface (`.agent/decisions/ADR-008-mynn-jax-implementation.md:30`, `.agent/decisions/ADR-008-mynn-jax-implementation.md:34`).

## Acceptance Evidence

AC1 independent budget probe: pass. The proof object `artifacts/m5/tier2_mynn_independent_budget.json` compares JAX one-step mean-field tendencies against WRF `mynn_tendencies` arrays at the same input state. Max absolute residuals are `u=2.5574529788071048e-05`, `v=9.349085167409388e-06`, `theta=1.5993035522872863e-06`, and `qv=2.3522538483006757e-10`, all below the fixed `1e-3` target. This is no longer solver-self-consistency: the WRF arrays come from the harness call to real `mynn_tendencies`, whose output arguments are `Du,Dv,Dth,Dqv` in WRF (`/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/module_bl_mynnedmf.F90:4195`, `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/module_bl_mynnedmf.F90:4257`).

AC2 radicand decision: pass. Chose Path A. WRF does plain `SQRT(ri**2 - ri3*ri + ri4)` in the flux-Richardson formula (`/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/module_bl_mynnedmf.F90:1918`, `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/module_bl_mynnedmf.F90:1919`), so JAX now propagates NaN for negative radicands instead of clamping. `tests/test_m5_mynn_radicand.py:13` constructs a discriminant-positive boundary case and asserts both the WRF plain-sqrt formula and JAX helper return NaN.

AC3 surface-layer interface: pass. ADR-008 now defines MYNN inputs `ustar`, `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc`, and `fltv`, sign convention, grid location, outputs, and synchronous RK3 substep timing. WRF citations in the ADR cover surface TKE production (`module_bl_mynnedmf.F90:3421`, `:3428`), scalar flux sign conventions (`:1575`, `:1578`), drag use (`:4436`, `:4510`), scalar RHS use (`:4589`, `:4796`), surface density (`:4311`), and virtual heat flux (`:970`, `:977`). No real Monin-Obukhov code was implemented.

AC4 honest accounting: pass for this sprint scope. No `min(raw, cap)` launch/report fudge was introduced. `artifacts/m5/mynn_profile.json` reports raw and displayed launch counts as 35, HLO size 279074 bytes, and post-init H2D/D2H/temp bytes as zero. Tier-1 numbers are unchanged from M5-S2: `u=7.672e-4`, `theta=6.281e-5`, `tke=1.460e-6`, `el=3.064e-3`.

AC5 tests: MYNN-focused tests pass; full-suite pass count is not proven in this isolated worktree. `pytest -q tests/test_m5_mynn_*.py` returned `11 passed in 9.53s`. Full `pytest -q` returned `394 passed, 11 skipped, 17 failed in 460.00s`; all observed failures are outside the allowed M5-S2.x files: missing external Canary and Thompson artifacts, M2 Triton venv install failure with `[Errno 28] No space left on device` on `/tmp` (96% full), and pre-existing RRTMG Tier-1/gate failures. No MYNN test failed in the full run.

## Commands Run

- `bash scripts/wrf_mynn_harness_build.sh` -> pass, harness SHA `57198fa33207578714eef4d93c79de46687c0642da9edd20742e57cead00ac23`.
- `python scripts/m5_generate_mynn_fixture.py` -> pass, sample SHA `04eb2115e61648266f501c6cc346110d7f609ae3874e7bd081ec3dc2603cb0a5`.
- `python scripts/m5_run_mynn.py` -> pass, writes Tier-1, Tier-2 invariant, independent budget, profile, and HLO artifacts.
- `python scripts/m5_gate_mynn.py` -> pass, `GO_CARRYFORWARD`, `tier2_independent_budget_pass=true`.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-mynn-pbl-column-v1.yaml` -> pass.
- `python scripts/validate_agentos.py` -> pass, `ok=true`, 31 required files, 13 skills.
- `pytest -q tests/test_m5_mynn_*.py` -> pass, `11 passed`.
- `pytest -q` -> failed outside MYNN scope as described above.

## Proof Objects Produced

- `artifacts/m5/tier2_mynn_independent_budget.json`
- `artifacts/m5/mynn_gate_result.json`
- `artifacts/m5/mynn_profile.json`
- `fixtures/samples/analytic-mynn-pbl-column-v1.npz`
- `fixtures/manifests/analytic-mynn-pbl-column-v1.yaml`
- Updated HLO proof files under `artifacts/m5/hlo_dump/`

## Unresolved Risks

The independent budget probe is still dry MYNN2.5 with EDMF/cloud arrays disabled, matching the current harness/JAX scope. Full MYNN-EDMF remains M6 follow-up work. Full-repo pytest needs environment/artifact remediation before a clean 410+ pass can be claimed in this worktree; I did not modify unrelated RRTMG, Thompson, M2 Triton, or Canary fixture code to chase those failures.

## Next Decision Needed

Dispatch the mandatory fresh-context Claude Opus 4.7 reviewer. Reviewer should decide whether AC5 is acceptable with focused MYNN pass plus documented unrelated full-suite blockers, or whether manager wants a separate environment-restoration step before close.
