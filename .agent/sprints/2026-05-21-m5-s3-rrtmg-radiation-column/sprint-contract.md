# Sprint Contract — M5-S3 RRTMG Radiation Column Kernel

**Sprint ID**: `2026-05-21-m5-s3-rrtmg-radiation-column`
**Created**: 2026-05-21 ~01:15 by manager (Claude Opus 4.7 1M-context)
**Trigger**: ADR-005 deferred-schemes section puts RRTMG at "M5-S3 or M6 boundary". Required for credible M6 operational validation (diurnal heating drives T2 RMSE). User directive 2026-05-21 ~01:10: dispatch in parallel with M6 plan scout + M5-S2 retroactive reviewer; aim for ≥2 sprints concurrent.

## Objective

Implement WRF RRTMG (Rapid Radiative Transfer Model for GCMs) shortwave + longwave radiation column kernel in JAX, following the same governance pattern as M5-S1 Thompson and M5-S2 MYNN: Fortran-harness oracle (structural anti-tautology — link against real `module_ra_rrtmg_*.o` if available), per-field Tier-1 fixture parity, Tier-2 invariants, fused JAX kernel, debuggability HLO discipline. ADR-009 documents implementation choices.

This is the third physics scheme; with Thompson + MYNN already merged, it gives M5 the radiation piece needed for M6 diurnal-cycle realism.

## Non-Goals

- Full 3-D radiation (cloud overlap, terrain shading) — start with column kernel
- Cloud-radiation coupling beyond standard RRTMG inputs (`qc/qi/qs/qg`, cloud-fraction stub)
- Aerosol radiation coupling (use climatology default)
- Solar geometry beyond a passed-in zenith angle (the column kernel takes pre-computed `coszen`)
- Real time-of-day cycling (zenith provided per-call by driver)
- Real LW upward at TOA / SW downward at TOA decomposition beyond what the WRF RRTMG band-summed output provides

## File Ownership

Worker may CREATE:
- `src/gpuwrf/physics/rrtmg_sw.py` (shortwave column kernel)
- `src/gpuwrf/physics/rrtmg_lw.py` (longwave column kernel)
- `src/gpuwrf/physics/rrtmg_constants.py` (constants — gas constants, solar constant, band coefficients)
- `src/gpuwrf/physics/rrtmg_tables.py` (lookup-table loader/binder, device-resident arrays — pattern from M5-S1.x `thompson_tables.py`)
- `scripts/wrf_rrtmg_harness.f90`
- `scripts/wrf_rrtmg_harness_build.sh`
- `scripts/extract_rrtmg_tables.py` (one-shot extractor — pattern from M5-S1.x)
- `scripts/m5_generate_rrtmg_fixture.py`
- `scripts/m5_run_rrtmg.py`
- `scripts/m5_gate_rrtmg.py`
- `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml`
- `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml`
- `fixtures/samples/analytic-rrtmg-sw-column-v1.npz`
- `fixtures/samples/analytic-rrtmg-lw-column-v1.npz`
- `data/fixtures/analytic-rrtmg-sw-column-v1/full.npz`
- `data/fixtures/analytic-rrtmg-lw-column-v1/full.npz`
- `data/fixtures/rrtmg-tables-v1.npz` (extracted lookup tables, gitignored)
- `data/scratch/wrf_rrtmg_harness` (external binary, gitignored)
- `artifacts/m5/tier1_rrtmg_sw_parity.json`
- `artifacts/m5/tier1_rrtmg_lw_parity.json`
- `artifacts/m5/tier2_rrtmg_invariants.json`
- `artifacts/m5/rrtmg_profile.json`
- `artifacts/m5/rrtmg_gate_result.json`
- `artifacts/m5/hlo_dump/rrtmg_sw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_sw_debug_stripped.txt`
- `artifacts/m5/hlo_dump/rrtmg_sw_debug_vs_stripped.diff`
- `artifacts/m5/hlo_dump/rrtmg_lw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_lw_debug_stripped.txt`
- `artifacts/m5/hlo_dump/rrtmg_lw_debug_vs_stripped.diff`
- `src/gpuwrf/validation/tier1_rrtmg.py`
- `src/gpuwrf/validation/tier2_rrtmg.py`
- `tests/test_m5_rrtmg_*.py`
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`
- Worker report.

Worker may NOT modify:
- `src/gpuwrf/physics/thompson_*.py` or `mynn_*.py` (already-merged work)
- ADRs other than new ADR-009 (and minor cross-reference touch to ADR-005 if needed for follow-on hook closure)
- Other sprint folders.
- `feedback_*.md` memory files.
- `MORNING-REPORT.md` if present.

## Inputs

Required read order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/writing-gpu-kernels/SKILL.md`
4. `.agent/skills/validating-physics/SKILL.md`
5. `.agent/rules/sprint-lifecycle.md` — esp. the **double-AI principle hard rule** added 2026-05-21
6. `.agent/SPRINT-TRACKER.md` (your sprint's place in the parallel-management dashboard)
7. **`.agent/sprints/2026-05-21-m5-s3-rrtmg-radiation-column/sprint-contract.md`** — your full spec (this file)
8. `.agent/decisions/ADR-005-first-physics-suite.md` (deferred-schemes section)
9. `.agent/decisions/ADR-006-thompson-jax-implementation.md` (pattern from M5-S1)
10. `.agent/decisions/ADR-008-mynn-jax-implementation.md` (pattern from M5-S2)
11. `.agent/decisions/ADR-007-precision-policy.md` (RRTMG falls under `FP32-OK` for spectral arithmetic; cloud-optical inputs need empirical test)
12. `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/manager-closeout.md` (Thompson playbook)
13. `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/manager-closeout.md` (table-export playbook — HLO-safe array references, NOT JIT-time constants)
14. `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md`
15. WRF source: search for RRTMG sources:
    ```
    find /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF -name 'module_ra_rrtmg_*' 2>/dev/null
    ```
    Likely paths: `WRF/phys/module_ra_rrtmg_sw.F`, `WRF/phys/module_ra_rrtmg_lw.F` (or in subdir; verify with find).
16. Compiled objects: search for `module_ra_rrtmg_*.o`. The harness aim is to link against these (avoid the M5-S2 author-derived-harness anti-tautology gap).

## Acceptance Criteria

1. **Fortran harness** compiles via `nvfortran` and (if `module_ra_rrtmg_*.o` available) **links against real WRF RRTMG compiled objects**. If linking fails (e.g. dependency-chain too large), worker may fall back to source-derived harness BUT must document this as a named anti-tautology gap (NOT swept under) and the reviewer treats it the same as M5-S2.

2. **Lookup tables extracted** into `data/fixtures/rrtmg-tables-v1.npz` via `scripts/extract_rrtmg_tables.py`. Reproducible SHA. Tables include: SW k-distribution coefficients (16 bands), LW k-distribution coefficients (16 bands), gas absorption coefficients, cloud optical properties. Mirror the M5-S1.x extraction pattern.

3. **JAX RRTMG-SW kernel** (`rrtmg_sw.py`) takes column state (T, p, qv, qc, qi, qs, qg, cloud-fraction, surface-albedo, coszen) and produces shortwave heating tendency `dT/dt` per level + TOA/surface fluxes. Single fused `@jit` entry. Tables loaded as device-resident `jnp.array` (NOT JIT-time constants) per M5-S1.x HLO-safe pattern.

4. **JAX RRTMG-LW kernel** (`rrtmg_lw.py`) takes column state + surface temperature/emissivity and produces longwave heating tendency `dT/dt` per level + TOA/surface fluxes. Same fused-JIT discipline.

5. **Tier-1 fixture parity** under per-field tolerances (carry-forward acceptable per validation philosophy; strict ADR-005 deferred to M5-S3.x if needed). Per-band SW/LW heating rates compared. Operational impact metric: column-integrated heating-rate RMSE.

6. **Tier-2 invariants**: SW down at TOA = SW up at TOA + absorbed (energy conservation, ≤1e-10 fractional); LW emission satisfies Stefan-Boltzmann at surface within numerical tolerance; no NaN/Inf in heating rates.

7. **Profile metrics**: target ≤5 launches per call (SW + LW combined); 0 temp bytes; 0 H2D post-init; HLO ≤500 KB per kernel (RRTMG is larger than Thompson/MYNN due to band structure).

8. **HLO debug-vs-stripped diff = 0 bytes** per the M4+ debuggability pattern.

9. `python scripts/validate_agentos.py` passes.

10. `pytest -q` passes (count grows by new RRTMG tests; no regression).

## Validation Commands

```bash
bash scripts/wrf_rrtmg_harness_build.sh
python scripts/extract_rrtmg_tables.py --output data/fixtures/rrtmg-tables-v1.npz
python scripts/m5_generate_rrtmg_fixture.py
python scripts/m5_run_rrtmg.py
python scripts/m5_gate_rrtmg.py
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml
python scripts/validate_agentos.py
pytest -q
```

## Performance Metrics

- `artifacts/m5/rrtmg_profile.json`: launches ≤5, temp 0, H2D post-init 0.
- HLO size per kernel ≤500 KB.
- HLO debug-vs-stripped diff 0 bytes (both SW + LW).

## Proof Object

- Worker report ≥3000 bytes per-AC verdict with file:line citations.
- ADR-009-rrtmg-jax-implementation.md ≥1500 bytes documenting key choices (band structure, optical-property handling, cloud-overlap assumption, table-extraction approach).

## Risks

- **Largest spectral table volume of any M5 scheme**: 32 bands × per-band coefficients. HLO size + lookup gather discipline are the load-bearing concern (mirror M5-S1.x learnings; tables as device-resident array references, NOT JIT constants).
- **Solar geometry**: keep `coszen` as an input parameter for now; pre-computed by driver. Full diurnal cycling is a follow-on (M6 driver responsibility).
- **Cloud-radiation coupling**: Thompson hydrometeors (qc/qi/qs/qg) feed into RRTMG cloud-optical-property block. Worker may use simplified cloud-fraction (maximum-random overlap) — full overlap option deferred to M5-S3.x.
- **Aerosol**: use climatology defaults; full aerosol-radiation coupling deferred to M7.
- **Real WRF object availability**: if `module_ra_rrtmg_*.o` not in compiled WRF tree, anti-tautology gap (same as M5-S2) — document explicitly, do not paper over.

## Handoff Requirements

- Worker report + ADR-009.
- **Mandatory Claude Opus 4.7 reviewer pass** per the new sprint-lifecycle hard rule (no manager-only close).
- Manager closeout + merge after reviewer Accepts.

## Dispatch Pattern

- Primary worker: codex gpt-5.5 xhigh (frontrunner, solo for code-writing phase).
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY per sprint-lifecycle hard rule).
- Gemini: reactive only — if codex worker or Opus reviewer fail to find a bug, manager dispatches Gemini bug-chase side-runner.

## Bigger-steps authorization

- Worker may fix minor inline issues without filing blocker; only file blocker for substantive scope changes or true regressions.
- Reviewer authorized to apply trivial (≤3 line) fixes inline if confident; otherwise file findings for manager triage.

## Expected wall-time

Worker phase: 6-12 hours (RRTMG is bigger than Thompson/MYNN due to band structure + larger Fortran harness scope).
Opus reviewer phase: 1-2 hours (per double-AI principle).
Manager merge + closeout: 30 min.
Total: 8-15 hours wall-clock.

When done, commit + push to `worker/codex/m5-s3-rrtmg-radiation-column` + `/exit`.
