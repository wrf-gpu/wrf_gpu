# Sprint Contract — M5-S3.x RRTMG Transfer Solver (M6 Prologue)

**Sprint ID**: `2026-05-21-m5-s3x-rrtmg-transfer-solver`
**Created**: 2026-05-21 06:15 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — to dispatch in M6 prologue alongside M5-S1.x HLO-fusion + Thompson process-residual work
**Trigger**: M5-S3 closed ACCEPT-AS-GROUNDWORK; Opus reviewer A3 identified hand-rolled SW two-stream + fabricated tau_gas + non-Eddington LW transfer as physics gaps preventing M6 coupled validation

## Objective

Replace the M5-S3 hand-rolled simplifications with real RRTMG band-by-band transfer solver:
1. SW: Eddington two-stream + delta-scaling (Joseph et al. 1976, Meador-Weaver 1980)
2. LW: correlated-k integration with proper g-point logic per RRTMG conventions
3. Real gas absorption (replace `tau_gas = vapor_path * 0.01 * log1p(gas_coeff)` fabricated curve at `rrtmg_sw.py:177`)
4. Cloud-radiation coupling matching RRTMG inflow conventions

## Acceptance (pre-M6-coupled-validation gate)

- Tier-1 strict pass: SW flux residual ≤1 W/m², LW ≤1 W/m², heating ≤1e-4 K/s (the tolerances M5-S3 left vacuous-removed and now correctly tight)
- Tier-2 conservation/closure non-tautological
- Profile: HLO ≤500 KB per kernel; raw launches ≤10 per call (radiation is multi-band, target lower not realistic)
- ADR-009 amended with real-Eddington + correlated-k formulas + WRF citations
- M6 coupled validation can run AFTER this merges

## Inputs (carried forward from M5-S3)

- `data/fixtures/rrtmg-tables-v1.npz` (1.5 MB real WRF data, byte-exact READ-list extraction — preserve)
- `scripts/wrf_rrtmg_harness.f90` (real driver binding via `nm`-verified 106 symbols — preserve)
- `scripts/m5_generate_rrtmg_fixture.py` (carries forward)
- `scripts/m5_gate_rrtmg.py` (honest FALLBACK semantics — preserve)

## Files Worker May Modify

- `src/gpuwrf/physics/rrtmg_sw.py` (REWRITE: real Eddington two-stream + delta-scaling + band loop)
- `src/gpuwrf/physics/rrtmg_lw.py` (REWRITE: real correlated-k LW transfer)
- `src/gpuwrf/physics/rrtmg_constants.py` (add transfer-solver constants per WRF)
- `src/gpuwrf/physics/rrtmg_tables.py` (extend if real correlated-k needs additional table dimensions)
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (amend with real-physics formulas)
- `tests/test_m5_rrtmg_*` (extend)
- Worker report

## Dispatch (when M6 prologue starts)

- Primary worker: codex gpt-5.5 xhigh (per frontrunner role)
- Reviewer (mandatory per sprint-lifecycle hard rule): Claude Opus 4.7 xhigh
- Wall-time: 8-16 hours

## Sequencing

Part of M6 prologue alongside:
- M5-S1.x (Thompson HLO-safe table-gather + process residual)
- M5-S2 follow-ups (4 deferrable items from M5-S2-A2 reviewer)
- M5-S3.x (this sprint)

All three M6 prologue items can run in parallel (different file ownership). All must close before M6 coupled-forecast dispatch.
