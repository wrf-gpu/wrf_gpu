# Sprint Contract — M6-S6 Tier-3 TSC1.0 (Drift Envelope)

**Sprint ID**: `2026-05-21-m6-s6-tier3-tsc`
**Created**: 2026-05-21 17:20
**Status**: ACTIVE — dispatching parallel with M6-S5 and M6-S7
**Trigger**: M6-S2 closed + M6-S4 closed (Tier-2 PASS); M6-S6 owns Tier-3 short-run convergence per scout plan + critic amendment

## Objective

Establish dt-sensitivity envelope for coupled forecast and prove GPU drift stays within CPU-baseline envelope per `VALIDATION_STRATEGY.md` Tier-3 family (TSC1.0 method).

Per M6 plan critic amendment #7: use **controlled dt-refinement reduced case**, NOT `wrf_l2 vs wrf_l3` config-noise comparison.

## Acceptance

- **AC1 TSC1.0 dt-sensitivity envelope** on idealized reduced case: GPU forecast at base `dt=18s` and refinement `dt=9s` + `dt=4.5s`; envelope per variable.
- **AC2 CPU envelope reference**: compare Gen2 `wrf_l3` 3km d02 at multiple dt refinements (if available) OR document use of analytic-fixture envelope.
- **AC3 GPU drift within envelope**: per-variable `U10/V10/T2/qv2/precip` at +6/+12/+24h leads — GPU drift NOT exceeding CPU envelope.
- **AC4 Per-variable per-lead status**: GREEN / PARTIAL / BLOCKED / FAIL per variable per lead.
- **AC5 Tier-3 artifact**: `artifacts/m6/tier3/tsc_envelope.json` with base/refined dt, boundary mode, forcing mode, norm definitions, per-variable per-lead envelope + GPU drift + CPU sanity deltas + regridding details.
- **AC6 M6-S4 follow-up F-min-1**: expose Thompson microphysical source/sink tendency as side channel; use as **independent oracle** in water-budget conservation (replaces M6-S4 tautological closure).
- **AC7 M6-S4 follow-up F-min-2**: wrfbdy file decoder; compare GPU boundary tendency against wrfbdy prescribed forcing — substantively closes M6-S4 R-11 tautology.
- **AC8 Schema**: `Tier3DriftEnvelope` schema in `proof_schemas.py`.

## Files Worker May Modify

- `src/gpuwrf/validation/tier3_coupled.py` (NEW)
- `src/gpuwrf/coupling/physics_couplers.py` (expose Thompson tendency side channel for F-min-1)
- `src/gpuwrf/io/boundary_replay.py` (extend with wrfbdy decoder for F-min-2)
- `src/gpuwrf/io/proof_schemas.py` (add `Tier3DriftEnvelope`)
- `scripts/m6_run_tsc.py` (NEW), `m6_gate_tier3.py` (NEW)
- `tests/test_m6_tier3_tsc.py` (NEW)
- `artifacts/m6/tier3/tsc_envelope.json` (NEW)
- Worker report

## File-disjointness with sister sprints

- M6-S5 owns `coupling/driver.py` dycore cap lift + `dynamics/`
- M6-S7 owns `validation/tier4_probtest.py`
- M6-S6 owns `validation/tier3_coupled.py` only

If M6-S6 needs driver instrumentation, coordinate with M6-S5 via shared `validation/` instead of touching `driver.py`.

## HARD RULES

1. NO `min(raw, cap)` fudge
2. Independent oracles for water budget + boundary closure (close M6-S4 tautologies)
3. Per-variable per-lead status reporting (no aggregate-only pass)
4. File-disjoint
5. `/exit` slash-command

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall: **18-30h**
- Worktree: `/tmp/wrf_gpu2_m6s6`
- Branch: `worker/codex/m6-s6-tier3-tsc`
