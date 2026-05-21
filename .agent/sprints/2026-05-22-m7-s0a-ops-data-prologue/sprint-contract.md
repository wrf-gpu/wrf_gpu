# Sprint Contract — M7-S0a Operational/Data Readiness Prologue

**Sprint ID**: `2026-05-22-m7-s0a-ops-data-prologue`
**Created**: 2026-05-22 ~00:15
**Status**: ACTIVE
**Trigger**: Plan critic 2026-05-22 finding §AC4 — M7 plan allows S0 to run in parallel with M6 closeout (emit BLOCKED if M6 evidence missing). Manager was wasting time serializing operational work behind M6.x. This sprint catches that miss.
**Parallel with**: M6.x dycore (file-disjoint — no dynamics touched).

## Objective

Build the operational/data readiness scaffolding that does NOT depend on M6.x: AIFS/WPS ingest manifest, station observation source manifest, output/status schemas, Gen2 corpus backfill plan, and M7-S0 sprint contract. Land BLOCKED status on M6-dependent claims; deliver everything else.

## Acceptance

- **AC1 AIFS/WPS ingest manifest**: schema-validated YAML/JSON manifest for AIFS GRIB2 → WRF wrfinput/wrfbdy pipeline. Cite Gen2's existing WPS configuration on disk (READ-ONLY at `/mnt/data/canairy_meteo/...`). Document v0 reuse-Gen2-WPS strategy per M7 plan §331-333.
- **AC2 Station observation source manifest**: identify and document station obs sources (METAR, SYNOP, AEMET local network) for Canary domain. Per M7-S5 (`m7-milestone-plan.md:191-213`): observation source manifest + operational scores against station observations. Manifest only — no live ingest yet. Status of each source: AVAILABLE / PARTIAL / UNAVAILABLE.
- **AC3 Output/status schemas**: skeleton schemas for operational output (NetCDF/Zarr) + operational status JSON (per M7-S4 `:160,182` and M7-S7 `:255,270`). Extend `proof_schemas.py` with `OperationalOutput`, `OperationalStatus`, `OperationalScheduler` schemas.
- **AC4 Gen2 corpus backfill plan**: concrete plan to grow the 3-complete-runs corpus to 10+ complete runs needed for production Tier-4 (per M6-S7 reviewer §8). Owner identification (Canairy team coordination), wall estimate, blockers.
- **AC5 M7-S0 sprint contract draft**: pre-drafted ready-to-dispatch contract for M7-S0 (Tier-4 RMSE harness) consuming M6.5-D1's RMSE adapter. Status BLOCKED-on-M6.x but everything else specified.
- **AC6 M6-S8 rename**: M6-S8 sprint contract currently says "operational closeout" but plan critic §AC1 + M7 plan §449-453 say operational requires station observations. **Rename M6-S8 to "model-consistency closeout"** and amend AC1 to explicitly disclaim "operational" language. Reserve "operational" for when station-obs verification is in scope.
- **AC7 1km nest risk audit**: document RTX 5090 32GB memory ceiling for 1km nest per M7 plan `:118,123`. Estimate (cells × leaves × bytes); document compile-buffer overhead from M6-S5/S6 experience. Surface terrain correctness need per RISK_REGISTER.md:15.
- **AC8 Critic findings catalog**: extract every actionable finding from `.agent/sprints/2026-05-22-plan-critic/critique-report.md` into a per-finding tracker with disposition (acted-on/deferred/rejected) and owner.

## Files Worker May Modify

- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/**` (NEW)
- `src/gpuwrf/io/proof_schemas.py` (extend with OperationalOutput/Status/Scheduler schemas)
- `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md` (rename "operational" → "model-consistency"; explicit disclaimer)
- `.agent/sprints/2026-05-22-m7-s0/sprint-contract.md` (NEW, draft)
- `tests/test_m7_s0a_schemas.py` (NEW; validates extended schemas)
- `data/manifests/aifs_ingest_v0.json` (NEW)
- `data/manifests/station_obs_sources_v0.json` (NEW)

## Files Worker Must NOT Modify

- `src/gpuwrf/dynamics/**` (M6.x ownership)
- `src/gpuwrf/coupling/driver.py` (M6.x ownership)
- `src/gpuwrf/contracts/state.py` (M6.x ownership)
- `src/gpuwrf/physics/**` (frozen)
- `/mnt/data/canairy_meteo/**` (READ-ONLY)
- `src/gpuwrf/io/{gen2_wrfout_loader,data_inventory}.py` + `src/gpuwrf/validation/data_quality.py` (M6.5-D1 ownership; no overlap)
- `.agent/decisions/ADR-007*.md`, `ADR-015*.md` (M6.x ownership)
- `.agent/decisions/ADR-016*.md` (M6.5-D1 closeout)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY)
- Wall-time: **12-18h**
- Worktree: `/tmp/wrf_gpu2_m7s0a`
- Branch: `worker/codex/m7-s0a-ops-data-prologue`

## HARD RULES

1. **/mnt/data/canairy_meteo/** READ-ONLY
2. NO dynamics changes — file-disjoint from M6.x
3. v0 manifests reuse Gen2 WPS where possible (no new regridder)
4. BLOCKED status acceptable where M6-evidence missing (per M7 plan §66-68)
5. BEFORE `/exit`: `git add . && git commit && git push`
6. `/exit` slash-command

## End-goal context

Per plan critic: "Run M7-S0a operational/data readiness in parallel now." This is the highest-leverage immediate action. When M6.x lands, M7-S0 dispatches with this scaffolding ready, saving 12-18h of serialization waste.
