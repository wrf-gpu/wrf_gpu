# Reviewer Report — M7-S0a Operational/Data Readiness Prologue

**Reviewer**: Claude Opus 4.7 xhigh (mandatory M7-S0a Opus review)
**Worker commit reviewed**: `d4a5e5b` on `worker/codex/m7-s0a-ops-data-prologue`
**Branch**: `worker/codex/m7-s0a-ops-data-prologue`
**Sprint contract**: `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/sprint-contract.md`
**Wall**: ~14 min worker (vs 12-18h budget — fast)
**Date**: 2026-05-22

---

## Verifiability Triple (run independently — not trusting worker transcript)

### 1. Tests reproduce ✅

- `pytest -q tests/test_m7_s0a_schemas.py` → **8 passed in 0.20s** (matches worker report L64).
- `pytest -q tests/test_m6_proof_schemas.py tests/test_m7_s0a_schemas.py` → **11 passed in 0.19s** (no regression; matches L68).

### 2. Gen2 path verification ✅

Independent `ls -la` of every load-bearing path:

- `/mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/l3/namelist.wps` → EXISTS (1092 B, May 21).
- `/mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/ungrib/namelist.wps` → EXISTS (553 B, May 21).
- `/mnt/data/canairy_meteo/data/aifs_ens/aifs_ens_20260414_00z.grib2` → EXISTS (144 MB, Apr 14).

Additional spot-checks against `data/manifests/aifs_ingest_v0.json:137-145`:
- `ungrib/step_000.grib2`, `ungrib/step_024.grib2`, `l3/met_em.d02.2026-05-20_18:00:00.nc`, `Gen2/configs/Vtable.AIFS_PURE`, `data/state/live_18z_state.json`, `artifacts/wps_geog/WPS_GEOG_LOW_RES` → **all EXIST**.

Gen2 station manifest paths in `data/manifests/station_obs_sources_v0.json:114-119` → also all EXIST (registry, AEMET YAML, GRAFCAN YAML, station-cube parquet, two skill-matched parquets).

### 3. Station obs reachability ✅ (with honest PARTIAL marks)

Live curl checks performed independently:

- METAR `https://aviationweather.gov/api/data/metar?ids=GCRR,GCLP,GCTS&format=json` → **HTTP 200, JSON with GCLP/GCRR live obs at reportTime 2026-05-21T23:00:00Z** (lat 27.923, lon -15.389, wind 360@18, T2 20°C). Worker's `AVAILABLE` label honest.
- AEMET `https://opendata.aemet.es/opendata/api/observacion/convencional/todas` → **HTTP 200, empty body** without credentials. Worker's `PARTIAL` label honest — endpoint reachable but auth required (matches manifest L36 `endpoint responded HTTP 200 with an empty text body without credentials`).
- GRAFCAN `https://sensores.grafcan.es/api/v1.0/` → **HTTP 200, Swagger UI HTML**. Worker's `PARTIAL` label honest — inventory/API root reachable but observations require `GRAFCAN_API_KEY`.

**No optimistic AVAILABLE labels detected.** AEMET/GRAFCAN are explicitly gated on credentials in `binding_policy.operational_claim_rule` (L103).

---

## Per-AC findings

### AC1 — AIFS/WPS ingest manifest — **PASS**

- `data/manifests/aifs_ingest_v0.json` validates against `AIFSIngestManifest` (test `test_aifs_ingest_manifest_validates_and_cites_existing_gen2_paths` enforces every artifact path exists; passed).
- Strategy `reuse_gen2_wps_v0` matches M7 plan §331-333 directive (no new regridder).
- Cites real Gen2 WPS case (`20260520_18z_72h`) with real `namelist.wps`, real metgrid files, real Vtable; projection (Lambert, ref 28.3/-16.4, truelat 25/30) matches Canary domain. d03/d04/d05 marked `CONDITIONAL_M7_S2` (L63/70/77) — gated correctly on 1 km nest risk audit.
- `completeness_gate` (L106-124) defines concrete preflight checks for v0 minimum.

### AC2 — Station observation source manifest — **PASS**

- `data/manifests/station_obs_sources_v0.json` validates; AEMET + GRAFCAN both PARTIAL (verified above); METAR AVAILABLE with sample-fetch evidence committed.
- `binding_policy.operational_claim_rule` (L103) explicitly forbids operational validation claims without fresh observations.
- Gen2 station cube (193 stations, 6.94M rows, 2016–2026) registered as `LOCAL_HISTORICAL_STATION_CORPUS` — honest scope.

### AC3 — Output/status schemas — **PASS**

- `OperationalOutput`, `OperationalStatus`, `OperationalScheduler` added to `src/gpuwrf/io/proof_schemas.py:299-366`.
- Plus a bonus pair: `AIFSIngestManifest:255-276`, `StationObservationSourceManifest:279-296` — backs AC1/AC2 with same validation surface.
- Pattern conforms to existing `ProofObjectSchema` base (L27-65); registry entries added L457-470; `__all__` updated L495-515.
- Tests cover positive validation + negative type rejection (e.g. `test_operational_status_rejects_wrong_alert_type` L94-109, `test_operational_scheduler_rejects_string_cycle_hour` L130-146).
- **No accidental fork** — schemas slot into the existing dataclass-backed registry verbatim.

### AC4 — Gen2 corpus backfill plan — **PASS-WITH-MINOR**

- `gen2_corpus_backfill_plan.md` records: 3 complete + 22 partial → target 10 (margin to 14). Wall estimates 7-14 days live or 1-3 operator-days rerun. Blockers concrete (retention policy, AIFS late, disk).
- **Minor**: Owner is "Canairy Gen2 operator/team" (L22) — organizational not named. Acceptable for a manifest at this stage; the user-memory note `[[project_canairy_meteo_baseline]]` does not pin a single name either, and this is a coordination ask that the human arbiter must route. Worker correctly bounded their write scope.

### AC5 — M7-S0 sprint contract draft (BLOCKED handling) — **PASS** ⭐ load-bearing

- `.agent/sprints/2026-05-22-m7-s0/sprint-contract.md:5` status: `**DRAFT - BLOCKED-on-M6.x**`.
- AC3 (L16) consumes `gpuwrf.validation.data_quality.compute_rmse_against_gen2(gpu_forecast_state, gen2_wrfout_path, valid_time, fields=("U10", "V10", "T2"))` — exact M6.5-D1 adapter signature, no reimplementation.
- AC1 (L14) defines `m6_inheritance_gate.json` BLOCKED-emission if M6.x missing/failed.
- AC2 (L15) emits `BLOCKED_CORPUS` if <10 pinned members.
- AC8 (L21) separates model-consistency from operational station verification.
- Hard Rule #4 (L61) explicitly disallows operational validation language.
- **No model-validity claim found.** Dispatch condition (L6) and AC1-2 cleanly gate on missing evidence.

### AC6 — M6-S8 rename in-place — **PASS** ⭐ load-bearing

- `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:1` title: `# Sprint Contract — M6-S8 Model-Consistency Closeout`.
- Sprint ID (L3) updated to `2026-05-22-m6-s8-model-consistency-closeout`.
- Objective (L13): "This sprint does not make an operational verification claim."
- AC1 (L17) carries explicit verbatim disclaimer: "**NOT an operational verification claim — operational binds to station observations per M7-S5.** Document threshold rationale and label AIFS as a comparison denominator, not deterministic truth."
- Hard Rule #5 (L52) reserves operational verification for M7-S5 station-obs scope.
- **One residual**: directory path is still `.agent/sprints/2026-05-22-m6-s8-operational-closeout/`. Worker correctly disclosed this as in-place edit (sprint contract L28 permitted file scope was the file path, not the directory). Acceptable — directory-level renames create commit churn and break links; in-content disclaimer is the binding semantic fix.

### AC7 — 1 km nest risk audit — **PASS**

- `1km_nest_risk_audit.md` has real cells×bytes math: per-domain cell counts (table L20-25, sum d01+d02+d03+d04+d05 = 1,432,530 cells) and persistent-state bytes per domain (L29-37, peak 116 MiB for full nest).
- Cites concrete prior evidence: M6 spacetime `per_kernel.rrtmg.memory_analysis.temporary_bytes = 13221287152` (L12) — **independently verified**: `grep` in `artifacts/m6/spacetime_budget_d02.json` returns `temporary_bytes": 13221287152` exactly.
- M6-S6 OOM cited with file:line.
- Risk classification d03-only `HIGH`, d03+d04+d05 `VERY_HIGH`, 3 km-only `LOWER_RISK` — concrete, decision-relevant.
- Terrain/static correctness surfaced as gate (L13, L54) per RISK_REGISTER.md:15.
- **Not hand-wavy.**

### AC8 — Critic findings tracker — **PASS-WITH-MINOR** ⭐ load-bearing

- `critic_findings_tracker.md` extracts **23 findings** (PC-1..PC-23) from `critique-report.md` — more granular than the 9 in manager-response.md committed at `3c76d38`. Cross-ref:
  - Manager PC-1 (M7-S0a parallel) ↔ Tracker PC-1 ✓
  - Manager PC-2 (M6-S8 mislabel) ↔ Tracker PC-2 ✓
  - Manager PC-3 (ADR-016) ↔ Tracker PC-12 ✓
  - Manager PC-4/PC-9 (contingency c2/c3) ↔ Tracker PC-15 ✓
  - Manager PC-5 (M6.x kill-gate) ↔ Tracker PC-3 ✓
  - Manager PC-6 (M6.5-D1 misnamed) ↔ Tracker PC-8 + PC-22 ✓
  - Manager PC-7 (throughput language) ↔ Tracker PC-20 ✓
  - Manager PC-8 (F-5 denominator) ↔ Tracker PC-4 ✓
- **All 9 manager-response findings are captured.** Worker extracted independently from `critique-report.md` (not the manager-response), which is the correct primary source per sprint-contract AC8 wording.
- **Minor**: Tracker PC-12 disposition reads "DEFER … current M6.5-D1 manager closeout says ADR amendment was applied" — slightly understated; the code default 0.01 → TOLERANCES amendment per manager-response PC-3 is still a M7-S0a-adjacent follow-up. The tracker still defers it to the M6.5-D1 manager which is correct file-ownership-wise.

---

## Adversarial probes

1. **Schema fork?** No. New schemas use `ProofObjectSchema` base, same `FieldRule` dataclass, same JSON-schema emission, same registry pattern. Two new files (`AIFSIngestManifest`, `StationObservationSourceManifest`) follow the same conventions and are added to `SCHEMA_REGISTRY` + `__all__`.
2. **1 km audit math?** Real. Cell counts derived from M7 plan dimensions; persistent-state bytes derived from current `State` leaf set + ADR-007 precision matrix; RRTMG 13.22 GB temporary independently verified in `artifacts/m6/spacetime_budget_d02.json`.
3. **Gen2 backfill owner?** Organizational ("Canairy Gen2 operator/team") not named. Acceptable — user-memory `[[project_canairy_meteo_baseline]]` does not pin a single individual, and this is a human-arbiter routing decision. Worker did not over-claim a named contact.
4. **`data/manifests/` force-added past .gitignore?** Real minor hygiene risk. `.gitignore:62-63` lists `data` / `data/` because canonical setup has `data` as a symlink to `/mnt/data/wrf_gpu2`. In this worktree, `data/` exists as a real directory and the two JSON manifests are tracked (verified `git ls-files data/manifests/` returns both). Future *edits* to the manifests work fine (Edit on tracked files). Future *new* manifests under `data/` would need `git add -f` again. **Recommendation (non-blocking)**: a follow-up commit could add `!data/manifests/` exception to `.gitignore`. Worker correctly disclosed the force-add in worker-report L37.
5. **File-disjointness from M6.x?** Verified via `git diff e0fa6f9..d4a5e5b --name-only`: all 10 changed paths are within sprint-contract "Files Worker May Modify" list. No dynamics/coupling/physics/state/M6.5-D1 loader files touched.

---

## Binding decision: **ACCEPT-WITH-MINOR**

All 8 ACs pass on substance. The three minor follow-ups are non-blocking:

- **F-S0a-1** (hygiene): add `!data/manifests/` exception to `.gitignore` so future contributors don't need `-f`.
- **F-S0a-2** (cosmetic): consider directory rename `m6-s8-operational-closeout/` → `m6-s8-model-consistency-closeout/` if/when M6-S8 dispatches (creates churn now without semantic gain; in-content disclaimer is binding).
- **F-S0a-3** (clarity): tracker PC-12 disposition could note that code default `0.01 → TOLERANCES` (per manager-response PC-3) remains an open follow-up tied to ADR-016. Currently file-ownership-correct but slightly understated.

---

## M7-S0 dispatch impact

**Definitely UNBLOCKED on M6.x close + M6.5-D1 + F-5.** M7-S0a delivers exactly the scaffolding the M7 plan §64-72 expected to be in place before M7-S0 dispatch:

- `compute_rmse_against_gen2` consumer signature pinned in M7-S0 AC3.
- `m6_inheritance_gate.json`, `gen2_baseline_inventory.json`, `tier4_member_split.json`, `tier4_rmse_harness.json` artifact schemas implied by M7-S0 AC1/2/4/5 ready.
- AIFS + station-obs manifests live and validated; M7-S0 AC6 reads them as readiness context.
- BLOCKED-emission paths cleanly defined for both `M6.x missing` and `BLOCKED_CORPUS (<10 members)`.

The remaining gates are dycore-side (M6.x GREEN or c1 pivot) and data-side (Gen2 owner retention decision per `gen2_corpus_backfill_plan.md:68-70`). Neither is a M7-S0a follow-up.

**Net result**: M7-S0a saves 12-18h of serialization waste exactly as the plan critic predicted. ACCEPT-WITH-MINOR.

---

## Files inspected

- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/{sprint-contract,worker-report,role-prompts/reviewer,1km_nest_risk_audit,gen2_corpus_backfill_plan,critic_findings_tracker}.md`
- `.agent/sprints/2026-05-22-m7-s0/sprint-contract.md`
- `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md`
- `.agent/sprints/2026-05-22-plan-critic/critique-report.md` + manager-response.md (via `git show 3c76d38`)
- `src/gpuwrf/io/proof_schemas.py`, `tests/test_m7_s0a_schemas.py`
- `data/manifests/aifs_ingest_v0.json`, `data/manifests/station_obs_sources_v0.json`
- `artifacts/m6/spacetime_budget_d02.json` (spot-check for RRTMG temporary_bytes)
- `.gitignore` (force-add hygiene check)

## Commands run

- `pytest -q tests/test_m7_s0a_schemas.py`
- `pytest -q tests/test_m6_proof_schemas.py tests/test_m7_s0a_schemas.py`
- `ls -la <Gen2 paths>` (×9 paths)
- `curl` METAR (HTTP 200, live JSON), AEMET (HTTP 200, empty), GRAFCAN (HTTP 200, Swagger HTML)
- `git ls-files data/manifests/`, `git check-ignore data/manifests/aifs_ingest_v0.json`
- `git diff e0fa6f9..d4a5e5b --name-only`
- `grep 'temporary_bytes' artifacts/m6/spacetime_budget_d02.json`
- `git show 3c76d38:.agent/sprints/2026-05-22-plan-critic/manager-response.md`
