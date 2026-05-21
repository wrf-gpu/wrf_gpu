# Worker Report - M7-S0a Operational/Data Readiness Prologue

## Objective

Execute M7-S0a file-disjoint from the active M6.x dycore worker. Build operational/data readiness scaffolding that does not depend on dycore implementation: AIFS/WPS ingest manifest, station observation source manifest, output/status/scheduler schemas, Gen2 corpus backfill plan, blocked M7-S0 draft contract, M6-S8 terminology correction, 1 km nest risk audit, and plan-critic finding tracker.

## Outcome

Status: **PASS for M7-S0a scaffolding; BLOCKED only on M6-dependent downstream claims**.

No dynamics, physics, coupling driver, state contract, M6.5-D1 loader, M6.5-D1 data-quality code, or `/mnt/data/canairy_meteo/**` files were modified.

## Acceptance Mapping

- AC1 AIFS/WPS ingest manifest: complete. `data/manifests/aifs_ingest_v0.json` validates against `AIFSIngestManifest` and cites real Gen2 WPS/AIFS paths. The required max-depth-3 inventory found only `/mnt/data/canairy_meteo/AIFS.log` and `/mnt/data/canairy_meteo/data/aifs_ens/aifs_ens_20260414_00z.grib2`; a targeted read-only search found the actual deeper WPS case under `/mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/`.
- AC2 Station observation source manifest: complete. `data/manifests/station_obs_sources_v0.json` validates against `StationObservationSourceManifest`. METAR is marked AVAILABLE with live AviationWeather/NOAA sample evidence. AEMET and GRAFCAN live paths are marked PARTIAL because credentials/terms are not pinned, while their local Gen2 manifests are available. The Gen2 station cube is marked AVAILABLE as a local historical corpus.
- AC3 Output/status schemas: complete. Added `OperationalOutput`, `OperationalStatus`, and `OperationalScheduler` to `src/gpuwrf/io/proof_schemas.py`, plus registry entries and tests. I also added `AIFSIngestManifest` and `StationObservationSourceManifest` so AC1/AC2 have the same validation surface.
- AC4 Gen2 corpus backfill plan: complete. `gen2_corpus_backfill_plan.md` records the 3 complete + 22 partial state, target 10+ complete pinned-grid d02 members, owner coordination, wall estimate, blockers, and proof objects for the follow-up.
- AC5 M7-S0 sprint contract draft: complete. `.agent/sprints/2026-05-22-m7-s0/sprint-contract.md` is ready-to-dispatch but explicitly `DRAFT - BLOCKED-on-M6.x`. It consumes `compute_rmse_against_gen2` and rejects operational station-verification language.
- AC6 M6-S8 rename: complete in place. The existing M6-S8 contract now says model-consistency closeout, amends AC1 to disclaim operational verification, and reserves operational validation for station observations per M7-S5.
- AC7 1 km nest risk audit: complete. `1km_nest_risk_audit.md` estimates current persistent state bytes by domain and highlights the real risk: XLA temporaries/compile buffers, not raw state. It cites the M6-S6 OOM diagnosis and terrain/static correctness risk.
- AC8 Critic findings catalog: complete. `critic_findings_tracker.md` extracts actionable plan-critic findings with severity, citation, disposition, and owner.

## Files Changed

- `src/gpuwrf/io/proof_schemas.py`
- `tests/test_m7_s0a_schemas.py`
- `data/manifests/aifs_ingest_v0.json`
- `data/manifests/station_obs_sources_v0.json`
- `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md`
- `.agent/sprints/2026-05-22-m7-s0/sprint-contract.md`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/gen2_corpus_backfill_plan.md`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/1km_nest_risk_audit.md`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/critic_findings_tracker.md`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/worker-report.md`

Note: `data/` is ignored by `.gitignore` because it is normally external storage. The two small required manifest JSON files must be force-added with `git add -f data/manifests/aifs_ingest_v0.json data/manifests/station_obs_sources_v0.json`.

## Commands Run

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
- `sed -n '1,240p' AGENTS.md`
- `sed -n '1,260p' .agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/sprint-contract.md`
- `sed -n '1,260p' .agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/role-prompts/worker.md`
- `sed -n '1,240p' .agent/skills/managing-sprints/SKILL.md`
- `sed -n '1,220p' .agent/skills/updating-docs-minimally/SKILL.md`
- `sed -n '1,220p' .agent/skills/reporting-to-human/SKILL.md`
- `nl -ba .agent/sprints/2026-05-22-plan-critic/critique-report.md | sed -n '1,260p'`
- `nl -ba .agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md | sed -n '1,540p'`
- `nl -ba .agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md | sed -n '1,260p'`
- `nl -ba .agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/manager-closeout.md | sed -n '1,320p'`
- `nl -ba .agent/sprints/2026-05-21-m6-s7-tier4-probtest/reviewer-report.md | sed -n '1,340p'`
- `nl -ba RISK_REGISTER.md | sed -n '1,120p'`
- `find /mnt/data/canairy_meteo/ -maxdepth 3 \( -name 'namelist.wps*' -o -name 'Vtable*' -o -name 'ungrib*' -o -name 'AIFS*' -o -name '*.grib2' \) -print | sort | head -200`
- `find /mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h -maxdepth 2 -type f | sort | head -120`
- `sed -n '1,220p' /mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/l3/namelist.wps`
- `sed -n '1,220p' /mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/ungrib/namelist.wps`
- `ncdump -h /mnt/data/canairy_meteo/runs/wps_cases/20260520_18z_72h/l3/met_em.d02.2026-05-20_18:00:00.nc`
- `curl -L --max-time 20 --silent --show-error --fail 'https://aviationweather.gov/api/data/metar?ids=GCRR,GCLP,GCTS,GCFV,GCXO,GCHI,GCLA,GCGM,GCRT&format=json' | head -c 1200`
- `curl -L --max-time 20 --silent --show-error --fail 'https://tgftp.nws.noaa.gov/data/observations/metar/stations/GCRR.TXT' | head -c 500`
- `curl -L --max-time 20 --silent --show-error --head 'https://opendata.sitcan.es/upload/meteorologia/estaciones.csv'`
- `curl -L --max-time 20 --silent --show-error 'https://sensores.grafcan.es/api/v1.0/' | head -c 500`
- `curl -L --max-time 20 --silent --show-error --include 'https://opendata.aemet.es/opendata/api/observacion/convencional/todas' | head -c 1200`
- `pytest -q tests/test_m7_s0a_schemas.py` -> `8 passed in 0.20s`
- `python -m json.tool data/manifests/aifs_ingest_v0.json >/dev/null`
- `python -m json.tool data/manifests/station_obs_sources_v0.json >/dev/null`
- `python -m py_compile src/gpuwrf/io/proof_schemas.py tests/test_m7_s0a_schemas.py`
- `pytest -q tests/test_m6_proof_schemas.py tests/test_m7_s0a_schemas.py` -> `11 passed in 0.19s`

## Proof Objects Produced

- `data/manifests/aifs_ingest_v0.json`
- `data/manifests/station_obs_sources_v0.json`
- `tests/test_m7_s0a_schemas.py`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/gen2_corpus_backfill_plan.md`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/1km_nest_risk_audit.md`
- `.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/critic_findings_tracker.md`
- `.agent/sprints/2026-05-22-m7-s0/sprint-contract.md`

## Unresolved Risks

- M6.x dycore evidence is still required before M7-S0 can make any model-validity claim. The M7-S0 draft therefore stays BLOCKED-on-M6.x.
- The Gen2 corpus still needs 7+ additional complete pinned-grid d02 24 h members, or recent pinned-grid cycles must be rerun/retained by the Canairy Gen2 owner.
- AEMET and GRAFCAN live station observations need credential and license/terms handling before M7-S5 can use them as live operational truth.
- The 1 km nest memory risk is not resolved by this sprint. Persistent state is small, but compile/HBM temporaries can still exceed the RTX 5090 gate.
- Terrain/geog/static correctness remains a required M7-S2/S3 proof before any 1 km evidence is meaningful.

## Next Decision Needed

Manager/human should choose whether Canairy Gen2 retains the next 7+ daily d02 histories or reruns recent pinned-grid cycles to reach 10+ complete members faster. M6.x manager must also provide a green/red dycore decision before M7-S0 is dispatched beyond BLOCKED preflight.
