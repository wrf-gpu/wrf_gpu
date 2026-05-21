# Plan Critic Findings Tracker

Source: `.agent/sprints/2026-05-22-plan-critic/critique-report.md`.

| ID | Severity | Finding | Citation | Disposition | Owner |
|---|---|---|---|---|---|
| PC-1 | CRITICAL | Stop serializing file-disjoint operational/data work behind M6.x; run M7-S0a now. | critique-report.md:9, :67-69, :88 | ACTED-ON in this sprint. | M7-S0a worker |
| PC-2 | MAJOR | M6-S8 must not be called operational without station-observation verification. | critique-report.md:15-17, :89 | ACTED-ON by amending M6-S8 contract in place. | M7-S0a worker |
| PC-3 | CRITICAL | Treat M6.x as a fast green/red decision gate; pivot to c1 if it fails rather than stretching. | critique-report.md:24-25, :63-65, :91 | DEFER-TO-M6.x manager; outside M7-S0a write scope. | M6.x manager |
| PC-4 | MAJOR | M6-S8 pre-dispatch must explicitly require F-5 denominator acceptance. | critique-report.md:27 | ALREADY-IN-CONTRACT; preserved in M6-S8 checklist. | M6-S8 manager |
| PC-5 | MAJOR | M6-S7 scaffold is useful but the 10-member corpus premise failed; do inventory before more Tier-4 code. | critique-report.md:31 | ACTED-ON via Gen2 corpus backfill plan. | M7 manager / Canairy Gen2 |
| PC-6 | MAJOR | M6-S6 d02 drift must be rerun under M6.x uncapped dynamics. | critique-report.md:33, :79 | DEFER-TO-M6-S6-followup/M6-S8. | M6-S8 worker |
| PC-7 | MAJOR | M6-S4 invariant PASS language should be restrained because some closures were bookkeeping or cap-dependent. | critique-report.md:35 | DEFER-TO-M6 closeout report; outside allowed files. | M6 closeout manager |
| PC-8 | MAJOR | M6.5-D1 built loader/audit/RMSE adapter but did not backfill the missing corpus. | critique-report.md:37 | ACTED-ON via Gen2 corpus backfill plan. | M7 manager / Canairy Gen2 |
| PC-9 | CRITICAL | Live AIFS/WPS ingest is still plan text and should reuse Gen2 WPS rather than inventing a regridder. | critique-report.md:43 | ACTED-ON via `data/manifests/aifs_ingest_v0.json`. | M7-S0a worker |
| PC-10 | CRITICAL | Observation verification is not staged; M7-S5 needs station-source manifest and operational scores. | critique-report.md:45 | ACTED-ON via `data/manifests/station_obs_sources_v0.json`. | M7-S0a worker; M7-S5 follow-up |
| PC-11 | MAJOR | Output, restart, monitoring, and scheduler contracts are late; skeleton schema ownership can start now. | critique-report.md:47 | PARTLY-ACTED-ON via OperationalOutput/Status/Scheduler schemas; restart remains M7-S6. | M7-S0a worker; M7-S6/S7 |
| PC-12 | MAJOR | ADR-016 threshold/status amendments were missing in critic worktree. | critique-report.md:49, :81 | DEFER; current M6.5-D1 manager closeout says ADR amendment was applied, but ADR/code files are outside this sprint ownership. | M6.5-D1 manager |
| PC-13 | CRITICAL | M4 dycore is a reduced proxy; M6.x canonical/c1 decision is necessary. | critique-report.md:53-55 | DEFER-TO-M6.x; M7-S0 draft marks BLOCKED-on-M6.x. | M6.x worker |
| PC-14 | MAJOR | If c1 clean-room path is used, acceptance must state it is not WRF-canonical dyn_em. | critique-report.md:55 | DEFER-TO-c1 contract/ADR if invoked. | M6.x contingency manager |
| PC-15 | MINOR | Contingency insurance skipped c2/c3 contracts and ADR-017 unless c1 is sole realistic option. | critique-report.md:57 | DEFER; not file-owned by M7-S0a. | M6.x manager |
| PC-16 | MAJOR | AIFS boundary quality and late/missing readiness must move up. | critique-report.md:59 | ACTED-ON via AIFS manifest completeness gates and status states. | M7-S0a worker; M7-S1 |
| PC-17 | MAJOR | Terrain/static correctness for 1 km nest must move up. | critique-report.md:59; RISK_REGISTER.md:15 | ACTED-ON via 1 km risk audit; proof remains M7-S2. | M7-S2 worker |
| PC-18 | MAJOR | RTX 5090 1 km memory/compile gate must move up. | critique-report.md:59 | ACTED-ON via 1 km risk audit; profiler proof remains M7-S2. | M7-S2 worker |
| PC-19 | MAJOR | Observation-source availability is not secondary to dycore for operational daily forecast. | critique-report.md:59; RISK_REGISTER.md:16 | ACTED-ON via station source manifest; live ingest remains M7-S5. | M7-S5 worker |
| PC-20 | MAJOR | M6-S5 throughput should stay provisional until final dycore and denominator are in the same artifact. | critique-report.md:77 | DEFER-TO-M6-S8/ADR-007; outside M7-S0a allowed files. | M6-S8 worker |
| PC-21 | MAJOR | M6-S8 must not close until uncapped d02 drift retry actually runs. | critique-report.md:79 | DEFER-TO-M6-S8 pre-dispatch and follow-up. | M6-S8 manager |
| PC-22 | MAJOR | M6.5-D1 should be labeled "loader ready, corpus incomplete." | critique-report.md:81 | ACTED-ON in M7-S0a backfill plan and M7-S0 draft blocker language. | M7-S0a worker |
| PC-23 | MAJOR | Do not add more review process; make next moves sharper. | critique-report.md:85-91 | ACTED-ON for M7-S0a; other parts deferred to managers. | M7 manager |

## Summary

This sprint directly acts on the file-disjoint M7 readiness findings: AIFS/WPS ingest, station-source manifest, schemas, corpus backfill plan, M7-S0 draft, M6-S8 terminology, and 1 km risk. M6.x dycore validity, c1 pivot policy, ADR amendments, and final M6 closeout claims remain outside this worker's allowed write set.
