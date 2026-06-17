# v0.18 CU family step-1 report

- Step-1 honesty gate: `True`
- Full v0.18 CU ship gate: `True`
- Operational CU scan preserved: `[0, 1, 2, 3, 6]`
- Tail CU7/10/11 accepted without oracle: `[]`
- Tail CU7/10/11 relevance: `RELEVANT_NOT_PROVEN_IRRELEVANT`

| CU | Scheme | Status | Oracle | Blocker |
|---:|---|---|---|---|
| 0 | disabled | OPERATIONAL_GREEN_BASELINE | existing operational proof lane |  |
| 1 | Kain-Fritsch | OPERATIONAL_GREEN_BASELINE | existing operational proof lane |  |
| 2 | Betts-Miller-Janjic | OPERATIONAL_GREEN_BASELINE | existing operational proof lane |  |
| 3 | Grell-Freitas | OPERATIONAL_GREEN_BASELINE | existing operational proof lane |  |
| 4 | Scale-aware GFS SAS | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_savepoints_present | shared SAS JAX endpoint is RED vs pristine-WRF oracle; not scan-wired |
| 5 | Grell-3D ensemble | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_nontrivial | faithful source-specific JAX operational endpoint not ported in this run; not scan-wired |
| 6 | Tiedtke | OPERATIONAL_GREEN_BASELINE | existing operational proof lane |  |
| 7 | Zhang-McFarlane CAMZM | REFERENCE_ONLY_WITH_REAL_ORACLE | real_pristine_wrf_completed_diagnostic_only | source-specific JAX operational endpoint not ported in this run; not scan-wired |
| 10 | KF-CuP | REFERENCE_ONLY_WITH_REAL_ORACLE | real_pristine_wrf_nontrivial | source-specific JAX operational endpoint not ported in this run; not scan-wired |
| 11 | MSKF | REFERENCE_ONLY_WITH_REAL_ORACLE | real_pristine_wrf_nontrivial | source-specific JAX operational endpoint not ported in this run; not scan-wired |
| 93 | Grell-Devenyi ensemble | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_nontrivial | faithful source-specific JAX operational endpoint not ported in this run; not scan-wired |
| 94 | 2015 GFS SAS / HWRF | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_savepoints_present | shared SAS JAX endpoint is RED vs pristine-WRF oracle; not scan-wired |
| 95 | Previous GFS SAS / HWRF OSAS | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_savepoints_present | shared SAS JAX endpoint is RED vs pristine-WRF oracle; not scan-wired |
| 96 | Previous new GFS SAS / YSU NSAS | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_savepoints_present | shared SAS JAX endpoint is RED vs pristine-WRF oracle; not scan-wired |
| 99 | previous Kain-Fritsch | REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX | real_pristine_wrf_savepoints_present | candidate reuses KF-eta family and is RED vs module_cu_kf.F old-KF oracle; not scan-wired |

Proof commands refreshed in this run:

- `taskset -c 0-3 proofs/v017/oracle/cumulus_sas/build_and_run.sh`
- `taskset -c 0-3 proofs/v017/oracle/cumulus/build_oldkf_oracle.sh`
- `taskset -c 0-3 bash proofs/v018/oracle/cumulus_grell/build_and_run.sh`
- `taskset -c 0-3 bash proofs/v018/oracle/cumulus_tail_wrf/build_and_run.sh`
- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 proofs/v017/run_sas_family_parity.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 proofs/v017/run_cu_kfgrell_parity.py --build-oldkf --allow-red`
- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 proofs/v018/cu_family_status.py`

Tail CU7/10/11 were treated as relevant rather than proven irrelevant; each now has a completed pristine-WRF reference savepoint. CU7 is diagnostic-only in this fixture (base/top fields nonzero, heating/moistening zero) and remains fail-closed reference-only.

No GPU operational smoke was run for CU5/7/10/11/93/94/95/96/99 because none of those RED/reference-only schemes were scan-wired.
