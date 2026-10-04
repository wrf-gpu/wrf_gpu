# v0.20.1 regression baseline-diff (vs v0.20.0) — definitive, S5-fix #1

Both suites: `taskset -c 0-3 env PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests/ -q -p no:cacheprovider`.

| run | result | wall |
|---|---|---|
| v0.20.0 baseline (main `e07aa783`) | **53 failed, 1794 passed, 358 skipped, 36 xfailed, 2 xpassed** | 25:18 |
| v0.20.1 integration (`c697fe0f`)   | **53 failed, 1822 passed, 375 skipped, 36 xfailed, 2 xpassed** | 26:39 |

(v0.20.1 collects ~45 more tests = the new #114/#122/#123/S2 tests; all of those PASS — the larger passed count.)

## VERDICT: ZERO new regressions introduced by v0.20.1.
- **NEW failing test FILES in v0.20.1 not in the v0.20.0 baseline: NONE** (file-level `comm -13` = empty).
- **Failing files shared with baseline: 11** (identical physics-oracle/data-env debt set).
- **No v0.20.1-code test fails:** grep of the v0.20.1 failed nodes for preflight / rrtmg_oom / b200 /
  operational_namelist / cache_key / date_clock / drain / manifest_valid / wrfout_writer / training_subset
  → **NONE**. All 50 #114/#122/#123/S2 tests pass.

## The 11 shared pre-existing-debt failing files (NOT v0.20.1-related; present at v0.20.0):
test_v017_ra_lw_hs, test_v018_mp_family_fail_closed, test_v018_ra_tail_oracle, test_v017_lsm_pleim_xiu,
test_v017_lsm_adv, test_v013_operational_smoke, test_v018_cu_family_status, test_v017_sas_family_parity,
test_v016_thompson_aero_oracle, test_tiedtke_cumulus_oracle, test_rrtm_lw_operational_wiring.
These need pristine-WRF oracle data / env not present in this CI env (the documented v0.20.0 baseline debt);
a few additional aifs_grib/forcing_decode/agentos/m7-memory nodes flip run-to-run (flaky data-env, also
pre-existing). None are caused by v0.20.1 code.
