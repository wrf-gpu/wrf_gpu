# v0.18 MP-family — Opus adversarial critic report

- **Branch / HEAD:** `worker/gpt/v018-mp` @ `48ee2d9e` (confirmed)
- **Worktree:** `<USER_HOME>/src/wrf_gpu2/.wt-v018-mp`
- **Date:** 2026-06-16
- **Critic:** Opus (adversarial verification of frontrunner `full_ship_gate=true`)
- **Pristine WRF arbiter:** `<USER_HOME>/src/wrf_pristine/WRF`

## Overall verdict: **ACCEPT**

The 0.18 ship-gate is satisfied. Every mp option is exactly one of operational+green,
ref-with-real-exact-module-oracle (fail-closed), or proven-irrelevant (source-cited).
No silent gaps, no still-open, no fake-green, no tolerance widening detected.
Two cosmetic doc-nits (below), neither a gate violation.

## Per-class verdict

### Operational (mp 0,1,2,3,4,6,8,10,13,14,16,24,26,28,97) — PASS
- `ACCEPTED_MP_PHYSICS` (`src/gpuwrf/contracts/physics_registry.py:80`) and
  `_SCAN_WIRED_OPTIONS["mp_physics"]` (`src/gpuwrf/runtime/operational_mode.py:3153`)
  both == the reported operational set exactly. No accidental promotion of a
  ref/irrelevant option into the operational set.
- Genuine run+mutate spot-checks PASS (savepoint parity, not just "registered"):
  SBU-YLin mp13, WSM7 mp24, WDM7 mp26, Goddard mp97 all green
  (`tests/test_{sbu_ylin,wsm7,wdm7,goddard}_savepoint_parity.py`, 232 passed in the batch).

### Ref-with-oracle (mp 5,7,9,18,27,29,38,40,50,51,52,53,56,95) — PASS
- **Exact-module mapping (#1): all 14 correct vs pristine driver** (independently
  traced through `phys/module_microphysics_driver.F` SELECT CASE + Registry packages +
  `frame/module_state_description.F`). The critical mp5 vs mp95 distinction is correct
  and enforced:
  - mp5 → `module_mp_fer_hires.F:FER_HIRES` (driver CASE FER_MP_HIRES); oracle driver
    `use module_mp_fer_hires, only: ..., FER_HIRES`; `call FER_HIRES`.
  - mp95 → `module_mp_etanew.F:ETAMP_NEW` (driver CASE ETAMPNEW); separate artifact
    root; checksums contain `module_mp_etanew.F` and NOT `module_mp_fer_hires.F`
    (`tests/test_v018_mp_family_fail_closed.py:296-300`).
  - mp38 (THOMPSONGH) shares `module_mp_thompson.F` with mp8/28 but the oracle driver
    specifically trips the `is_hail_aware` variable-density-graupel path (passes `ng`
    PRESENT → `qr_acr_qg_mp38V1` tables), so it exercises the GH code path, not the
    operational mp8 path. Acceptable exact-module exercise.
  - P3 50/51/52/53 all correctly share `module_mp_p3.F` but are **distinct WRF runs**
    (distinct `history_netcdf` sha256, distinct nontrivial-field counts 7/8/10/9).
- **Nontrivial (#2): PASS.** 10 full-model oracles all have distinct file md5 AND
  distinct run-output sha256, with 7–23 nontrivial seeded-field deltas each (mp9
  inspected in full: rich Milbrandt 2-moment column, hail mass→0 physically consistent,
  distinct number concentrations). The extract script (`extract_active_oracles.py:239-240`,
  `:221-222`) RAISES if `wrf_success` is false or if there is no nontrivial delta — the
  "all trial columns null" failure mode (the CU-family nit) is structurally impossible here.
- **Re-buildable (#3): PASS.** Standalone oracles (5,7,38,95) ship committed
  `build_*.sh` + `*_oracle_driver.f90` + stub modules + source checksums of the exact
  pristine module. Full-model oracles (9,18,27,29,40,50–53,56) ship the committed
  `extract_active_oracles.py` plus full `run_artifact_hashes` (wrf.exe, wrfinput,
  namelist, stdout, history). NetCDF in /tmp is the deliberate non-committed input; the
  JSON proof object is committed. No un-reproducible one-off found.

### Proven-irrelevant (mp 11,17,19,21,22,30,32,55,96) — PASS
- Source-citation spot-checks (#6) confirmed against pristine WRF:
  - mp96 MadWRF: driver CASE body is literally only `CALL wrf_debug(100,...)`
    (`module_microphysics_driver.F:3090-3091`); `test_madwrf_mp96_driver_case_is_noop`
    asserts the exact normalized body.
  - mp30 fast SBM: `#if( BUILD_SBM_FAST != 1)` compile gate (`module_mp_fast_sbm.F:1`).
  - mp55 Jensen-ISHMAEL: external `ishmael-*.bin` tables
    (`module_mp_jensen_ishmael.F:119,138,161`).
  - mp17/19/21/22 (legacy NSSL): `doc/README.NSSLmp:24-37` explicitly states these are
    legacy options that should use mp=18 with namelist modifier flags, giving exact
    equivalences. Classification as duplicates-of-mp18 is exactly what WRF's own README
    says. (See NIT-1 on the `exact_module` field for these.)

### Fail-closed wiring (#4) — PASS (the most important correctness check)
- Two-layer fail-closed authority, no silent fallthrough:
  1. `resolve_physics_suite` → `scheme_entry` (`coupling/physics_dispatch.py:467-479`)
     raises `UnsupportedSchemeSelection` for any mp not in `ACCEPTED_MP_PHYSICS`,
     listing accepted options. Since every ref/irrelevant mp is absent from
     `ACCEPTED_MP_PHYSICS`, selecting mp=9/50/56/etc. raises IMMEDIATELY — it does NOT
     silently run another scheme's physics.
  2. `_resolve_operational_suite` (`operational_mode.py:3256-3339`) re-checks against
     `_SCAN_WIRED_OPTIONS` and raises with a scheme-specific reason.
  - Namelist layer also rejects (`validate_namelist`,
    `tests/test_v018_mp_family_fail_closed.py:137-147`).
- `tests/test_v018_mp_family_fail_closed.py` (34 tests) passes, including
  `test_v018_open_mp_family_fails_closed_with_named_reason` and
  `..._rejected_at_namelist_layer`.

### Set-union integrity (#7) — PASS
- `tests/test_v018_conditional_state_leaves.py` (passed) proves the #37 conditional
  allocation is intact: mp=8 → 60-leaf base only (no hail/aero leaves); mp=24/26 →
  base + HAIL lanes only (no aerosol); mp=28 → base + AEROSOL lanes only (no hail).
  Static mp_physics gates the leaves, so non-hail/non-aero programs are byte-unchanged.

## Gates re-run (CPU, cores 0-3, JAX_PLATFORMS=cpu, PYTHONPATH=src)
- `tests/test_v018_mp_family_fail_closed.py` — **34 passed**
- conditional-state + apply-physics-non-dry + catalog-fail-closed + namelist-check +
  4 savepoint-parity suites + physics-dispatch — **232 passed**
- m3-state + qh-hail-state + physics-interfaces contracts — **17 passed, 3 skipped**
- precision-matrix + restart-roundtrip + restart-full-carry + wrfrst-netcdf +
  thompson-aero oracle + thompson-aero threading — **29 passed, 1 skipped**
- Total: **312 passed, 4 skipped, 0 failed.**
- GPU smoke: NOT re-run — GPU lock is currently HELD
  (`/tmp/wrf_gpu2_gpu.lock.holder` present, dated today). Per instructions, did not
  contend. This MP-tail sprint adds NO new operational kernel (only reference-only /
  proven-irrelevant endpoints), so the operational GPU smoke is unaffected by it; the
  operational MP set was already GPU-green in v0.16/v0.17.

## Findings (both NIT severity; neither blocks ship)
- **NIT-1 (manifest cosmetic):** mp 17/19/21/22 carry `exact_module:
  phys/module_mp_nssl_2mom.F`. Independent driver tracing shows these four integers
  have NO Registry package and NO driver CASE in this pristine build (they fall to
  CASE DEFAULT). The manifest's *irrelevance_basis* is nonetheless correct (README.NSSLmp
  maps them onto mp=18 with flags). Since the proven-irrelevant class requires only a
  documented WRF-source citation — which is provided and accurate — this is not a gate
  violation; the `exact_module` is a "would-be module" pointer, not an oracle claim.
  Suggest a one-word note ("legacy alias of mp18; not a distinct driver CASE") for clarity.
- **NIT-2 (stale error string):** the `UnsupportedSchemeSelection` message in
  `_resolve_operational_suite` (`operational_mode.py:3331`) lists
  `mp_physics in {0,1,2,3,4,6,8,10,13,14,16,28,97}` — it omits 24 and 26, which ARE
  scan-wired (line 3153). Cosmetic only; the gate logic uses the tuple, not the string.

## Things I tried to break and could not
- Reuse of one oracle across distinct mp options: ruled out — distinct md5 AND distinct
  run sha256 for all 10 full-model oracles; mp5/mp95 have separate artifact roots with
  the test explicitly forbidding cross-contamination of checksums.
- mp5/mp95 Ferrier confusion: ruled out — correct module each, enforced by test.
- Silent fallthrough to a sibling scheme on a ref-only selection: ruled out — two-layer
  raise on absence from `ACCEPTED_MP_PHYSICS`.
- All-null / empty oracle (the CU-family failure mode): structurally impossible — the
  extract script raises on no-nontrivial-delta / wrf-failure.
