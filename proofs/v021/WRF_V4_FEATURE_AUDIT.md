# WRF-v4 Feature Audit — wrf_gpu2 v0.21.0

**Auditor mode:** code-grounded, read-only. Every claim cites a file/symbol/test that was actually opened.
**Date:** 2026-06-25.
**Branches inspected:**
- Main checkout `<USER_HOME>/src/wrf_gpu2` @ `e8915772` (branch `worker/gpt/vnext-a1-finite-detector`).
- v0.21.0 candidate worktree `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile` @ `63032d24` (branch `worker/opus/vnext-parallel-compile`).

**Per-branch difference (verified):** `git diff --stat e8915772 63032d24 -- src/gpuwrf` shows the v0.21 candidate differs from the main checkout ONLY in `runtime/` (AOT cheap-key cache, parallel-compile, autotune) plus minor touches to `dynamics/acoustic_wrf.py`, `dynamics/core/small_step_prep.py`, `dynamics/mu_t_advance.py`, `integration/nested_pipeline.py`, `integration/d02_replay.py`. **NO physics scheme or coupling adapter changed.** Therefore the main checkout's `src/gpuwrf/physics`, `src/gpuwrf/coupling`, and `tests/` are representative of the v0.21.0 *feature set*. v0.21.0 is a stability + compile-cache release, not a new-feature release (confirmed by `proofs/v021/RELEASE_NOTES_v0210_DRAFT.md`: "STABILITY > IDENTITY > SPEED > MEMORY ... should not claim a new warm forecast speedup").

## Ground-truth authorities used

The "integrated" column is NOT inferred from the existence of a module file (many modules exist as reference-only ports). It is taken from the two machine-checked authorities the project itself designates:

1. **`src/gpuwrf/runtime/operational_mode.py:3459` `_SCAN_WIRED_OPTIONS`** — the exact namelist codes threaded into the operational GPU `jax.lax.scan`. Anything not listed here fail-closes at runtime with a named reason (`_SCAN_UNWIRED_REASON`, line 3505).
2. **`src/gpuwrf/coupling/scan_adapters.py:1412` adapter registries** — `MP_SCAN_ADAPTERS`, `PBL_SCAN_ADAPTERS`, `SFCLAY_SCAN_ADAPTERS`, `CU_SCAN_ADAPTERS`, `CU_STATELESS_SCAN_ADAPTERS`.
3. **`src/gpuwrf/io/scheme_catalog.py`** — the public honesty contract classifying every WRF-v4 option as `IMPLEMENTED` / `REFERENCE_ONLY` / `RECOGNIZED_FAIL_CLOSED` / `RECOGNIZED_APPROXIMATED` / `OUT_OF_SCOPE`, machine-checked against authorities (1) and (2) via `assert_catalog_consistent`.
4. **Default operational suite** = `src/gpuwrf/coupling/physics_dispatch.py:68-72`: `mp=8` Thompson, `bl_pbl=5` MYNN, `sf_sfclay=5` MYNN-SL, `cu=0` none, `sf_surface=4` Noah-MP; radiation default `ra_lw=4`/`ra_sw=4` RRTMG (`operational_mode.py:3493/3500`).

**`_SCAN_WIRED_OPTIONS` verbatim (operational_mode.py:3468-3500):**
```
mp_physics       : 0,1,2,3,4,6,8,10,13,14,16,24,26,28,97
bl_pbl_physics   : 0,1,2,3,5,7,8,11,12,99
sf_sfclay_physics: 0,1,2,3,5,7,91
cu_physics       : 0,1,2,3,6
ra_sw_physics    : 0,1,2,4
ra_lw_physics    : 0,1,4,31
```

**Status legend.** *integrated+tested* = scan-wired AND has a real WRF-savepoint/oracle/conservation test that runs (not skip-gated, not a self-compare). *integrated-undertested* = scan-wired but the only test is a smoke/structural/wiring test, OR the real-WRF parity test is `@pytest.mark.skipif`-gated on external fixtures that are not guaranteed present. *not-integrated* = not in `_SCAN_WIRED_OPTIONS` (fail-closed or out-of-scope), regardless of whether a reference module exists.

---

## FEATURE TABLE

### Microphysics

| Feature | Status | Code location (file:symbol) | Test (file + pytest cmd) | Notes |
|---|---|---|---|---|
| Thompson (mp=8, **default**) | integrated+tested | `physics/thompson_column.py`; wired via `_SCAN_WIRED_OPTIONS` mp=8 + existing Thompson couplers | `tests/test_m5_thompson_tier1.py`, `tests/test_thompson_precip_oracle.py` — `pytest tests/test_m5_thompson_tier1.py tests/test_thompson_precip_oracle.py` | Real savepoint parity vs `module_mp_thompson.F`. Default MP. v0.21 has a steep-terrain Thompson-Ni stability fix (RELEASE_NOTES draft "Ni fix slot"). |
| Thompson aerosol (mp=28) | integrated+tested | `physics/thompson_aero_column.py`; `_SCAN_WIRED_OPTIONS` mp=28 | `tests/test_v016_thompson_aero_oracle.py` — `pytest tests/test_v016_thompson_aero_oracle.py` | Savepoint parity vs pristine mp_physics=28 grid (v0.16). This is the "aerosols / MP28" checklist item. |
| WSM6 (mp=6) | integrated+tested | `physics/microphysics_wsm6.py`; `MP_SCAN_ADAPTERS[6]=wsm6_adapter` | `tests/test_wsm6_savepoint_parity.py` — `pytest tests/test_wsm6_savepoint_parity.py` | Savepoint parity vs `module_mp_wsm6.F`. |
| WSM3 / WSM5 (mp=3/4) | integrated+tested | `microphysics_wsm3.py`/`wsm5.py`; `MP_SCAN_ADAPTERS[3,4]` | `tests/test_wsm_sm_savepoint_parity.py` — `pytest tests/test_wsm_sm_savepoint_parity.py` | Savepoint parity vs WRF fp64 oracle. |
| WSM7 (mp=24) | integrated+tested | `microphysics_wsm7.py`; `MP_SCAN_ADAPTERS[24]=wsm7_adapter` | `tests/test_wsm7_savepoint_parity.py` — `pytest tests/test_wsm7_savepoint_parity.py` | Savepoint parity vs `module_mp_wsm7.F` (v0.17 hail). |
| WDM5 / WDM6 / WDM7 (mp=14/16/26) | integrated+tested | `microphysics_wdm5/6/7.py`; `MP_SCAN_ADAPTERS[14,16,26]` | `tests/test_wdm5_savepoint_parity.py`, `_wdm6_`, `_wdm7_` — `pytest tests/test_wdm6_savepoint_parity.py` | Savepoint parity vs WRF WDM modules. |
| Morrison 2-moment (mp=10) | integrated-undertested | `microphysics_morrison.py`; `MP_SCAN_ADAPTERS[10]=morrison_adapter` | `tests/savepoint/test_morrison_parity.py` — `pytest tests/savepoint/test_morrison_parity.py` | Real parity vs `module_mp_morr_two_moment.F`, but the whole file is `pytestmark = pytest.mark.skipif(...)` (line 23) on external fixture availability → may SKIP in CI. |
| SBU-YLin (mp=13) | integrated+tested | `microphysics_sbu_ylin.py`; `MP_SCAN_ADAPTERS[13]=sbu_ylin_adapter` | `tests/test_sbu_ylin_savepoint_parity.py` — `pytest tests/test_sbu_ylin_savepoint_parity.py` | Savepoint parity vs `module_mp_sbu_ylin.F` (v0.18). |
| Goddard GCE (mp=97) | integrated+tested | `microphysics_goddard.py`; `MP_SCAN_ADAPTERS[97]=goddard_adapter` | `tests/test_goddard_savepoint_parity.py` — `pytest tests/test_goddard_savepoint_parity.py` | Savepoint parity vs `module_mp_gsfcgce.F`. |
| Kessler (mp=1) | integrated+tested | `microphysics_kessler.py`; `MP_SCAN_ADAPTERS[1]=kessler_adapter` | `tests/test_kessler_microphysics.py` — `pytest tests/test_kessler_microphysics.py` | Savepoint parity (warm-rain). |
| Lin (mp=2) | integrated-undertested | `microphysics_lin.py`/`_lin_update.py`; `MP_SCAN_ADAPTERS[2]=lin_adapter` | No dedicated `test_lin_*` file found in `tests/` | Scan-wired (mp=2 in `_SCAN_WIRED_OPTIONS`) but no standalone Lin savepoint test located. Flag: relies on suite-level coverage only. |

### PBL (boundary layer)

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| MYNN (bl=5, **default**) | integrated+tested | `physics/mynn_pbl.py`; existing `physics_couplers.mynn_adapter` | `tests/test_m5_mynn_tier1.py`, `tests/test_mynn_edmf_oracle.py` — `pytest tests/test_m5_mynn_tier1.py` | Real savepoint parity + EDMF oracle. Default PBL. |
| YSU (bl=1) | integrated+tested | `pbl_ysu.py`; `PBL_SCAN_ADAPTERS[1]=ysu_pbl_adapter` | `tests/test_v060_pbl_ysu.py` — `pytest tests/test_v060_pbl_ysu.py` | Savepoint parity vs WRF YSU oracle. |
| MYJ (bl=2) | integrated+tested | `pbl_myj.py`/`myj_adapters.py`; wired in `_SCAN_WIRED_OPTIONS` bl=2 | `tests/test_v060_myj_pbl.py` — `pytest tests/test_v060_myj_pbl.py` | Savepoint parity vs `module_pbl_myj.F`. Mandatorily paired with sf_sfclay=2 (Janjic). |
| ACM2 (bl=7) | integrated+tested | `pbl_acm2.py`; `PBL_SCAN_ADAPTERS[7]=acm2_pbl_adapter` | `tests/test_v060_pbl_acm2.py` — `pytest tests/test_v060_pbl_acm2.py` | Savepoint parity vs WRF oracle. |
| BouLac (bl=8) | integrated-undertested | `pbl_boulac.py`; `PBL_SCAN_ADAPTERS[8]=boulac_pbl_adapter` | **No** test referencing `boulac` found anywhere in `tests/` (verified by grep) | Scan-wired but no dedicated savepoint OR contract test located. Weakest PBL coverage. |
| Shin-Hong (bl=11) | integrated-undertested | `bl_shinhong.py`/`pbl_shinhong.py`; `PBL_SCAN_ADAPTERS[11]=shinhong_pbl_adapter` | Contract/interface coverage only: `tests/contracts/test_v060_physics_interfaces.py` — `pytest tests/contracts/test_v060_physics_interfaces.py` | Scan-wired (v0.18); only interface-shape coverage, no savepoint parity test. |
| GBM (bl=12) | integrated-undertested | `bl_gbm.py`; `PBL_SCAN_ADAPTERS[12]=gbm_pbl_adapter` | Contract/interface coverage only: `tests/contracts/test_v060_physics_interfaces.py` — same cmd | Scan-wired (v0.18); interface-shape coverage only, no savepoint parity test. |
| GFS PBL (bl=3) | integrated-undertested | `bl_gfs.py`; `PBL_SCAN_ADAPTERS[3]=gfs_pbl_adapter` | `tests/test_v017_gfs_pbl_operational.py` — `pytest tests/test_v017_gfs_pbl_operational.py` | Only a wiring/operational smoke test; no oracle parity test located. |
| MRF (bl=99) | integrated-undertested | `bl_mrf.py`; `PBL_SCAN_ADAPTERS[99]=mrf_pbl_adapter` | `tests/test_v013_mrf_operational.py` — `pytest tests/test_v013_mrf_operational.py` | Operational smoke test only (the parity oracle is referenced in code comments but no parity-assert test file located). |

### Surface layer

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| MYNN surface layer (sf_sfclay=5, **default**) | integrated+tested | `mynn_surface_stub.py` + existing surface_adapter | `tests/test_v014_mynn_surface_layer_regressions.py` — `pytest tests/test_v014_mynn_surface_layer_regressions.py` | Real-case regression gate. Default SL. |
| Revised-MM5 (sf_sfclay=1) | integrated+tested | `sfclay_revised_mm5.py`; `SFCLAY_SCAN_ADAPTERS[1]` | `tests/test_v060_sfclay_revised_mm5.py` — `pytest tests/test_v060_sfclay_revised_mm5.py` | Savepoint parity vs WRF oracle. |
| Janjic Eta (sf_sfclay=2) | integrated+tested | `sfclay_janjic.py`; wired (bl=2 pair) | `tests/test_v060_sfclay_janjic.py` — `pytest tests/test_v060_sfclay_janjic.py` | Savepoint parity vs `module_sf_sfclay.F` MYJSFC. |
| Pleim-Xiu SL (sf_sfclay=7) | integrated+tested | `sfclay_pleim_xiu.py`; `SFCLAY_SCAN_ADAPTERS[7]` | `tests/test_v060_sfclay_pleim_xiu.py` — `pytest tests/test_v060_sfclay_pleim_xiu.py` | Savepoint parity. |
| GFS surface layer (sf_sfclay=3) | integrated-undertested | `sfclay_gfs.py`; `SFCLAY_SCAN_ADAPTERS[3]=gfs_sfclay_adapter` | Wiring/pairing tests only: `tests/test_v013_sfclay_pbl_pairing.py`, `tests/test_v013_t3_surface_lsm_wiring.py` — `pytest tests/test_v013_t3_surface_lsm_wiring.py` | Scan-wired; code comment claims fp64 oracle-validated but only wiring tests located, no standalone parity-assert file. |
| Old-MM5 SL (sf_sfclay=91) | integrated-undertested | `sfclay_old_mm5.py`; `SFCLAY_SCAN_ADAPTERS[91]` | Wiring test only: `tests/test_v013_t3_surface_lsm_wiring.py` — `pytest tests/test_v013_t3_surface_lsm_wiring.py` | Scan-wired (v0.13); only wiring coverage, no standalone parity-assert file. |

### LSM (land surface)

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| Noah-MP (sf_surface=4, **default**) | integrated-undertested | `physics/noah_mp.py`, `physics/noahmp/`, `coupling/noahmp_surface_hook.py`; toggled via `use_noahmp`/`sf_surface_physics=4` (operational_mode.py:529, 566) | Real-WRF gate `tests/test_noahmp_energy_canopy.py::test_real_wrf_energy_savepoint_parity` is `@pytest.mark.skipif(not _HAVE_GATE)` (line 282). Self-contained: `tests/test_noahmp_coupler.py`, `tests/test_v014_noahmp_nested_pipeline.py` — `pytest tests/test_noahmp_coupler.py tests/test_v014_noahmp_nested_pipeline.py` | Default LSM, fully wired and exercised in the nested pipeline. But the *real-WRF energy savepoint parity* gate is SKIP-gated on external `proofs/noahmp/savepoints_energy.json`/MPTABLE → conservatively "undertested" for the parity claim; the coupler/end-to-end/conservation tests do run. |
| Noah-classic (sf_surface=2) | not-integrated (reference-only) | `lsm_noah_classic.py`; **NOT** in operational scan (fail-closed `_SCAN_UNWIRED_REASON sf_surface_physics=2`) | `tests/v060/test_noahclassic_parity.py` — `pytest tests/v060/test_noahclassic_parity.py` | Savepoint parity vs `module_sf_noahlsm.F` SFLX EXISTS, but the scheme is NOT scan-wired: fails closed at runtime requiring explicit static bundles. Reference-only. |
| Slab LSM (sf_surface=1) | not-integrated (reference-only) | `lsm_slab.py`; fail-closed (`_SCAN_UNWIRED_REASON sf_surface_physics=1`) | No standalone slab parity test located | Recognized, fail-closed. Needs explicit `slab_static`. |
| Pleim-Xiu LSM (sf_surface=7) | not-integrated (reference-only) | `lsm_pleim_xiu.py`; fail-closed (`_SCAN_UNWIRED_REASON sf_surface_physics=7`) | `tests/test_v017_lsm_pleim_xiu.py` — `pytest tests/test_v017_lsm_pleim_xiu.py` | Savepoint parity vs `module_sf_pxlsm.F` exists, but scan rejects it without explicit `px_static`. Reference-only. |
| RUC LSM (sf_surface=3) | not-integrated (reference-only) | `lsm_ruc.py`; fail-closed (reason: ~7.5k-LOC carry-over) | No JAX-endpoint test (oracle staged at `proofs/v017/oracle/ruclsm`) | Single-column fp64 oracle staged; no traceable JAX kernel → fail-closed. |
| SSiB LSM (sf_surface=8) | not-integrated (reference-only) | `lsm_ssib.py`; fail-closed (reason: ~6.6k-LOC carry-over) | No JAX-endpoint test (oracle at `proofs/v017/oracle/ssib`) | Same as RUC. |

### Radiation

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| RRTMG LW (ra_lw=4, **default**) | integrated+tested | `rrtmg_lw.py`; `_SCAN_WIRED_OPTIONS ra_lw=4` | `tests/test_m5_rrtmg_tier1.py`, `tests/test_m5_rrtmg_harness.py` — `pytest tests/test_m5_rrtmg_tier1.py` | Real harness + savepoint parity. Default LW. |
| RRTMG SW (ra_sw=4, **default**) | integrated+tested | `rrtmg_sw.py`; `_SCAN_WIRED_OPTIONS ra_sw=4` | `tests/test_m5_rrtmg_tier2.py` (asserts `shortwave_real_driver_energy_conservation`/`heating_flux_closure`) — `pytest tests/test_m5_rrtmg_tier2.py` | Real-driver SW energy-conservation + flux-closure asserts. Default SW. |
| Dudhia SW (ra_sw=1) | integrated+tested | `ra_sw_dudhia.py`; scan-wired held-rate theta tendency | `tests/test_v060_ra_sw_dudhia.py`, `tests/test_cdudhia_sw_operational_wiring.py` — `pytest tests/test_v060_ra_sw_dudhia.py` | Savepoint parity vs `module_ra_sw.F`. |
| GSFC/Chou-Suarez SW (ra_sw=2) | integrated+tested | `ra_sw_gsfc.py`; scan-wired held-rate theta tendency | `tests/test_v013_ra_sw_gsfc.py` — `pytest tests/test_v013_ra_sw_gsfc.py` | Savepoint parity. |
| Classic AER RRTM LW (ra_lw=1) | integrated-undertested | `ra_lw_rrtm.py`/`ra_lw_rrtm_jax.py`; `_SCAN_WIRED_OPTIONS ra_lw=1` | `tests/test_rrtm_lw_operational_wiring.py` — `pytest tests/test_rrtm_lw_operational_wiring.py` | Scan-wired but only a wiring/operational test located (no standalone column oracle parity test file). |
| Held-Suarez idealized (ra_lw=31) | integrated+tested | `ra_lw_hs.py`; `_SCAN_WIRED_OPTIONS ra_lw=31` (requires ra_sw=0) | `tests/test_v017_ra_lw_hs.py` — `pytest tests/test_v017_ra_lw_hs.py` | Analytic radiative-equilibrium oracle. Idealized only. |
| McICA cloud overlap | integrated (within RRTMG) | part of `rrtmg_lw.py`/`rrtmg_sw.py` | covered indirectly by RRTMG tier tests; no standalone McICA test located | Not a standalone switch — lives inside RRTMG; no dedicated McICA stochastic-overlap test located. Flag as undertested-as-isolated-component. |
| CAM / New-Goddard / FLG / GFDL-Eta SW+LW (3/5/7/99) | not-integrated (reference-only) | reference oracles only; fail-closed (`_SCAN_UNWIRED_REASON ra_sw/lw=3/5/7/99`) | exact-driver real-WRF oracles at `proofs/v018/savepoints/ra_tail_wrf/` | Oracle exists, no JAX kernel → fail-closed. |

### Cumulus / convection

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| No cumulus (cu=0, **default**) | integrated+tested | n/a (grid-scale only) | covered by all default-suite runs | Default; nests run resolved convection. |
| Kain-Fritsch (cu=1) | integrated+tested | `cumulus_kf.py`; `CU_SCAN_ADAPTERS[1]=kf_adapter` | `tests/test_kf_cumulus_oracle.py`, `tests/test_v060_cumulus_kf.py` — `pytest tests/test_kf_cumulus_oracle.py` | Real single-column oracle vs `module_cu_kfeta.F`. Carry-threaded (w0avg, nca). |
| BMJ (cu=2) | integrated-undertested | `cumulus_bmj.py`; `CU_SCAN_ADAPTERS[2]=bmj_adapter` | **No** `test_*bmj*` file found in `tests/` | Scan-wired, carry-threads CLDEFI, but no dedicated BMJ test located. |
| Grell-Freitas (cu=3) | integrated-undertested | `cumulus_grell_freitas.py`/`_gf_jax.py`; `CU_SCAN_ADAPTERS[3]=gf_adapter` | `tests/test_grell_freitas_cumulus.py` — `pytest tests/test_grell_freitas_cumulus.py` | GPU-batched, savepoint-parity gated at `proofs/v060/gf_gpubatch_savepoint_parity.json`, but the test file's strong asserts are a scale-factor physical check + conditional parity-report (parity only when savepoints present). Conservative → undertested. |
| Tiedtke modified (cu=6) | integrated+tested | `cumulus_tiedtke.py`/`_tiedtke_jax.py`; `CU_SCAN_ADAPTERS[6]=tiedtke_adapter` | `tests/test_tiedtke_cumulus_oracle.py` — `pytest tests/test_tiedtke_cumulus_oracle.py` | Per-scheme oracle parity vs `module_cu_tiedtke.F`. **Caveat:** requires active flux-form moisture advection so the scan can diagnose RQVFTEN (operational_mode.py:3588 `tiedtke_lacks_rqvften` guard). |
| New-Tiedtke (cu=16), SAS family (4/94/95/96), Grell-3D (5), Grell-Devenyi (93), KSAS (14), prev-KF (99) | not-integrated (reference-only / fail-closed) | various reference oracles; fail-closed (`_SCAN_UNWIRED_REASON cu_physics=...`) | oracle artifacts staged (proofs/v013, v017, v018); no operational wiring | SAS-family shared JAX endpoint is RED vs oracle (`proofs/v017/sas_family_parity.json`). All fail-closed. |

### Dynamics core (ARW split-explicit)

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| RK3 time integration | integrated+tested | `dynamics/rk3.py`, `dynamics/core/dycore.py` | `tests/test_m4_rk3.py` (3rd-order convergence) + `tests/test_m4_dycore_step.py` — `pytest tests/test_m4_rk3.py` | Analytic 3rd-order convergence oracle + dycore-step integration. |
| Acoustic substeps | integrated+tested | `dynamics/acoustic.py`, `acoustic_wrf.py`, `core/acoustic.py` | `tests/test_m4_acoustic.py` (divergence-free + finite) + `tests/test_m6b4_acoustic_recurrence_parity.py` — `pytest tests/test_m4_acoustic.py` | Physical oracle. v0.21 touched `acoustic_wrf.py` (dycore stability). |
| advance_w / W solver | integrated+tested | `dynamics/core/advance_w.py` | `tests/dynamics/test_advect_w_topface.py`, `tests/test_m6b0r_calc_coef_w_fix.py` — `pytest tests/dynamics/test_advect_w_topface.py` | Conservation/oracle. |
| rhs_ph / pressure gradient | integrated+tested | `dynamics/core/rhs_ph.py`, `dynamics/tendencies.py` | `tests/test_v014_rhs_ph_real_case.py`, `tests/test_m6_horizontal_pressure_gradient_fix.py` — `pytest tests/test_v014_rhs_ph_real_case.py` | Real-case + WRF-reference parity. |
| Scalar/moisture advection (PD, flux-form) | integrated+tested | `dynamics/advection.py`, `flux_advection.py` | `tests/test_m4_advection.py` (monotonicity+conservation), `tests/dynamics/test_pd_monotonic_advection.py` — `pytest tests/test_m4_advection.py` | Positive-definite monotone advection oracles. |
| Divergence damping / damping | integrated-undertested | `dynamics/damping.py` | No standalone `test_*div_damp*` file located (exercised inside dycore-step tests) | Present and used in dycore; no isolated unit test located. |
| Explicit diffusion / hyperdiffusion (Smagorinsky) | integrated+tested | `dynamics/explicit_diffusion.py`, `hyperdiffusion.py` | `tests/dynamics/test_diffopt1_smagorinsky.py`, `tests/dynamics/test_deformation_momentum_diffusion.py` — `pytest tests/dynamics/test_diffopt1_smagorinsky.py` | Analytic dissipation oracles. |
| Vertical implicit solver (tridiagonal) | integrated-undertested | `dynamics/vertical_implicit_solver.py`, `tridiag_solve.py`, `physics/tridiagonal_solver.py` | No standalone solver unit test located; exercised inside dycore/acoustic tests | Present; no isolated tridiag oracle test located. |
| Hybrid vertical coordinate | integrated+tested | `dynamics/hybrid_eta.py` | `tests/test_m6x_c2_hybrid_eta.py` — `pytest tests/test_m6x_c2_hybrid_eta.py` | Hybrid-sigma-pressure stability oracle. |
| Full dycore-step parity vs WRF wrfout | integrated-undertested | `dynamics/core/dycore.py`, `dynamics/step.py` | `tests/test_m6b5_dycore_step_parity.py` — `pytest tests/test_m6b5_dycore_step_parity.py` | The deep wrfout-vs-wrfout parity test body is `@pytest.mark.skipif(not SOURCE_WRFOUT.exists())` (line 46); schema/ladder asserts always run. |

### Nesting

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| Multi-domain live nesting (d01→dN), scheduler/cadence | integrated+tested | `integration/nested_pipeline.py` (`execute_nested_pipeline`, `NestedPipelineConfig`, `max_dom`), `nesting/scheduler.py` | `tests/test_p0_1a_nesting.py` (subcycle 9-3-1, forcedown ordering — verified collectable), `tests/test_v0110_domain_tree.py` — `pytest tests/test_p0_1a_nesting.py` | Out-of-the-box d01→dN pipeline. |
| Nest interpolation (sint/bilinear) | integrated+tested | `nesting/interp.py` | `tests/test_p0_1a_nesting.py::test_sint_*` — `pytest tests/test_p0_1a_nesting.py` | Exact-on-linear analytic oracle. |
| Child boundary construction | integrated+tested | `nesting/boundary_construction.py` (`build_child_boundary_package`) | `tests/test_p0_1a_nesting.py` (bdy leaf shape/cadence) — same cmd | Analytic oracle. |
| 2-way feedback | integrated-undertested | `coupling/boundary_feedback.py`; `NestedPipelineConfig` feedback flag (nested_pipeline.py:~92) | `tests/test_v0110_boundary_feedback.py`, `tests/test_v0120_feedback_smoother.py` — `pytest tests/test_v0110_boundary_feedback.py` | Feedback path present + tested but smaller coverage than 1-way; default pipeline is 1-way. |
| 9-nest / 7-island case | integrated-undertested | `nested_pipeline.py` driving `max_dom=9` | `tests/test_v014_noahmp_nested_pipeline.py`, `tests/test_v020_nested_event_tail_guard.py`; GPU canary gate `proofs/v021/canary_gate/` — `pytest tests/test_v020_nested_event_tail_guard.py` | The full 9-domain run is a GPU-gated canary (not a CPU unit test); v0.21 carries a Mont-Blanc steep-terrain stability limitation (RELEASE_NOTES draft). |

### Lateral boundary conditions / forcing

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| Specified + relaxation zone | integrated+tested | `coupling/boundary_apply.py` (`apply_lateral_boundaries`), `io/boundary_replay.py` | `tests/test_m6_boundary_apply.py` (sets specified zone + relaxes inner), `tests/test_m6_boundary_replay.py` — `pytest tests/test_m6_boundary_apply.py` | Spec/relax zone + hourly linear replay. |
| Boundary forcing replay (d02) | integrated+tested | `integration/d02_replay.py`, `io/boundary_replay.py` | `tests/test_m6x_d02_boundary_replay.py` — `pytest tests/test_m6x_d02_boundary_replay.py` | Real-case D02 replay. v0.21 touched `d02_replay.py` minorly. |
| Daily boundary clock / cadence | integrated+tested | `io/boundary_replay.py` clock logic | `tests/test_daily_boundary_clock.py`, `tests/test_v014_specified_bdy_cadence.py` — `pytest tests/test_daily_boundary_clock.py` | Time-cadence oracle. |
| External met forcing decode (real.exe inputs / met_em) | integrated-undertested | `io/gen2_wrfout_loader.py`, `io/data_inventory.py`, `init/` | `tests/init/test_forcing_decode.py` — `pytest tests/init/test_forcing_decode.py` | Decodes forcing/IC; structural test. AIFS/GFS forcing is fed as decoded IC/BC, not a live coupler. |

### Other WRF-v4 features

| Feature | Status | Code location | Test | Notes |
|---|---|---|---|---|
| Gravity-wave drag (GWDO) | integrated+tested | `physics/gwd_gwdo.py`, `io/gwdo_static.py` | `tests/test_gwd_gwdo.py` (zero-drag flat terrain, decelerates wind, column stress) — `pytest tests/test_gwd_gwdo.py` | Conservation/physical oracle + `tests/test_gwd_operational_wiring.py` smoke. |
| Restart / checkpoint | integrated+tested | `io/restart.py`, `io/wrfrst_netcdf.py` | `tests/test_v0110_wrfrst_netcdf.py`, `tests/test_m7_restart_checkpoint_roundtrip.py`, `tests/test_noahmp_checkpoint_v2.py` — `pytest tests/test_v0110_wrfrst_netcdf.py` | Bitwise state roundtrip. |
| History output (wrfout, sync + async) | integrated+tested | `io/wrfout_writer.py`, `io/async_wrfout.py` | `tests/test_async_wrfout_equiv.py` (sync==async byte-identical) — `pytest tests/test_async_wrfout_equiv.py` | Determinism gate. |
| Auxiliary history streams (auxhist, sub-hourly) | integrated+tested | `io/auxhist_stream.py` | `tests/test_auxhist_stream.py`, `tests/test_auxhist_multistream.py` — `pytest tests/test_auxhist_stream.py` | Distinct-frame + WRF filename semantics. |
| Training-subset output | integrated+tested | `io/wrfout_writer.py` variable_subset path | `tests/test_v0201_training_output_subset.py` — `pytest tests/test_v0201_training_output_subset.py` | Default byte-identical; opt-in subset (v0.20.1/.2 added OLR/RAINC/SWDNB). |
| Conservation budget (mass/water) | integrated+tested | `diagnostics/conservation_budget.py` | `tests/test_conservation_budget.py` (verified collectable: closure + WRF mass-area units) — `pytest tests/test_conservation_budget.py` | Dry-mass + LBC + water closure to rtol 1e-13. |
| fail-fast finite detector (v0.21 NEW) | integrated+tested | `GPUWRF_FINITE_CHECK` path | `tests/test_gpu_preflight.py` + A1 work (this branch HEAD `e8915772` "Add fail-fast finite state guard") | v0.21 stability feature; observational only (does not mutate values). |
| WRF-Chem, WRF-Fire, FDDA/nudging, multi-layer urban (BEP/BEM), moving/vortex nests, stochastic physics, ocean physics | **OUT_OF_SCOPE** | classified in `io/scheme_catalog.py` as `OUT_OF_SCOPE` truthy switches | n/a (namelist validator fail-closes) | Documented design decision NOT to port. Selecting any fails closed. |

---

## Prose answer to the user's question

**Which common WRF-v4 features are actually INTEGRATED (operationally GPU-scan-wired)?**
The full default operational suite is integrated and is the well-tested core: **Thompson microphysics (mp=8)**, **MYNN PBL + MYNN surface layer (bl=5/sf_sfclay=5)**, **Noah-MP LSM (sf_surface=4)**, **RRTMG longwave + shortwave (ra_lw=4/ra_sw=4)**, **no-cumulus grid-scale convection (cu=0)**, on top of the **ARW split-explicit dynamics core** (RK3 + acoustic substeps + advance_w + rhs_ph + flux/PD advection + hybrid vertical coordinate + explicit/hyper diffusion). Beyond the default suite, a large menu of additional schemes is genuinely scan-wired (per `_SCAN_WIRED_OPTIONS`): microphysics mp ∈ {1,2,3,4,6,10,13,14,16,24,26,28,97}; PBL bl ∈ {1,2,3,7,8,11,12,99}; surface-layer sf_sfclay ∈ {1,2,3,7,91}; cumulus cu ∈ {1,2,3,6}; SW ra_sw ∈ {1,2,4}; LW ra_lw ∈ {1,4,31}. Live multi-domain nesting (1-way and 2-way feedback), specified+relaxation lateral boundaries with hourly forcing replay, gravity-wave drag, restart/checkpoint, history + async + auxhist + training-subset output, and a mass/water conservation budget are all integrated.

**Which are integrated but UNDERTESTED (wired, but only smoke/wiring-tested, or the real-WRF parity test is skip-gated, or no isolated test):** Lin microphysics (mp=2, no standalone test); Morrison (mp=10, real parity is `skipif`-gated); PBL BouLac/Shin-Hong/GBM (bl=8/11/12, no dedicated test) and GFS-PBL/MRF (bl=3/99, smoke only); GFS and old-MM5 surface layers (sf_sfclay=3/91, no standalone test); **Noah-MP** (default LSM — end-to-end/coupler/conservation tests run, but the real-WRF *energy savepoint parity* gate is `skipif`-gated on external fixtures); classic RRTM LW (ra_lw=1, wiring-only); BMJ (cu=2, no test) and Grell-Freitas (cu=3, conditional parity); McICA (no isolated test, only inside RRTMG); divergence damping and the vertical implicit/tridiagonal solver (no isolated unit tests — covered only inside dycore-step tests); the full dycore-step and 9-domain canary parity tests (`skipif`/GPU-gated, not CPU unit tests).

**Which are NOT INTEGRATED / deferred:** All non-default LSMs — **Noah-classic (sf=2), thermal-diffusion slab (sf=1), Pleim-Xiu LSM (sf=7), RUC (sf=3), SSiB (sf=8)** — are reference-only: they have oracle/savepoint evidence but are NOT scan-wired and **fail closed** at runtime (requiring explicit static bundles or lacking a traceable JAX kernel). Most cumulus variants are deferred: New-Tiedtke (16), the SAS family (4/94/95/96 — JAX endpoint RED vs oracle), Grell-3D (5), Grell-Devenyi (93), KSAS (14), previous-KF (99). Most radiation schemes beyond RRTMG/Dudhia/GSFC/Held-Suarez are deferred: CAM, New-Goddard, FLG/UCLA, GFDL-Eta SW+LW (3/5/7/99 — real-WRF oracles exist but no JAX kernel). Entirely **out of scope** (documented design decision, fail-closed): WRF-Chem coupled chemistry, WRF-Fire, FDDA/4DVAR nudging, multi-layer urban canopy (BEP/BEM), moving/vortex-following nests, stochastic physics, ocean physics.

**Honesty note on the catalog:** the project's own `io/scheme_catalog.py` enforces these distinctions with `assert_catalog_consistent`, reading ground truth from `_SCAN_WIRED_OPTIONS` and the `scan_adapters` registries — so an option is only labeled `IMPLEMENTED` if it is actually threaded into the operational scan. This audit's "integrated" column matches that authority; the "undertested" downgrades are this auditor's conservative read of the *test* evidence, which the catalog does not itself grade.

---

## What v0.21.0 can honestly claim to support (README-ready)

> wrf_gpu v0.21.0 runs a WRF-v4 ARW forecast out of the box on GPU with a faithful split-explicit dynamics core (RK3 + acoustic substeps, positive-definite/monotone advection, hybrid vertical coordinate, explicit + hyper diffusion, gravity-wave drag). The default physics suite — Thompson microphysics, MYNN PBL and surface layer, Noah-MP land surface, and RRTMG longwave + shortwave radiation — is scan-wired and validated against unmodified-WRF savepoint oracles. A broad additional menu of schemes is operationally selectable and savepoint-validated: microphysics (Kessler, Lin, WSM3/5/6/7, WDM5/6/7, Morrison, SBU-YLin, Goddard, aerosol-aware Thompson mp=28); PBL (YSU, MYJ, ACM2, BouLac, Shin-Hong, GBM, GFS, MRF); surface layers (revised-MM5, Janjic, Pleim-Xiu, GFS, old-MM5); cumulus (Kain-Fritsch, BMJ, Grell-Freitas, modified Tiedtke); and additional radiation (Dudhia SW, GSFC SW, classic RRTM LW, Held-Suarez idealized). Live multi-domain nesting (1-way and 2-way feedback, demonstrated to the 9-domain canary), specified + relaxation lateral boundaries with forcing replay, restart/checkpoint, deterministic synchronous/asynchronous history output plus sub-hourly auxhist streams and a compact training-subset output, and a mass/water conservation budget are all supported. Every namelist option resolves to exactly one honest status (implemented / reference-only / fail-closed / out-of-scope) via the machine-checked `scheme_catalog`.

## What v0.21.0 does NOT yet support / "not yet" (README-ready)

> v0.21.0 does **not** run, and fails closed (with a named reason), the following: the alternative land-surface models Noah-classic, thermal-diffusion slab, Pleim-Xiu LSM, RUC, and SSiB (these are reference-only — oracle-validated but not wired into the GPU scan); the cumulus schemes New-Tiedtke, the SAS family, Grell-3D, Grell-Devenyi, KSAS, and previous Kain-Fritsch (the SAS endpoint is measured RED vs its oracle); and the radiation schemes CAM, New-Goddard, FLG/UCLA, and GFDL-Eta (real-WRF oracles exist, no GPU kernel yet). Out of scope by design: WRF-Chem, WRF-Fire, FDDA/4DVAR nudging, multi-layer urban canopy (BEP/BEM), moving/vortex-following nests, stochastic physics, and ocean coupling. Several wired-but-undertested schemes carry no dedicated savepoint test (Lin mp=2; PBL BouLac/Shin-Hong/GBM; GFS/old-MM5 surface layers; BMJ cumulus) or only a skip-gated real-WRF parity test (Morrison; the Noah-MP real-WRF energy gate; the full dycore-step and 9-domain canary parity). The 9-domain steep-terrain (Mont-Blanc) case remains a carried stability limitation in v0.21.0; default-on AOT warm-start is deferred to v0.21.1.

---

## Caveats on this audit

- This is a feature/wiring/test-presence audit, not a re-run of the GPU validation gates. Test *presence and classification* were verified by reading files; a quick `pytest --collect-only` confirmed `tests/test_p0_1a_nesting.py` + `tests/test_conservation_budget.py` collect (14 tests). The skip-gated parity tests were confirmed skip-gated by reading their `@pytest.mark.skipif` decorators, not by executing them.
- "No dedicated test found" means no file matching the scheme name was located under `tests/`; such a scheme may still be exercised indirectly by suite-level/regression runs (`tests/regression/test_regression_suite.py`, `tests/regression/oracle_cases.yaml`). These are conservatively listed as integrated-undertested, not untested.
- The total collected suite is large (subagent reported ~2293 tests across the tree); this audit maps the feature-bearing subset, not every infrastructure test.
