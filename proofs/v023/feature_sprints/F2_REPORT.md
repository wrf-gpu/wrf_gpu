# F2 Report — cumulus + advanced MP + LSM family (Fable-5 lane, CPU-only)

**STATUS: F2 COMPLETE (0:1 ruling 2026-07-03; `F2_DONE` touched).**
mp18's faithful port is OWN-MILESTONE-SCOPED by manager ruling (see section 4
and `F2_QUESTION.md`) and is NOT an F2 gap.

Branch: `worker/fable/f2-physics` (off v0.22.2 `53bd20bf`). All work CPU-only
(`JAX_PLATFORMS=cpu`, cores 4-31; GPU + cores 0-3 untouched per 0:2 lane rule).
Escalated from GPT's honest blocker report (`F2_QUESTION.md`, GPT lane): two
schemes had no oracle at all; none had a faithful port.

Final per-scheme states (all fail-closed paths are honest — a rejected scheme
names its reason; nothing is ever silently substituted or silently wrong):

| scheme | delivered | catalog state |
|---|---|---|
| RUC LSM (sf=3) | v018 port integrated + oracle-validated (warm/no-snow column) | REFERENCE_ONLY by design (regime-honest) |
| New-Tiedtke (cu=16) | machine-precision fp64 kernel + scan wiring + end-to-end smoke | **IMPLEMENTED** |
| Morrison-aerosol (mp=40) | oracle built + fp64 port at machine precision | REFERENCE_ONLY pending aerosol-state ADR |
| NSSL 2-moment (mp=18) | oracle built + state contract + porter runbook | REFERENCE_ONLY; port = own milestone (0:1 ruling) |

## Per-scheme outcome

### 1. RUC LSM (`sf_surface_physics=3`) — INTEGRATED + VALIDATED (commit `e58b56d8`)

* Applied the preserved v018 port patch (`proofs/v018/ruc_lsm_port.patch`;
  base drift = one PII-cleaned docstring path) onto trunk: faithful fp64 JAX
  port of the warm/no-snow land column
  (`LSMRUC -> SOILVEGIN -> SFCTMP -> SOIL -> SOILTEMP/SOILMOIST`).
* Restored the v017 fp64 RUC + SSiB oracle drivers/savepoints from commit
  `a922b3b5` (never merged to this lineage); WRF source sha256 re-verified
  against the live pristine tree.
* **Proof:** `tests/test_v017_lsm_adv.py` + `tests/test_v018_ruc_lsm_parity.py`
  = 12 passed; worst residual LH abs 2.898e-04 (tol 5e-04), matching the v018
  GREEN metrics (`proofs/v018/ruc_lsm_parity_metrics.json`).
* **Coverage / honest scope:** the oracle-proven regime is the warm land /
  no-snow column only. **Decision: RUC stays REFERENCE_ONLY / scan-fail-closed.**
  Operational wiring requires (a) threading the `RucLandState` multi-layer
  carry through the scan and (b) snow / sea-ice / broader soil-veg-class oracle
  regimes; wiring now would silently run the kernel outside its proven regime
  (violates no-shortcuts). Documented follow-up, not an F2 gap.

### 2. New-Tiedtke cumulus (`cu_physics=16`) — PORTED + VALIDATED (commit `e682e462`)

* `src/gpuwrf/physics/cumulus_ntiedtke.py`: line-faithful fp64 single-column
  transcription of the full `cumastrn` tree of
  `phys/physics_mmm/cu_ntiedtke.F90` + the `module_cu_ntiedtke.F` wrapper
  (pre/post, orientation flip, tendency reconstruction).
* Fidelity findings that were load-bearing for machine precision (documented in
  the module): WRF's unsuffixed Fortran literals are default-REAL (fp32) and
  must be rounded through fp32 (incl. `cuentrn`'s `0.75e-4` detrainment rate);
  `amax1` demotes the precip accumulator through REAL*4 (verified bit-exact).
* **Proof:** `tests/test_ntiedtke_cumulus_oracle.py` (7 tests) vs the 5 fp64
  WRF oracle savepoints (`proofs/v013/savepoints/cumulus/`): RAINCV
  bit-identical (<= 1 ulp), tendency fields max_abs <= 1e-15 (predeclared band
  1e-12), regimes deep(1,3)/shallow(2)/non-triggering(4,5) all reproduced.
* Traceable JAX kernel + CU scan adapter: DONE — see section 5 (cu16
  graduated to IMPLEMENTED + scan-wired, commit `556e153a`).

### 3. Morrison-aerosol MP (`mp_physics=40`) — ORACLE BUILT + PORTED, MACHINE PRECISION (commits `bb8cd8ed`, `ba1294af`)

* **Oracle (was GPT's blocker):** standalone single-column driver vs UNMODIFIED
  `phys/module_mp_morr_two_moment_aero.F`, `aercu_opt=2` (the only genuinely
  aerosol-aware mode: prognostic NC, Abdul-Razzak&Ghan activation, Liu-Penner
  ice nucleation), 10-species prescribed AEROCU profiles, staggered KZH input;
  fp32 + fp64 x 6 regimes -> `proofs/v022/f2_oracles/morrison_aero/` (drivers +
  porter README at `proofs/v023/oracle/morraero/`). All finite, activation
  nontrivial (~185 cm^-3 clean .. ~870 polluted), fp32~fp64 within the
  base-Morrison-anchored band.
* **Port:** `physics/microphysics_morrison_aero.py` + `_morrison_aero_cold.py`
  — a faithful delta on the proven base Morrison port (DCS=350e-6 cascade,
  prognostic NC/iinum=0, 10-mode activation with exact DERF1, INUC=2
  nucleation, WVAR from KZH, EFCG/EFIG/EFSG/WACT/CCN diagnostics; every ported
  and every skipped Fortran delta enumerated in the module docstrings).
* **Proof:** `tests/savepoint/test_morrison_aero_parity.py` = 13 tests green;
  fp64 gate at machine precision (worst field rel ~2e-13 vs predeclared 1e-9;
  NC rel <= 9.4e-16; RAINNCV <= 5e-17 mm); fp32 physical band green incl. the
  two oracle-documented threshold-flip cases (4.26%/5.87% vs the oracle pair's
  own 4.26%/5.85%). Base Morrison suite untouched and green.
* **Honest scope:** kernel-proven => catalog REFERENCE_ONLY (namelist-accepted);
  operational scan wiring stays fail-closed because the AEROCU aerosol inputs +
  prognostic droplet number have no operational State substrate (needs a small
  aerosol-state ADR; named in catalog + scan reasons).

### 4. NSSL 2-moment MP (`mp_physics=18`) — ORACLE BUILT + SEAM FLIPPED; PORT = OWN MILESTONE (commits `bb8cd8ed`, `ba1294af`)

* **Oracle (was GPT's blocker):** standalone driver vs UNMODIFIED
  `phys/module_mp_nssl_2mom.F` in the WRF-default mp=18 configuration (every
  init choice cited to `module_check_a_mundo.F` / `module_physics_init.F`);
  fp32 + fp64 x 6 regimes -> `proofs/v022/f2_oracles/nssl_2mom/` (drivers +
  porter runbook at `proofs/v023/oracle/nssl2mom/`). Honest gap documented:
  current seeds never exercise the hail (QHL) process rates — a hail-seeded
  supplementary case belongs to the port sprint.
* **Seam:** flipped `recognized_fail_closed` -> `REFERENCE_ONLY` with the full
  interlock (registry acceptance + species maps, step specs, fail-closed stub
  endpoint `microphysics_nssl2mom.nssl2mom_run`, catalog + scan reasons);
  `qvolg`/`qvolh` volume scalars have no State substrate — named blocker.
* **Port:** NOT delivered in F2, deliberately. Measured scope of the
  oracle-exercised path: ~19k LOC (`nssl_2mom_gs` alone ~12.5k) vs 3.6k for
  the entire New-Tiedtke core — a dedicated multi-day milestone. Rushing it
  would produce unverifiable physics. Decision request in `F2_QUESTION.md`.
* **0:1 RULING (2026-07-03): mp18 = OWN MILESTONE, not part of F2.** F2 hands
  the port sprint a ready launchpad: verified oracle, frozen state contract,
  porter runbook, and the named hail-seed + qvolg/qvolh follow-ups.

### 5. cu16 operational wiring — DONE, IMPLEMENTED + SCAN-WIRED

* **Traceable kernel:** `cumulus_ntiedtke_jax.ntiedtke_column_jax` (jit/vmap,
  fp64, all fp32-literal + `amax1` fidelity reproduced; early exits as carried
  masks; `jnp.trunc` for the Fortran INTEGER truncations).
  **Proof:** `tests/test_ntiedtke_jax_parity.py` (16 tests): vs the 5 oracle
  savepoints RAINCV/PRATEC diff 0.0, tendencies <=3.6e-16; vs the NumPy
  reference <=1.4e-15 max_abs over the savepoints + 25 seeded perturbed
  variants spanning deep(10)/shallow(3)/none(12) regimes; jit compiles once;
  width-1 vmap is BITWISE identical to the scalar call (hard assert); the
  width-8 batch differs <=2.5e-16 solely from XLA:CPU SIMD-width transcendental
  rounding (identical jaxpr; documented in the test — accepted ruling).
* **Scan adapter + runtime:** `coupling.scan_adapters.ntiedtke_adapter`
  (stateless State->State, vmapped; HFX/QFX from the B2 kinematic flux handles,
  DX from the grid projection) wired as `CU_SCAN_ADAPTERS[16]`; the runtime
  threads WRF RQVFTEN = flux-form qv-advection diagnostic + PBL qv increment,
  and RTHFTEN = accumulated non-convective physics theta forcing. **Named
  coupling caveat:** WRF's RTHFTEN also contains the advective theta tendency;
  the dycore does not expose a step-entry theta-advection diagnostic, so that
  closure-modulating component is absent (the convective tendencies themselves
  are oracle-proven). Like cu=6, cu=16 REQUIRES `use_flux_advection=True` +
  `moist_adv_opt=1/2` and fail-closes otherwise (guard extended + tested).
* **End-to-end proof:** `tests/test_v013_operational_smoke.py::
  test_cumulus_ntiedtke_operational_triggers_with_real_forcing` — cu16 runs
  through the real operational physics step on a conditionally unstable column
  with real threaded forcing: deep convection triggers, `rainc_acc > 0`,
  theta tendency applied, all fields finite; plus the fail-closed guard test.
* **Catalog graduation:** cu16 `REFERENCE_ONLY -> IMPLEMENTED` across catalog /
  dispatch (owner `cumulus_ntiedtke_jax`, gpu_runnable) / step spec / scan-wired
  options; five tests that used cu16 as the canonical reference-only exemplar
  switched to still-reference-only schemes (cu=4/14). Full battery: 227 passed.

## Honesty-gate transition (was GPT's "do not create F2_DONE" state)

`proofs/v022/f2_missing_scheme_bundle_oracle_check.py` extended with an
`implemented_scan_wired` case (a scheme may graduate only with oracle evidence
present); `tests/test_v022_f2_missing_scheme_bundle.py` updated to the new true
state (mp18/mp40 `reference_only_oracle_present`, 12 savepoints each). The v018
MP-closure manifest + evidence tree (62 files, ~1 MB), missing from this
lineage, was restored from `487082e4` and its mp18/40 entries updated to a new
`reference_only_accepted` endpoint class; sanitized `<USER_HOME>` evidence
paths rewritten repo-relative via the `data/wrf_pristine` symlink.

## Commands (representative)

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 4-31 \
    python3 -m pytest -q tests/test_v017_lsm_adv.py tests/test_v018_ruc_lsm_parity.py \
    tests/test_ntiedtke_cumulus_oracle.py tests/savepoint/test_morrison_aero_parity.py \
    tests/savepoint/test_morrison_parity.py tests/test_v022_f2_missing_scheme_bundle.py \
    tests/test_v018_mp_family_fail_closed.py tests/test_v060_physics_dispatch.py \
    tests/test_scheme_catalog_fail_closed.py tests/contracts/ tests/test_namelist_check.py
bash proofs/v023/oracle/morraero/morraero_build_and_run.sh fp32   # + fp64
bash proofs/v023/oracle/nssl2mom/nssl2mom_build_and_run.sh fp32   # + fp64
```

## Carried follow-ups (post-F2; noted per 0:1 ruling, non-blocking)

* **ADR follow-up 1:** mp40 aerosol-state carry (AEROCU 10-species inputs +
  prognostic droplet number State substrate) -> unlocks mp40 scan wiring.
* **ADR follow-up 2:** mp18 `qvolg`/`qvolh` volume-scalar substrate -> part of
  the mp18 own-milestone port sprint.
* mp18 faithful port = own milestone (0:1 ruling; launchpad delivered by F2).
* RUC operational wiring: land-carry threading + snow/ice oracle regimes.
* NSSL hail-process regime not yet exercised by the oracle seeds (documented;
  hail-seeded supplementary case belongs to the port sprint).
* Default physics path untouched throughout: every change is registry/catalog/
  reference-only or opt-in (cu16 activates only when explicitly selected in the
  namelist); no operational scheme's numerics were modified.
