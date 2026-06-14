# WRF `namelist.input` compatibility

**Bring your existing WRF `namelist.input`.** The GPU port reads a standard
Fortran WRF v4 ARW `namelist.input` (all the usual groups: `&time_control`,
`&domains`, `&physics`, `&dynamics`, `&bdy_control`, ...) and, *before any
expensive JAX import or compile*, validates the physics/dynamics scheme
selections fail-closed. You never get a silent wrong answer from selecting a
scheme the port has not implemented. This is the **no-silent-gaps** contract:
every WRF v4 option is either implemented, or loudly fail-closed with a named
reason, or a documented out-of-scope decision.

## What this port is (and is not)

This is a **fast, GPU-native, GPU-scalable, WRF-compatible** atmospheric model
(JAX/XLA, fp64). It reuses WRF's physics formulations and reads WRF inputs, but
it is **not** a bit-for-bit Fortran port: **bit-identity with CPU-WRF is not
claimed**. The validation target is operational fidelity (savepoint / oracle
parity on isolated schemes; RMSE-class equivalence on the integrated forecast),
not per-cell byte equality. For the open 24 h forecast-skill equivalence gap and
all shipped carry-overs see [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

### OPERATIONAL vs REFERENCE-ONLY — the distinction that matters

Two of the statuses below carry a load-bearing distinction that a gate-keeper
must not blur:

* **OPERATIONAL** (`implemented`) — the scheme is **wired into the operational
  GPU forecast scan** and runs end-to-end under `gpuwrf run`. This is the only
  status that runs an actual forecast.
* **REFERENCE-ONLY** (`reference_only`) — the scheme has a **parity-proven
  single-column oracle** (savepoint / isolated / analytic), and is *accepted by
  the namelist validator* so you can run a reference / single-column comparison
  — but it is **NOT wired into the operational scan**. `gpuwrf run`
  **refuses it loudly** (it would otherwise silently substitute a different
  scheme). A REFERENCE-ONLY scheme is **never** part of an operational forecast.

The tables below tag every option `[OPERATIONAL]` or `[REFERENCE-ONLY]`. The
classification is derived directly from `scheme_catalog._IMPLEMENTED` (which
mirrors `runtime.operational_mode._SCAN_WIRED_OPTIONS` + the
`coupling.scan_adapters` registries) and the `scheme_catalog._REFERENCE_ONLY`
set — not hand-maintained. `assert_catalog_consistent()` enforces
`accepted = OPERATIONAL ∪ REFERENCE-ONLY` per parameter, so the catalog can
never over-claim a reference-only scheme as operational or silently drop an
accepted one.

## The support catalog: five honest statuses

A machine-readable catalog (`src/gpuwrf/io/scheme_catalog.py`) classifies
*every* WRF v4 code of the gated namelist parameters — `mp_physics`,
`cu_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `sf_surface_physics`,
`ra_lw_physics`, `ra_sw_physics`, the dynamics options `diff_opt`, `km_opt`,
`damp_opt`, `diff_6th_opt`, `rk_order`, `w_damping`, and `sf_urban_physics` —
against the **full WRF v4 enumeration** (`src/gpuwrf/io/wrf_scheme_catalog.py`,
transcribed from `WRF/run/README.namelist`). Each selection resolves to exactly
one status:

1. **`implemented` (OPERATIONAL) -> runs.** Operationally GPU-scan-wired and
   consumed normally by the forecast.

2. **`reference_only` (REFERENCE-ONLY) -> accepted at the validation layer,
   REFUSED by the operational `gpuwrf run` path.** A recognized WRF scheme with
   a parity-proven (savepoint / isolated / analytic-oracle) adapter that is *not
   yet* threaded into the operational GPU scan. The base namelist validator
   (`validate_namelist`) accepts it so you can run a single-column / reference
   comparison against it. But the **operational forecast** (`gpuwrf run`, via
   `validate_operational_namelist`) **refuses it loudly, before any JAX import**,
   with a named reason — because the operational scan cannot actually select it
   and would otherwise *silently run a different scheme* than you requested.
   Refusing is the honest behavior: never a silent wrong-scheme result. Today
   the `reference_only` schemes are Grell-3D ensemble (`cu_physics=5`), KIM SAS
   (`cu_physics=14`), New Tiedtke (`cu_physics=16`), thermal-diffusion slab LSM
   (`sf_surface_physics=1`), and GSFC/Goddard NUWRF longwave (`ra_lw_physics=5`).
   Example (`gpuwrf run` with `ra_lw_physics=5`):

   ```
   physics.ra_lw_physics=5 (GSFC/Goddard NUWRF longwave): parity-proven WRF v4
   longwave-radiation scheme, but NOT operationally wired into the GPU forecast
   scan. Running it would SILENTLY use a DIFFERENT scheme than requested (the
   operational longwave-radiation path runs the implemented scheme instead) --
   refusing rather than producing a silent wrong-scheme run.
   Operationally-wired ra_lw_physics values: 0, 1, 4. Action: Use
   ra_lw_physics=4 (RRTMG, GPU-operational default) or 1 (classic RRTM).
   ```

3. **`recognized_approximated` -> runs, with a warning.** A recognized WRF
   *cadence* control (`cudt`/`bldt`) whose requested value the port does not
   honor exactly but whose effect is a documented **conservative
   approximation** rather than a wrong scheme: the port calls cumulus/PBL
   **every dynamics step** (more frequent than the requested N-minute
   sub-stepping). The run **proceeds** and a non-fatal warning names the
   approximation. Never used for a genuine wrong-substitution.

4. **`recognized_fail_closed` -> fail closed, specific message.** A valid WRF v4
   option the port does not implement. Example:

   ```
   physics.mp_physics=40 (Morrison 2-moment w/ CESM-NCSU RCP4.5 aerosol):
   recognized WRF v4 microphysics scheme, NOT YET IMPLEMENTED in the GPU port.
   Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 14, 16, 28. ...
   ```

   (Aerosol-*aware* Thompson `mp_physics=28` is now IMPLEMENTED/operational as of
   v0.16 — see the microphysics table below; `mp=40` Morrison-aerosol and the
   NSSL schemes remain fail-closed. A value that is not a WRF option at all —
   e.g. `mp_physics=99` — fails closed with a `not a recognized WRF v4 ...`
   message.)

5. **`out_of_scope` -> fail closed, named scope decision.** A WRF capability the
   port deliberately does not port (see the out-of-scope list below). Example:

   ```
   chem.chem_opt=401 (WRF-Chem coupled chemistry/aerosols): OUT OF SCOPE:
   ... -- a documented out-of-scope decision for this GPU port, NOT silently
   ignored. Action: Run a meteorology-only configuration (chem_opt=0); ...
   ```

The validator **never silently accepts** an unimplemented scheme or an
out-of-scope feature. Multi-domain columns (`mp_physics = 8, 28`) and the
Fortran repeat-count syntax (`mp_physics = 3*8` = three domains of `8`) are both
handled.

## Operational default suite

When a namelist does not pin a physics option, the dispatcher
(`coupling.physics_dispatch`) resolves to the v0.2.0-validated baseline:

| Parameter            | Default | Scheme                         |
|----------------------|---------|--------------------------------|
| `mp_physics`         | 8       | Thompson                       |
| `bl_pbl_physics`     | 5       | MYNN                           |
| `sf_sfclay_physics`  | 5       | MYNN surface layer             |
| `cu_physics`         | 0       | no cumulus (grid-scale only)   |
| `sf_surface_physics` | 4       | Noah-MP (`use_noahmp=True`)    |

Radiation has no dispatcher default; the operational radiation slot runs the
selected SW + LW independently (default `ra_sw_physics=4` / `ra_lw_physics=4`,
RRTMG). `sf_surface_physics` is also resolvable from the legacy `use_noahmp`
toggle (`True` -> 4 Noah-MP, `False` -> 2 Noah classic).

## Per-parameter scheme tables

Each row is one accepted namelist option: code -> WRF scheme name -> status ->
one-line note. `[OPERATIONAL]` runs under `gpuwrf run`; `[REFERENCE-ONLY]` is
accepted for a single-column / reference comparison but **refused operationally**
(it would silently run a different scheme). Codes not listed for a parameter are
`recognized_fail_closed` (valid WRF options the port does not implement) or, for
`sf_urban_physics` 1/2/3, `out_of_scope`.

### Microphysics — `mp_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled / passive qv | OPERATIONAL | qv-only, no condensate physics |
| 1  | Kessler warm rain | OPERATIONAL | qv/qc/qr; savepoint-parity |
| 2  | Purdue-Lin | OPERATIONAL | qv/qc/qr/qi/qs/qg single-moment; savepoint-parity |
| 3  | WSM3 simple ice | OPERATIONAL | qv/qc/qr + re_* diagnostics; savepoint-parity |
| 4  | WSM5 | OPERATIONAL | qv/qc/qr/qi/qs; savepoint-parity |
| 6  | WSM6 | OPERATIONAL | qv..qg; savepoint-parity |
| 8  | Thompson | OPERATIONAL | **default**; +qni/qnr; Tier-4 RMSE vs CPU-WRF corpus |
| 10 | Morrison two-moment | OPERATIONAL | +qni/qns/qnr/qng; savepoint-parity |
| 14 | WDM5 | OPERATIONAL | double-moment 5-class (WDM warm-rain + WSM5 ice, no graupel/hail); reuses WDM6 Nn/Nc/Nr leaves; 6/6 pristine-WRF fp64 oracle |
| 16 | WDM6 | OPERATIONAL | +qnn/qnc/qnr (additive State leaves Nc/Nn); savepoint-parity |
| 28 | aerosol-aware Thompson (water/ice-friendly) | OPERATIONAL | **v0.16 "+1"**; +QNWFA/QNIFA aerosol prognostics (append-only State leaves); climatological self-init only (`use_aero_icbc=.false.`, `wif_input_opt=1`; non-self-init aerosol IC/BC fails closed). **L1 WRF-module oracle PASS** (5187-col, GPU, vs unmodified `module_mp_thompson.F:mp_gt_driver`); the coupled short-real-grid field-gate is a documented **carry** (GPU-time only, separately validated +1, not inside the 25-target L2 sweep) |

WSM7 (`mp=24`) and WDM7 are NOT listed: WSM7's column kernel is ported and
fp64 savepoint-parity-proven (`physics.microphysics_wsm7`), but it carries a
separate precipitating **hail class (`qh`)** that the operational moist-state
pytree (`MOIST_SPECIES`) does not hold, so it fail-closes rather than silently
dropping hail. Wiring it needs a cross-cutting State/dynamics/I-O `qh` leaf.

### Cumulus — `cu_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled | OPERATIONAL | **default** (grid-scale convection only) |
| 1  | Kain-Fritsch | OPERATIONAL | scan-wired; carries NCA/W0AVG; savepoint-parity |
| 2  | Betts-Miller-Janjic | OPERATIONAL | adjustment scheme; carries CLDEFI; fp64 savepoint-parity |
| 3  | Grell-Freitas | OPERATIONAL | v0.9.0 GPU-batched jit/vmap scale-aware adapter; savepoint-parity |
| 5  | Grell-3D ensemble | REFERENCE-ONLY | fp64 single-column oracle staged; JAX kernel is a v0.13 carry-over |
| 6  | Tiedtke | OPERATIONAL | GPU-batched (`cumulus_tiedtke_jax`); savepoint-parity; requires active flux-form moisture advection (`use_flux_advection=True`, `moist_adv_opt=1/2`) so WRF `RQVFTEN` is available |
| 14 | KIM Simplified Arakawa-Schubert | REFERENCE-ONLY | fp64 single-column oracle staged; JAX kernel is a v0.13 carry-over |
| 16 | New Tiedtke | REFERENCE-ONLY | shares Tiedtke kernel but NOT separately source-gated; fail-closed |

### PBL — `bl_pbl_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled | OPERATIONAL | no PBL mixing |
| 1  | YSU | OPERATIONAL | jit/vmap-traceable; savepoint-parity (pair with sfclay=1) |
| 2  | MYJ | OPERATIONAL | v0.13 traceable; **mandatorily paired with `sf_sfclay_physics=2`**; savepoint-parity |
| 5  | MYNN | OPERATIONAL | **default**; Tier-4 RMSE vs CPU-WRF corpus (pair with sfclay=5) |
| 7  | ACM2 | OPERATIONAL | jit/vmap-traceable; savepoint-parity (pair with sfclay=7 or 1) |
| 8  | BouLac | OPERATIONAL | reuses qke TKE leaf; savepoint-parity |
| 99 | MRF | OPERATIONAL | v0.13 jit/vmap port of `module_bl_mrf.F`; savepoint-parity (consumes sfclay=1) |

### Surface layer — `sf_sfclay_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled | OPERATIONAL | no surface-layer fluxes |
| 1  | revised-MM5 (`sfclayrev`) | OPERATIONAL | savepoint-parity |
| 2  | Janjic Eta surface layer | OPERATIONAL | v0.13 traceable; **mandatorily paired with `bl_pbl_physics=2`**; savepoint-parity |
| 3  | NCEP-GFS surface layer | OPERATIONAL | v0.13 Tier-3; fp64 pristine-WRF oracle; land/soil blocks bypassed |
| 5  | MYNN surface layer | OPERATIONAL | **default**; Tier-4 RMSE vs CPU-WRF corpus |
| 7  | Pleim-Xiu surface layer | OPERATIONAL | savepoint-parity |
| 91 | old MM5 surface layer | OPERATIONAL | v0.13 Tier-3; classic 4-regime Monin-Obukhov; fp64 oracle |

### Land surface — `sf_surface_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled | OPERATIONAL | no land-surface model |
| 1  | thermal-diffusion slab LSM | REFERENCE-ONLY | JAX-ported + fp64 oracle, but needs a TSLB land carry + GSW/GLW forcing + TMN/THC/EMISS statics; LSM hook deferred |
| 2  | Noah classic | OPERATIONAL | 4-layer land carry; savepoint-parity |
| 4  | Noah-MP | OPERATIONAL | **default**; set `use_noahmp=True`; savepoint-parity |

### Shortwave radiation — `ra_sw_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled | OPERATIONAL | no SW radiation |
| 1  | Dudhia shortwave | OPERATIONAL | Stephens-1984 broadband; held-rate RTHRATEN; scan-wired with RRTMG/RRTM LW |
| 2  | GSFC (Chou-Suarez) shortwave | OPERATIONAL | v0.13 Tier-3; fp64 pristine-WRF oracle (~1e-12); held-rate RTHRATEN |
| 4  | RRTMG shortwave | OPERATIONAL | **default**; operational radiation slot runs RRTMG SW+LW |

### Longwave radiation — `ra_lw_physics`

| Code | Scheme | Status | Note |
|------|--------|--------|------|
| 0  | disabled | OPERATIONAL | no LW radiation |
| 1  | classic RRTM longwave | OPERATIONAL | AER 16-band; JAX-traceable `ra_lw_rrtm_jax`; held-rate RTHRATEN |
| 4  | RRTMG longwave | OPERATIONAL | **default** |
| 5  | GSFC/Goddard NUWRF longwave | REFERENCE-ONLY | fp64 single-column oracle staged (`module_ra_goddard.F:lwrad`); JAX kernel a v0.13 carry-over (~12.5k-LOC NUWRF SW+LW module, ~11.8k LW coefficients) |

SW and LW are selected and dispatched **independently**. Whatever SW scheme is
chosen is composed with the chosen LW scheme as the held-rate RTHRATEN θ
tendency. The surface SWDOWN / GSW / GLW **history diagnostics** remain
RRTMG-derived regardless of which SW/LW θ-tendency scheme is active.

### Status counts (derived, v0.16)

| Parameter            | OPERATIONAL | REFERENCE-ONLY |
|----------------------|-------------|----------------|
| `mp_physics`         | 11 | 0 |
| `cu_physics`         | 5 | 3 |
| `bl_pbl_physics`     | 7 | 0 |
| `sf_sfclay_physics`  | 7 | 0 |
| `sf_surface_physics` | 3 | 1 |
| `ra_sw_physics`      | 4 | 0 |
| `ra_lw_physics`      | 3 | 1 |

(Counts include the `0`/disabled option, which is operationally wired.)

### Dynamics / numerics

`rk_order=3` (RK3 only); `diff_opt` 0/1/2; `km_opt` 0/1/4; `diff_6th_opt` 0/2
(2 = monotonic 6th-order filter, no up-gradient flux); `damp_opt` 0/3 (3 =
upper-level w-Rayleigh); `w_damping` 0/1; `sf_urban_physics=0` only. The 3-D
closures `km_opt=2/3/5` fail closed (transition: constant-K `diff_opt=2`/`km_opt=1`).

### Mandatory WRF pairing enforced

MYJ PBL (`bl_pbl_physics=2`) <-> Janjic Eta surface layer
(`sf_sfclay_physics=2`) must be selected together (or neither as 2). Selecting
exactly one as 2 fails closed both at validation and in the dispatcher.

## Environment knobs that change operational behaviour

* **`GPUWRF_GWD_NESTED`** (default `1` = ON). Orographic gravity-wave drag +
  flow blocking (`gwd_opt=1`, Kim-GWDO of Choi & Hong 2015) on the **nested**
  pipeline. In v0.12.0 this was gated OFF by default because the 24 h
  nested-1 km + GWD run exceeded the single-GPU fp64 VRAM ceiling (~28 GB) at
  ~sim-hr 7. v0.13's RRTMG g-point + optics/taumol VRAM chunking (SW −88.6 % /
  LW −43.6 %) gave enough headroom that the run now passes `PIPELINE_GREEN`, so
  GWD is **ENABLED by default**; set `GPUWRF_GWD_NESTED=0` to force it off for a
  memory-tighter config. `gwd_opt=1` requires the sub-grid orography statics
  (VAR/CON/OA1-4/OL1-4) in `wrfinput`. `gwd_opt=3` (GSL drag suite) is not wired.

## Recognized non-enumerated controls (advection, GWD, MYNN-EDMF, cadence)

These real WRF keys are recognized and gated to their operationally-wired
value(s); any other value fails closed (or, for the cadence keys, warns and
runs):

* **Advection orders** frozen to `h=5 / v=3` (5th-order horizontal, 3rd-order
  vertical — the WRF real-data default). `moist_adv_opt` / `scalar_adv_opt`
  accept 0 (standard), 1 (positive-definite), 2 (monotonic); the WENO variants
  (3/4) are not wired.
* **`gwd_opt`** 0/1 (see `GPUWRF_GWD_NESTED` above); 3 not wired.
* **`slope_rad`** 0/1 and **`topo_shading`** 0/1 — RRTMG SW slope-radiation +
  topographic-shadow ARE wired (IMPLEMENTED). `slope_rad=2` not separately wired.
* **MYNN-EDMF sub-options** gated to the WRF default sub-config: `bl_mynn_edmf=1`,
  `edmf_mom=1`, `edmf_tke=0`, `mixscalars=1`, `mixqt=0`, `edmf_dd=0`,
  `mixlength` 1|2. `icloud_bl=1` (MYNN-radiation cloud-fraction coupling) and
  `bl_mynn_tkeadvect=.true.` are NOT scan-wired (fail closed if set).
* **`radt`** — any positive interval honoured as the radiation cadence
  (`radiation_cadence_steps = round(radt*60/dt)`). **`bldt` / `cudt`** — the
  port runs PBL/cumulus **every dynamics step**; a positive interval is a
  non-fatal approximation warning (the run proceeds), not a rejection.

## Out-of-scope features (documented decisions, fail closed)

These WRF capabilities are deliberately **not ported**. Selecting a truthy
(non-zero / `.true.`) value for the corresponding switch fails closed with a
scope decision and the disable/alternative recipe — it is never silently
ignored:

| Feature                                    | Switch(es)                          | Disable with |
|--------------------------------------------|-------------------------------------|--------------|
| WRF-Chem coupled chemistry/aerosols        | `chem_opt`                          | `chem_opt=0` |
| WRF-Fire (SFIRE) wildfire spread           | `ifire`                             | `ifire=0`    |
| WRF-Hydro hydrological coupling            | `wrf_hydro`                         | disable coupler |
| FDDA analysis/obs/surface nudging          | `grid_fdda`, `obs_nudge_opt`, `grid_sfdda` | `=0` |
| Stochastic physics                         | `sppt`, `skebs`, `spp`, `rand_perturb`, `stoch_force_opt` | `=0` |
| Moving / vortex-following nests            | `vortex_interval`, `num_moves`      | static nest only |
| Multi-layer urban canopy (UCM/BEP/BEM)     | `sf_urban_physics` 1/2/3            | `sf_urban_physics=0` |
| Wind-farm / turbine-drag parameterization  | `windfarm_opt`                      | `windfarm_opt=0` |
| Coupled ocean mixed-layer / 3-D ocean      | `sf_ocean_physics`                  | `sf_ocean_physics=0` |
| Time-varying SST lower-boundary update     | `sst_update`                        | `sst_update=0` |

## Bring-your-WRF-namelist transition recipe

To run an existing real-data WRF `namelist.input` on the port today:

* Physics suite: pick from the **`[OPERATIONAL]`** entries in the tables above
  (the operational default is `mp_physics=8` Thompson, `bl_pbl_physics=5` MYNN,
  `sf_sfclay_physics=5` MYNN-SL, `cu_physics=0` no-cumulus, `sf_surface_physics=4`
  Noah-MP + `use_noahmp`, `ra_lw_physics=4` / `ra_sw_physics=4` RRTMG). A
  **`[REFERENCE-ONLY]`** scheme (Grell-3D `cu=5`, KSAS `cu=14`, New-Tiedtke
  `cu=16`, slab LSM `sf_surface=1`, GSFC/Goddard LW `ra_lw=5`) is **rejected by
  `gpuwrf run`** — it is for reference comparisons only; the error names the
  operational swap.
* Turbulence/diffusion: WRF's recommended real-data defaults `diff_opt=1`,
  `km_opt=4` (2-D Smagorinsky) **run as-is** (see the honesty note below). If you
  use a 3-D closure (`km_opt=2/3/5`), switch to **constant-K: `diff_opt=2`,
  `km_opt=1`** (or `diff_opt=0`).
* Turn off any out-of-scope switch in the table above (`chem_opt=0`,
  `grid_fdda=0`, `sppt=0`, `sf_urban_physics=0`, `sf_ocean_physics=0`, ...).

## Known compatibility limitations (honest)

* **No bit-identity claim.** This is a WRF-compatible reimplementation, not a
  bit-true Fortran port; expect operational (RMSE-class) agreement, not byte
  equality. The dominant open fidelity gap is the **24 h forecast-skill
  equivalence (T2/U10/V10) vs CPU-WRF** (KNOWN_ISSUES KI-9); see that file for
  the full carry-over list.
* **2-D Smagorinsky vs 3-D closures.** WRF's recommended real-data turbulence
  settings `diff_opt=1` (2nd-order on coordinate surfaces) + `km_opt=4`
  (horizontal Smagorinsky) **are implemented and operationally wired** (WRF
  `smag2d_km`, parity in `proofs/v090/diffopt1_smagorinsky_parity.json`), as is
  the constant-K path `diff_opt=2`/`km_opt=1`. The Smagorinsky parity scope is
  the documented idealized-slab reduction (unit map factors, flat-eta
  `zx=zy=0`); the slope-correction branch is gated on `diff_opt==2`. The 3-D
  closures `km_opt=2` (3-D TKE), `km_opt=3` (3-D Smagorinsky) and `km_opt=5`
  (SMS-3DTKE) are **not** implemented and fail closed — transition recipe:
  **switch to constant-K `diff_opt=2 / km_opt=1`** (vertical mixing then comes
  from the PBL scheme).
* The scheme catalog reflects **WRF v4** option codes
  (`WRF/run/README.namelist`); re-audit on a WRF version bump using the
  per-entry `source_lines` references in `wrf_scheme_catalog.py`.
* The validator checks *selected* options, not namelist completeness. It does
  not (yet) validate non-physics consistency (timestep CFL, domain ratios, IO
  intervals); those remain the user's responsibility / a downstream concern.

## Where this lives

* `src/gpuwrf/io/wrf_scheme_catalog.py` — full WRF v4 code->name enumeration.
* `src/gpuwrf/io/scheme_catalog.py` — the machine-readable five-status support
  catalog (`implemented` / `reference_only` / `recognized_approximated` /
  `recognized_fail_closed` / `out_of_scope`), the public honesty contract.
  `_IMPLEMENTED` is the OPERATIONAL ground truth (mirrors
  `operational_mode._SCAN_WIRED_OPTIONS`); `_REFERENCE_ONLY` is the
  accepted-but-not-operational set. `classify_scheme()`,
  `classify_feature_switch()`, `classify_control()`, `iter_full_catalog()`,
  `status_counts()`, and `assert_catalog_consistent()` (the anti-over-claim
  invariants vs the frozen accept-matrix and the operational scan-wired set).
* `src/gpuwrf/io/namelist_check.py` — fail-closed entrypoints:
  `validate_namelist()` (the validation layer: scheme support + out-of-scope
  features; **accepts** `reference_only` schemes so reference comparisons can run)
  and `validate_operational_namelist()` (the **operational** layer: runs
  `validate_namelist()` then *additionally* rejects `reference_only` selections
  via `classify_scheme`/`SupportStatus` — raising `NotOperationallyWiredError`).
  The `gpuwrf run` CLI calls `validate_operational_namelist()` before any heavy
  import, so an operational forecast can never silently substitute a different
  scheme. `validate_supported_namelist()` remains the scheme-only checker they
  compose; `collect_namelist_warnings()` surfaces the `cudt`/`bldt`
  every-step approximation warnings.
* `src/gpuwrf/coupling/physics_dispatch.py` — the operational dispatcher:
  `resolve_physics_suite()` (fail-closed `(family, option) -> scheme` routing,
  MYJ-pairing enforcement) and `dispatch_matrix()` (one row per routable option,
  each carrying `gpu_runnable`). Some reference-only options carry a routable row
  flagged `gpu_runnable=False` (New-Tiedtke `cu=16`, slab `sf_surface=1`); others
  have no routable entry at all (Grell-3D `cu=5`, KSAS `cu=14`). Either way the
  authoritative OPERATIONAL discriminator is `scheme_catalog._IMPLEMENTED`, not
  dispatch-matrix presence.
* `src/gpuwrf/contracts/physics_registry.py` — the authoritative
  accepted matrix (`ACCEPTED_*`) and the per-scheme field/carry registry.
* `src/gpuwrf/contracts/physics_interfaces.py` — the frozen per-scheme
  `PhysicsStepSpec` adapter contracts (`SCHEME_STEP_SPECS`); note these specs
  exist for *every accepted* option (including reference-only), so spec presence
  is **not** the operational discriminator — `scheme_catalog._IMPLEMENTED` is.
* `tests/test_namelist_check.py`, `tests/test_scheme_catalog_fail_closed.py` —
  the compatibility test suites.
* [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — the open issues and deferred
  carry-overs (24 h skill gap, scheme long-tail, VRAM limits).
