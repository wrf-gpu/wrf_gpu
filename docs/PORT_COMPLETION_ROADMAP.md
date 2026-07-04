# Port-completion roadmap — closing every gap toward a complete WRF v4 port

**Goal (v0.24 and beyond): make the support matrix all-green.** Today the operational
scan wires **51 physics-scheme codes**; **25 are reference-only** (a WRF oracle is staged
but no JAX kernel is scan-wired yet) and **32 fail closed** (recognized WRF option, refused
before any compute). This file is the *complete, code-grounded inventory* of everything that
is **not yet operational** — every scheme, every dynamics closure (3-D TKE / 3-D Smagorinsky /
SMS-3DTKE), and every out-of-scope feature — so that successive releases can drive the
**"Fail-closed" and "Reference-only" columns to zero**.

> This roadmap is generated against the authoritative scheme registry
> (`src/gpuwrf/io/scheme_catalog.py` + `contracts/physics_registry.py`); reproduce the live
> lists any time with `python -m gpuwrf.cli namelist-support --json`. It complements
> [`WRF_PARITY_ROADMAP.md`](WRF_PARITY_ROADMAP.md) (which covers CLI *usage* parity) and the
> code-task inventory [`GPU_PORT_GAPS_TODO.md`](GPU_PORT_GAPS_TODO.md).

## Scoreboard

| Status | Count | Meaning | v0.24 target |
|---|---:|---|---|
| 🟢 **Operational** | 51 codes | Scan-wired, WRF-oracle-gated, runs a real forecast | grow |
| 🟡 **Reference-only** | 25 codes | WRF oracle staged; **needs a JAX kernel + scan wiring** (the cheapest graduations) | → 0 |
| 🔴 **Fail-closed** | 32 codes | Recognized WRF option, refused with a named reason; **needs oracle + kernel + wiring** | shrink hard |
| ⚫ **Out-of-scope** | features | Whole subsystems deliberately not ported (Chem/Fire/Hydro/DA/…) | scope decisions per subsystem |

## v0.24 roadmap at a glance (top level)

Effort: **S** ≈ 1–2 sprints · **M** ≈ 3–5 · **L** ≈ 5–10 · **XL** ≈ 10+. Full per-scheme detail
is in the [physics-scheme tables](#physics-schemes-not-yet-operational-the-full-list) and the
sections below.

| # | Area | Non-operational now | Close it by | Tier | Effort |
|---|---|---|---|---|---|
| 1 | **Microphysics** `mp_physics` | 🟡2 + 🔴21 = 23 | Wire mp=18 (NSSL), mp=40 (Morrison-aero) — oracles exist; then the deep families (P3 50–53, Milbrandt-Yau 9, HUJI SBM 30/32, Jensen-ISHMAEL 55, NTU 56, …) | A→C | L–XL |
| 2 | **Cumulus** `cu_physics` | 🟡8 + 🔴3 = 11 | Wire the SAS family (4/94/95/96/14), Grell 3D/GD (5/93) — oracles exist; then Zhang-McFarlane 7, MSKF 10/11 | A→C | M–L |
| 3 | **PBL** `bl_pbl_physics` | 🟡5 | Wire QNSE 4, UW/CAM5 9, TEMF 10, TKE-eps 16/17 (oracles exist); pair 4/10 with their surface layers | A | M each |
| 4 | **Surface layer** `sf_sfclay_physics` | 🔴2 | Stage oracle + port QNSE (4) & TEMF (10); graduate with their PBLs | A | M |
| 5 | **Land surface** `sf_surface_physics` | 🟡2 + 🔴2 = 4 | Wire RUC 3, SSiB 8 (oracles exist); CLM4 5 / CTSM 6 = architecture boundary (needs the CLM/CTSM column model) | A / D | M / XL |
| 6 | **Longwave radiation** `ra_lw_physics` | 🟡4 + 🔴2 | Wire CAM 3, Goddard 5, FLG 7, GFDL 99 (oracles exist); RRTMG-K 14 / fast-RRTMG 24 build-gated in WRF too (low prio) | A | M–L |
| 7 | **Shortwave radiation** `ra_sw_physics` | 🟡4 + 🔴2 | Same as longwave (CAM/Goddard/FLG/GFDL wire; 14/24 build-gated) | A | M–L |
| 8 | **Dynamics — 3-D closures** | 🔴 `km_opt=2/3/5` | Port **3-D TKE** (km_opt=2), **3-D Smagorinsky** (km_opt=3), **SMS-3DTKE** (km_opt=5) — self-contained dycore, no tables | B | M each |
| 9 | **Nesting** | two-way / moving / global / adaptive-Δt | Prove two-way 24 h equivalence (KI-11); validate the moving-nest driver; global/periodic BC; CFL adaptive-Δt | B / D | M–XL |
| 10 | **Data assimilation** | DFI, FDDA (grid/obs/surface), spectral nudging | Port the nudging tendencies + DFI integrator (only lateral-BC relaxation exists today) | D | L–XL |
| 11 | **Coupled subsystems** | Chem, Fire, Hydro, urban (UCM/BEP/BEM), lake, ocean, wind-farm, stochastic, SST-update | Scope decisions per subsystem; urban BEP/BEM + lake are 🟡 (v0.23 oracles); several plausibly **v1.0** | D | XL |
| 12 | **Output & grids** | full 375-var wrfout (🟡 opt-in), auxhist, projections | Make the 375-var stream a validated option (KI-3); aux stream writers; add rotated/global projections | C | M |
| 13 | **Meta-gate — forecast skill (KI-9)** | 24–72 h T2/U10/V10 equivalence | Hard dynamics-`ph'` / MYNN / `*_tendf` work — **gates any "complete port" claim** even after the columns are green | meta | L |

**How to read it:** Tier A (rows 1–7 reference-only) is the cheapest, highest-value wave — the
WRF oracles are already staged, so each is "port the JAX kernel + scan-wire it." Tier B (row 8 +
parts of 9) is bounded dycore work. Tiers C/D are the deep multi-moment microphysics and the
coupled subsystems (several = v1.0). Row 13 is the honest caveat: an all-green matrix is
necessary but not sufficient — the skill gate is the real credibility bar.

## The closure recipe (how a gap graduates)

Every scheme follows the project's validation pyramid; nothing is wired without an oracle.

1. **Stage a WRF oracle** — run the unmodified WRF Fortran scheme as a single-column /
   savepoint reference (only needed for 🔴 fail-closed; 🟡 reference-only already has this).
2. **Port a JAX column kernel** and prove it to **machine precision** against that oracle.
3. **Scan-wire it** into `operational_mode.py` / `scan_adapters.py` and add the namelist
   catalog entry (`implemented`).
4. **Cell-identity gate** it against CPU-WRF on a real case.

So: **🟡 reference-only = steps 2–4 remain (fastest wins).  🔴 fail-closed = steps 1–4.**

## Priority tiers toward v0.24

- **Tier A — graduate the reference-only tail (25 codes).** Oracles already exist; each is
  "port the kernel + wire it." Highest value-per-effort. Start with the ones with the widest
  operational demand: **RUC LSM (`sf=3`)**, **NSSL 2-moment (`mp=18`)**, **Morrison-aerosol
  (`mp=40`)**, **CAM-UW PBL (`bl=9`)**, the **SAS cumulus family (`cu=4/94/95/96/14`)**, and
  the **CAM/Goddard/FLG radiation** set. (mp=18, bl=9, urban/lake already have staged oracles
  from v0.23.)
- **Tier B — the 3-D turbulence closures (dynamics, self-contained).** `km_opt=2` (3-D TKE),
  `km_opt=3` (3-D Smagorinsky), `km_opt=5` (SMS-3DTKE). No external tables; a bounded dycore
  task. See [Dynamics](#dynamics--diffusion-closures) below.
- **Tier C — the deep microphysics families.** P3 (`mp=50–53`), Milbrandt-Yau (`9`),
  spectral-bin HUJI (`30/32`), Jensen-ISHMAEL (`55`), NTU (`56`) — large, multi-moment
  kernels; each is its own milestone.
- **Tier D — coupled subsystems (scope decisions / v1.0).** CLM4/CTSM land, WRF-Chem,
  WRF-Fire, WRF-Hydro, urban BEP/BEM, lake — architectural, see
  [Out-of-scope subsystems](#out-of-scope-subsystems).
- **Meta-gate that gates any "complete" claim: the 24–72 h forecast-**skill** equivalence
  (KI-9).** Making the matrix all-green is necessary but not sufficient; the skill gate is the
  credibility gate.

---

## Physics schemes not yet operational (the full list)

Regenerate with `python -m gpuwrf.cli namelist-support` (add `--json` for machine-readable).

#### Microphysics — `mp_physics` (2 reference-only, 21 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `5` | Ferrier (new Eta), HRW | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `7` | Goddard 4-ice | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `9` | Milbrandt-Yau 2-moment | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `11` | CAM 5.1 microphysics | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `17` | NSSL (legacy) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `18` | NSSL 2-moment 4-ice w/ predicted CCN | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `19` | NSSL (legacy) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `21` | NSSL (legacy) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `22` | NSSL (legacy) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `27` | UDM 7-class | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `29` | RCON (Thompson aerosol-aware, liquid-phase mods) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `30` | HUJI spectral-bin (fast) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `32` | HUJI spectral-bin (full) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `38` | Thompson w/ 2-moment graupel/hail | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `40` | Morrison 2-moment w/ CESM-NCSU RCP4.5 aerosol | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `50` | P3 1-ice, 1-moment cloud water | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `51` | P3 1-ice + double-moment cloud water | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `52` | P3 2-ice + double-moment cloud water | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `53` | P3 1-ice 3-moment + double-moment cloud water | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `55` | Jensen-ISHMAEL | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `56` | NTU multi-moment | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `95` | Ferrier (old Eta) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `96` | Madwrf | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |

#### Cumulus — `cu_physics` (8 reference-only, 3 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `4` | Scale-aware GFS SAS | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `5` | Grell 3D ensemble | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `7` | Zhang-McFarlane (CAM5) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `10` | Modified Kain-Fritsch (PDF trigger) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `11` | Multi-scale Kain-Fritsch (MSKF) | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `14` | KIM Simplified Arakawa-Schubert (KSAS) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `93` | Grell-Devenyi ensemble | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `94` | 2015 GFS SAS (HWRF) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `95` | previous GFS SAS (HWRF) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `96` | previous new GFS SAS (YSU) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `99` | previous Kain-Fritsch | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |

#### PBL — `bl_pbl_physics` (5 reference-only, 0 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `4` | QNSE-EDMF (Quasi-Normal Scale Elimination) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `9` | UW (CAM5) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `10` | TEMF (Total Energy Mass Flux) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `16` | TKE + TKE-dissipation (epsilon) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `17` | TKE + TKE-dissipation + TPE | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |

*Note: `bl=4` (QNSE) and `bl=10` (TEMF) require their paired surface-layer schemes
`sf_sfclay_physics=4/10`, which are 🔴 fail-closed below — graduate the pairs together.*

#### Surface layer — `sf_sfclay_physics` (0 reference-only, 2 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `4` | QNSE surface layer | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |
| `10` | TEMF surface layer | 🔴 fail-closed | stage a WRF single-column oracle, then port + wire a JAX kernel |

#### Land surface — `sf_surface_physics` (2 reference-only, 2 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `3` | RUC LSM | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `5` | CLM4 (Community Land Model v4) | 🔴 fail-closed | documented architecture boundary — needs the full CLM4 column model (XL / v1.0) |
| `6` | CTSM (Community Terrestrial Systems Model) | 🔴 fail-closed | documented architecture boundary — needs the CTSM column model (XL / v1.0) |
| `8` | SSiB (Simplified Simple Biosphere) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |

#### Longwave radiation — `ra_lw_physics` (4 reference-only, 2 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `3` | CAM | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `5` | Goddard longwave | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `7` | FLG (UCLA) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `14` | RRTMG-K (KIAPS) | 🔴 fail-closed | build-gated in WRF too (compiled-out); low priority |
| `24` | fast RRTMG (GPU/MIC) | 🔴 fail-closed | build-gated in WRF too (compiled-out); low priority |
| `99` | GFDL (Eta) longwave | 🟡 reference-only | wire a scan JAX kernel (ETARA source/tables located; no standalone oracle yet) |

#### Shortwave radiation — `ra_sw_physics` (4 reference-only, 2 fail-closed)

| Code | Scheme | Status | To close |
|---|---|---|---|
| `3` | CAM | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `5` | Goddard shortwave (new) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `7` | FLG (UCLA) | 🟡 reference-only | wire a scan JAX kernel (WRF oracle already staged) |
| `14` | RRTMG-K (KIAPS) | 🔴 fail-closed | build-gated in WRF too (compiled-out); low priority |
| `24` | fast RRTMG (GPU/MIC) | 🔴 fail-closed | build-gated in WRF too (compiled-out); low priority |
| `99` | GFDL (Eta) | 🟡 reference-only | wire a scan JAX kernel (ETARA source/tables located; no standalone oracle yet) |

---

## Dynamics — diffusion closures

The operational dynamics wires RK3, split-explicit acoustics, flux-form advection, and the
horizontal diffusion closures **constant-K (`diff_opt=2`/`km_opt=1`)** and **2-D Smagorinsky
(`diff_opt=1`/`km_opt=4`)**. The three **3-D closures fail closed** (transition recipe today:
switch to constant-K `diff_opt=2`/`km_opt=1`):

| Option | Closure | Status | To close |
|---|---|---|---|
| `km_opt=2` | 3-D TKE (prognostic TKE sub-grid mixing) | 🔴 fail-closed | port the 3-D TKE closure + its prognostic TKE equation; oracle vs WRF `module_em` diffusion |
| `km_opt=3` | 3-D Smagorinsky | 🔴 fail-closed | extend the 2-D Smagorinsky kernel to the full 3-D deformation tensor |
| `km_opt=5` | SMS-3DTKE (scale-aware 3-D TKE / LES-PBL) | 🔴 fail-closed | port the SMS-3DTKE scheme (couples sub-grid TKE with the PBL) |

Also parked at the dynamics boundary: `diff_6th_opt` beyond `0/2`, `damp_opt` beyond `0/3`,
non-RK3 `rk_order`, and higher `w_damping` variants — each a bounded dycore task.

## Nesting

| Capability | Status | To close |
|---|---|---|
| One-way live d01→d02→d03 | 🟢 operational | — |
| Two-way feedback (`--feedback`) | opt-in, finite/stable | prove **24 h equivalence vs CPU-WRF** (KI-11) |
| Moving / vortex-following nests (`vortex_interval`, `num_moves`) | opt-in driver (v0.23) | validate the moving-nest path against CPU-WRF; adaptive re-mesh |
| Adaptive time-step (`use_adaptive_time_step`) | ⚫ out-of-scope | port the CFL-driven Δt controller |
| Global / periodic nests | ⚫ out-of-scope | polar/periodic BC + global grid |
| **Wide open-boundary 3 km-nest stability** (Ni boundary-NaN class, KI-7) | 🔴 open dycore-boundary | Large open-ocean lateral boundaries (nx≈160+, e.g. all-islands 268×118) can drive Thompson `Ni` non-finite in the boundary-relaxation zone beyond ~14–20 h (finite guard catches it, 0 bad frames). Root-cause the boundary-zone acoustic mass-pump for this geometry class (same family as the Canary-d03 fix, re-exposed by new geometry); levers `GPUWRF_NORMAL_BDY_RELAX_STRENGTH` / `GPUWRF_SPECIFIED_ADV_DEGRADE` are diagnostics, not a validated fix. Reproducer: CPU-vs-GPU at the crash window. |

## Data assimilation

| Capability | Switch | Status | To close |
|---|---|---|---|
| Digital-filter init (DFI) | `dfi_opt` | ⚫ out-of-scope | port the DFI launcher/back-forward integrator |
| Grid / analysis nudging (FDDA) | `grid_fdda` | ⚫ out-of-scope | port the nudging tendency terms |
| Observation nudging | `obs_nudge_opt` | ⚫ out-of-scope | obs ingest + nudging |
| Surface analysis nudging | `grid_sfdda` | ⚫ out-of-scope | surface FDDA |
| Spectral nudging | `spec_nudging` | ⚫ out-of-scope | spectral filter + nudging |

Today only **lateral-boundary relaxation** is implemented (the operational forcing path).

## Out-of-scope subsystems

Whole WRF subsystems deliberately not ported yet — each is a scope decision (several are
plausibly **v1.0** boundaries). All fail closed with a named reason and a disable recipe.

| Subsystem | Switch(es) | Note |
|---|---|---|
| Urban canopy — single-layer UCM | `sf_urban_physics=1` | fail-closed |
| Urban canopy — BEP / BEM | `sf_urban_physics=2/3` | 🟡 reference-only (v0.23 oracle staged) |
| Lake model | `sf_lake_physics=1` | 🟡 reference-only (v0.23 oracle staged) |
| WRF-Chem (coupled chemistry/aerosols) | `chem_opt` | large coupled subsystem |
| WRF-Fire (SFIRE) | `ifire` | coupled fire-spread |
| WRF-Hydro | `wrf_hydro` | hydrological coupling |
| Stochastic physics | `sppt`, `skebs`, `spp`, `rand_perturb` | ensemble perturbation |
| Wind-farm / turbine drag | `windfarm_opt` | drag parameterization |
| Coupled ocean mixed-layer / 3-D ocean | `sf_ocean_physics` | ocean coupling |
| Time-varying SST update | `sst_update` | lower-boundary update |

## Output & grids

| Item | Status | To close |
|---|---|---|
| Focused 104-variable `wrfout` | 🟢 default | — |
| Full 375-variable `wrfout` | 🟡 opt-in (`GPUWRF_FULL_WRFOUT=1`) | make the full stream a validated default option (KI-3) |
| Auxiliary history streams (`auxhist*`) | ⚫ out-of-scope | port the aux stream writers |
| Map projections | Lambert / Mercator / Polar + hybrid-eta C-grid | add rotated-lat-lon / Cassini / global |

## Throughput — batched ensemble

| Capability | Status | To close |
|---|---|---|
| Replicated batch (same input, B lanes) | 🟢 operational | — |
| **Same-date, distinct-IC** batch (`GPUWRF_BATCH_INPUT_DIRS`) | 🟢 operational (v0.23.2) | — |
| **Multi-date** batch (different `start_date` per lane) | 🔴 not supported | Thread the date/time as a **per-lane runtime leaf** instead of namelist static-aux, so distinct dates share one `vmap` program (today the date is part of the compiled-program key → lanes with different dates fail the homogeneity gate `namelist/static treedef differs`). Requires a numerics-preserving re-plumb + validation. Until then, run multi-day sequentially. |

## The meta-gate — forecast-skill equivalence (KI-9)

Closing the columns above makes the **coverage** complete; it does **not** by itself close the
**24–72 h forecast-skill equivalence** vs CPU-WRF (T2/U10/V10), which remains the project's
open credibility gate (dominated by lead-time wind divergence; hard dynamics-`ph'` / MYNN /
`*_tendf` work). A "complete WRF v4 port" claim requires **both**: an all-green support matrix
**and** the skill gate closed. See the [User's Guide → Validation](https://wrf-gpu.github.io/wrf_gpu/validation.html).

---

*This inventory is regenerated from the live registry — if a scheme's status changes, rerun
`python -m gpuwrf.cli namelist-support --json` and update the scoreboard. The intent is that
each release moves codes up the ladder (fail-closed → reference-only → operational) until the
non-operational columns are empty.*
