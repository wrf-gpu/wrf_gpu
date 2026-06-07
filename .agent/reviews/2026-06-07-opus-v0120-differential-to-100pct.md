# v0.12.0 → "100% complete WRF v4 ARW port" gap analysis

**Author:** Opus 4.8 MAX-effort, skeptical-WRF-developer mindset. Read-only.
**Trunk inspected:** `/home/enric/src/wrf_gpu2/.claude/worktrees/v0120-integration` (branch `worker/opus/v0120-integration`).
**Ground truth used (NOT aspirational docs):**
- `src/gpuwrf/io/scheme_catalog.py` (the honesty contract) + `src/gpuwrf/io/wrf_scheme_catalog.py` (full WRF v4 enumeration).
- `src/gpuwrf/contracts/physics_registry.py` (`ACCEPTED_*` accept-matrix).
- `src/gpuwrf/runtime/operational_mode.py` `_SCAN_WIRED_OPTIONS` / `_SCAN_UNWIRED_REASON` (what is genuinely threaded into the GPU scan).
- `src/gpuwrf/coupling/scan_adapters.py` (MP/CU/PBL/SFCLAY scan-adapter registries).
- `src/gpuwrf/io/wrfout_writer.py` (85 unique vars the writer knows; verified).
- Real reference wrfout: `/mnt/data/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_.../wrfout_d02_*` → **368 distinct variables** (the "375" claim is in the right ballpark; this file has 368). Differential computed: writer emits **74 of those 368**; **294 are absent**.

**Cross-check result (catalog is honest):** `_SCAN_WIRED_OPTIONS` exactly matches `scheme_catalog._IMPLEMENTED`; `assert_catalog_consistent()` enforces `implemented ∪ reference_only == accepted` per key. One cosmetic drift worth noting: `physics_registry.ACCEPTED_RA_SW/LW_PHYSICS` lists `(0,1,4)` and `SchemeOption(...,1,..."implemented"...)` for Dudhia/RRTM, but `scheme_catalog` correctly demotes code `1` to REFERENCE_ONLY and the operational scan only wires `{0,4}`. The catalog (not the registry `status` string) is the binding statement, and it is honest. **No over-claim found in the public catalog.**

---

# PART 1 — Cheap-but-skipped completeness wins (ranked by value × feasibility)

Effort key: **S** ≤ half-day, **M** ~1–2 days, **L** ~3–5 days. "Tonight?" = good candidate for a few-hours push given no-GPU / CPU-only and frozen interfaces.

### TIER A — do tonight (CPU-only, near-zero risk, pure I/O / data-routing)

| # | Win | What it is | Why skipped | Effort | GPU? | Risk | Tonight? |
|---|-----|-----------|-------------|--------|------|------|----------|
| **A1** | **Vertical-coordinate + grid-metric static vars in wrfout** | ~30 WRF vars that are *already computed device-resident static arrays* in `DycoreMetrics`/grid and just not routed to the writer: `DN, DNW, RDN, RDNW, CFN, CFN1, CF1, CF2, CF3, FNM, FNP, C1F, C1H, C2F, C2H, C3F, C3H, C4F, C4H, RDX, RDY, ZS, DZS, DNW`-family, `P_TOP`(already)/`P00, T00, TLP, TISO, TLP_STRAT, P_STRAT, ZETATOP`, `MAPFAC_MX/MY, MAPFAC_UX/UY, MAPFAC_VX/VY` (x/y split of the MAPFAC_M/U/V the writer already emits), `MAX_MSFTX/Y`. These let any downstream WRF tool reconstruct the eta coordinate + projection without a separate wrfinput read. | Writer was deliberately "focused subset"; these are non-meteorological so they were never load-bearing for skill gates. | **S** | No | **Very low** — pure copy of existing static arrays into the payload + `_spec` metadata entries. No new physics, no new compute. | **YES — best tonight pick** |
| **A2** | **Trivially-derived near-surface diagnostics** | `TH2` (already in writer), add `RH2` (from `Q2,T2,PSFC`), `Q2B/Q2V/T2B/T2V` (Noah-MP land/veg split — emit `T2/Q2` copies or skip), `SNOWC` (snow cover fraction from `SNOWH`), `SR` (solid-precip fraction from snow_acc/rain_acc increments), `CLAT` (= XLAT copy), `COSZEN` (already computed in `physics_couplers._compute_coszen`!). | Considered "nice-to-have" diagnostics (P2-3). | **S** | No | **Low** — each is one closed-form expression from fields already present. RH2/SNOWC/SR are standard WRF diagnostics. | **YES** |
| **A3** | **Fix the stale "64-variable" claim everywhere** | README/KNOWN_ISSUES/RELEASE_NOTES say "64-variable writer"; the code actually knows **85** (74 of which overlap the 368-var reference). KI-3 text is simply wrong about the count and about "missing only stochastic-seed + Noah-MP snow-layer" (in fact 294 are missing). | Doc lag across versions. | **S** | No | **None** (doc-only) but **credibility-relevant** — a skeptical reviewer will diff the writer and catch this. | **YES — do alongside A1/A2** |

> A1+A2 together close ~35–45 of the 294 missing vars with essentially no correctness risk, taking wrfout coverage from 74/368 (20%) to ~110–115/368 (~31%), and make KI-3 honest. This is the single highest value-per-hour item.

### TIER B — strong v0.13 candidates (modest effort, real completeness gain)

| # | Win | What / why skipped | Effort | GPU? | Risk | Tonight? |
|---|-----|--------------------|--------|------|------|----------|
| **B1** | **Radiation flux-boundary diagnostics** | `SWDNB, SWDNBC, SWUPB, SWUPBC, LWDNB, LWDNBC, LWUPB, LWUPBC, SWDNT(C), SWUPT(C), LWDNT(C), LWUPT(C), OLR, SWNORM, ALBEDO`(emitted), `COSZEN`. The RRTMG SW/LW columns *already compute full up/down flux profiles*; these are just the top/bottom-of-column slices, not new physics. ~16 vars. | Only `SWDOWN/GLW` (surface down) were routed. | **M** | No (extract from existing radiation diag struct; the radiation step is GPU but the values already exist) | **Low–Med** — need to thread the flux profiles out of `rrtmg_radiation_diagnostics` to the writer; values exist, plumbing only. | No (needs a careful plumb + a parity check vs reference) |
| **B2** | **RRTMG SW `taug` UV-band fix (KI-6)** | `_extend_with_wrf_top_layer` (`rrtmg_sw.py:589`) duplicates the topmost ~190 hPa layer instead of inserting a ~100 Pa model-top layer, so the intermediate `taug` for 4 UV bands (jp≈12–14 O3/O2) is wrong. **Integrated fluxes already pass <0.05%** — this is an *intermediate-value* fidelity fix, not a forecast bug. Two fix paths documented: (A) regenerate oracle at current convention; (B) implement correct top-layer pressure. | Masked by a fixture-version mismatch; low forecast impact. | **M** | Yes (re-run RRTMG savepoint gate) | **Med** — touching the radiation top-layer convention risks the integrated-flux gate; must re-prove. | No (needs GPU + careful re-validation) |
| **B3** | **Noah-MP snow-layer diagnostics (KI-3 named gap)** | `TSNO, SNICE, SNLIQ` (`snow_layers_stag=3`), `ZSNSO` (`snso_layers_stag=7`), plus `SNEQVO, ISNOW, CANLIQ, CANICE, SNOWENERGY, SOILENERGY`. The prognostic Noah-MP carry (`NoahMPLandState`) already holds most of these internally; the writer has the var names listed (`LAND_SNOW_DIAGNOSTIC_VARIABLES`) but they self-gate to absent unless the carry is plumbed. ~10 vars. | Self-gating means "absent rather than fabricated"; routing was deferred. | **M** | No | **Low–Med** — need to confirm the carry actually holds each layer field and add the new stagger dims to the writer. | Maybe (S if carry already has them; verify first) |
| **B4** | **Stochastic-seed arrays (KI-3 named gap)** | `ISEEDARR_SPPT, ISEEDARR_SKEBS, ISEEDARRAY_SPP_*, ISEEDARR_RAND_PERTURB`. Writer already lists these (`STOCHASTIC_SEED_VARIABLES`) and they self-gate. Since stochastic physics is OUT_OF_SCOPE, these would be zero/empty arrays — emit only as empty placeholders for downstream-tool compatibility. | Stochastic physics deliberately out of scope. | **S** | No | **Low** — but arguably *should stay absent* (emitting empty seed arrays for an unimplemented feature is borderline dishonest). Recommend: document as intentionally-absent, do NOT fabricate. | No (recommend NOT doing) |
| **B5** | **Namelist recognition breadth** | The namelist checker (`SUPPORTED_OPTIONS`) gates the 7 physics groups + the dynamics keys. Adding *recognition* (fail-closed-with-reason, not implementation) for more WRF keys — `gwd_opt`, `moist_adv_opt`, `scalar_adv_opt`, `topo_shading`, `slope_rad` (latter two are actually implemented), `icloud_bl`, `bl_mynn_*`, `windfarm_opt`, `bldt/cudt/radt` cadence keys — so a real WRF namelist fails fast with named reasons instead of silently ignoring. | `OperationalNamelist` is a runtime dataclass, not a full namelist schema (P2-2). | **M** | No | **Low** — pure validator additions referencing the existing catalog. Strong credibility win (a WRF user can throw a real namelist at it and get honest per-key verdicts). | Partly (S for a first batch of keys) |

### TIER C — reference-only **scan-wiring** (the documented v0.13 deferral; per-scheme effort)

All four reference-only families already have **parity-proven column kernels present** in `src/gpuwrf/physics/` — wiring is "wrap the existing kernel as a jit/vmap `jax.lax.scan` State→State adapter + add the carry leaves + register in `scan_adapters` + flip `_SCAN_WIRED_OPTIONS`", **not** a from-scratch port. This is exactly the pattern already used for YSU/ACM2/GF/Tiedtke in v0.6.0/v0.9.0.

| Scheme | Code | Kernel present | What scan-wiring needs | Effort | GPU? | Risk |
|--------|------|----------------|------------------------|--------|------|------|
| **Dudhia SW** | `ra_sw_physics=1` | `physics/ra_sw_dudhia.py` | Radiation-family dispatch slot + held-cadence wire (same cadence machinery RRTMG uses). Simplest of the four — broadband, no gas-table coupling. | **M** | Yes (savepoint re-gate) | Low–Med |
| **Classic RRTM LW** | `ra_lw_physics=1` | `physics/ra_lw_rrtm.py` | Same radiation dispatch slot as Dudhia; host-NumPy single-column kernel needs the jit/vmap rewrite (the `_SCAN_UNWIRED_REASON` names this). | **M–L** | Yes | Med |
| **MYJ PBL + Janjic-Eta sfclay** | `bl_pbl_physics=2` + `sf_sfclay_physics=2` (paired) | `physics/pbl_myj.py`, `physics/sfclay_janjic.py`, `myj_constants.py` | TKE-carry path (`tke_pbl, el_pbl` already in registry `PBL_CARRY_MEMBERS[2]`) + scan adapter for the paired sfclay. Must wire the **pair** together (Janjic Eta sfclay must pair with MYJ PBL). | **L** | Yes | Med |
| **New-Tiedtke cumulus** | `cu_physics=16` | shares Tiedtke interface (`cu=6` is wired) | Needs a *distinct* savepoint gate vs modified-Tiedtke + GPU-batching; the only one without its own parity proof yet. | **L** | Yes | Med–High |

> Recommendation: Dudhia SW (C, smallest) is the cleanest single reference-only scheme to scan-wire in a focused v0.13 sprint to demonstrate the radiation-family dispatch generalizes. The others follow the same template.

### TIER D — cheap *honesty/credibility* fixes (not features, but tonight-able)

- **D1 (S, tonight):** Fill the two `<<MANAGER-FILL>>` placeholders left in `docs/KNOWN_ISSUES.md` (the "Deferred to v0.13.0" section, lines ~250–251: standalone nested 24 h 1 km skill proof, and the powered n=15 TOST status). A skeptical reviewer will spot raw `<<MANAGER-FILL>>` tokens in a shipped doc immediately.
- **D2 (S, tonight):** KI-8 brittle source-pattern tests — either xfail-mark them or update the patterns. They are test-hygiene noise that makes the suite look red for no real reason.

---

# PART 2 — The differential-to-"100%" feature inventory

Classification of **every** WRF v4 ARW feature/option, EXCLUDING the genuinely-nonsensical-for-single-GPU-JAX set (MPI `nproc_x/nproc_y`, quilting / I/O servers `nio_tasks_per_group`, DM-parallel build flags, `tile`/patch decomposition). For each enumerated physics/dynamics group I give the per-code verdict from the catalog ground truth, then a MISSING roadmap by subsystem.

## 2.1 Enumerated physics/dynamics groups (per-code, from the catalogs)

**Microphysics `mp_physics`** — IMPLEMENTED+scan-wired: `{0,1,2,3,4,6,8,10,16}` (no-MP, Kessler, Purdue-Lin, WSM3, WSM5, WSM6, Thompson, Morrison-2mom, WDM6).
GENUINELY-MISSING WRF v4 codes: `5` Ferrier, `7` Goddard-4ice, `9` Milbrandt-Yau, `11` CAM5.1, `13` SBU-YLin, `14` WDM5, `17/18/19/21/22` NSSL family, `24` WSM7, `26` WDM7, `27` UDM, `28` aerosol-aware Thompson, `29` RCON, `30/32` HUJI SBM, `38` Thompson-2mom-hail, `40` Morrison-CESM-aerosol, `50/51/52/53` P3 family, `55` Jensen-ISHMAEL, `56` NTU, `95/96/97` Ferrier-old/Madwrf/Goddard-GCE. **22 schemes missing.**

**Longwave `ra_lw_physics`** — IMPLEMENTED `{0,4}` (RRTMG); REFERENCE-ONLY `1` (RRTM, kernel present, not scan-wired).
MISSING: `3` CAM, `5` Goddard-LW, `7` FLG-UCLA, `14` RRTMG-K, `24` fast-RRTMG, `31` Held-Suarez, `99` GFDL-Eta. **7 missing + 1 ref-only.**

**Shortwave `ra_sw_physics`** — IMPLEMENTED `{0,4}` (RRTMG); REFERENCE-ONLY `1` (Dudhia, kernel present).
MISSING: `2` Goddard-SW, `3` CAM, `5` Goddard-SW-new, `7` FLG, `14` RRTMG-K, `24` fast-RRTMG, `99` GFDL. **7 missing + 1 ref-only.**

**Surface layer `sf_sfclay_physics`** — IMPLEMENTED `{0,1,5,7}` (none, revised-MM5, MYNN-SL, Pleim-Xiu); REFERENCE-ONLY `2` (Janjic-Eta, kernel present).
MISSING: `3` GFS-sfc, `4` QNSE-sfc, `10` TEMF-sfc, `91` old-MM5. **4 missing + 1 ref-only.**

**Land surface `sf_surface_physics`** — IMPLEMENTED `{0,2,4}` (none, Noah-classic, Noah-MP).
MISSING: `1` thermal-diffusion 5-layer-slab (cheap, M), `3` RUC-LSM, `5` CLM4, `6` CTSM, `7` Pleim-Xiu-LSM, `8` SSiB. **6 missing.**

**PBL `bl_pbl_physics`** — IMPLEMENTED `{0,1,5,7,8}` (none, YSU, MYNN, ACM2, BouLac); REFERENCE-ONLY `2` (MYJ, kernel present).
MISSING: `3` ACM-GFS-EDMF, `4` QNSE-EDMF, `9` UW-CAM5, `10` TEMF, `11` Shin-Hong (note: memory says Shin-Hong dynamics banked op — verify), `12` GBM, `16` TKE-eps, `17` TKE-eps-TPE, `99` MRF. **8–9 missing + 1 ref-only.**

**Cumulus `cu_physics`** — IMPLEMENTED `{0,1,2,3,6}` (none, KF, BMJ, Grell-Freitas, modified-Tiedtke); REFERENCE-ONLY `16` (New-Tiedtke).
MISSING: `4` scale-aware-SAS, `5` Grell-3D, `7` Zhang-McFarlane, `10` modified-KF-PDF, `11` MSKF, `14` KSAS, `93` Grell-Devenyi, `94/95/96` GFS-SAS variants, `99` previous-KF. **10 missing + 1 ref-only.** (Memory says GF/Goddard banked op.)

**Dynamics/numerics** — IMPLEMENTED: `diff_opt {0,1,2}`, `km_opt {0,1,4}` (2-D Smagorinsky + constant-K), `damp_opt {0,3}` (w-Rayleigh), `diff_6th_opt {0,2}` (monotonic), `rk_order {3}`, `w_damping {0,1}`.
GENUINELY-MISSING: `km_opt {2,3,5}` (1.5-order 3-D TKE, 3-D Smagorinsky, SMS-3DTKE — the LES/scale-aware closures), `damp_opt {1,2}` (diffusive, idealized-Rayleigh), `diff_6th_opt {1}` (up-gradient 6th-order), `rk_order {2}` (RK2). Plus advection-order options (`h_sca_adv_order`, `v_sca_adv_order`, `h_mom_adv_order`, `v_mom_adv_order`) and the positive-definite/monotonic scalar transport family (`moist_adv_opt`, `scalar_adv_opt` ∈ {1,2,3,4}) are fixed to h=5/v=3 with no PD/monotonic variants (P1-6).

## 2.2 MISSING-to-100% roadmap, grouped by subsystem (rough effort)

| Subsystem | Genuinely-missing items | Rough effort to "every option represented" |
|-----------|-------------------------|---------------------------------------------|
| **Microphysics** | 22 schemes. Pragmatic tiers: cheap-ish 1-mom (5 Ferrier, 7 Goddard-4ice, 14 WDM5, 24 WSM7, 26 WDM7) follow existing WSM/WDM template (**M each**); 2-mom/3-mom modern (9 MY, 10-CESM, 28/29 aerosol-Thompson, 38, 50–53 P3) are **L–XL each**; spectral-bin (30/32 HUJI), CAM5.1, NSSL family, NTU, ISHMAEL are **XL each** research kernels. | **XL** (the single biggest surface area; ~5 schemes/sprint at M, the SBM/P3/CAM at XL) |
| **Cumulus** | 10 schemes. SAS family (4/94/95/96), Grell-3D(5)/Grell-Devenyi(93) reuse the Grell ensemble machinery already present (**M–L each**); Zhang-McFarlane(7) and KSAS(14) and MSKF(11) are **L each**. New-Tiedtke(16) is ref-only→wire (**L**). | **L–XL** |
| **PBL** | 8–9 schemes + MYJ ref-only-wire. QNSE/UW/GBM/TEMF/Shin-Hong/TKE-eps each **M–L** (column-kernel ports following the YSU/MYNN/BouLac template). MRF(99) is legacy, cheap (**M**). | **L–XL** |
| **Surface layer** | 4 schemes (GFS, QNSE, TEMF, old-MM5) + Janjic ref-only-wire. Each pairs with a PBL family; **M each**. | **M–L** |
| **LSM** | 6 schemes. Thermal-diffusion 5-layer-slab(1) is cheap (**M**) and worth doing for completeness; RUC(3) **L**; Pleim-Xiu(7) **L**; SSiB(8) **L**; CLM4(5)/CTSM(6) are **XL** coupled land models. | **L–XL** |
| **Radiation** | LW: 6 missing + RRTM ref-only-wire. SW: 6 missing + Dudhia ref-only-wire. Goddard SW/LW, CAM, FLG each **L** (new gas/band tables). RRTMG-K(14) and fast-RRTMG(24) are RRTMG variants (**M**). GFDL-Eta(99) **L**. Dudhia/RRTM wiring **M** each. **Plus the KI-6 taug top-layer fix (M).** | **L–XL** |
| **Dynamics / diffusion** | 3-D TKE/Smagorinsky/SMS-3DTKE closures (`km_opt 2/3/5`) — real LES/scale-aware turbulence, **L–XL** (prognostic 3-D TKE field + closure). RK2 (**S**, low value). Up-gradient 6th-order (**S**). Advection-order options + PD/monotonic scalar transport (`moist_adv_opt/scalar_adv_opt` 2/3/4) + WRF boundary-order degradation — **L** (P1-6); genuinely affects water conservation near gradients. | **L–XL** |
| **Gravity-wave drag** | `gwd_opt=1` (orographic GWD / blocking, `module_bl_gwdo`). NOT implemented (only w-Rayleigh damping exists). Needs the sub-grid orography stats (`VAR_SSO, OA1–4, OL1–4` — note these appear in the reference wrfout as MISSING too). **L.** | **L** |
| **Nesting** | One-way static nesting IMPLEMENTED + 24 h GREEN. MISSING: **two-way feedback** (`feedback=1`), radiation-in-loop for nests, in-loop `w` relaxation, 5-domain long-run equivalence. Moving/vortex-following nests are correctly **OUT_OF_SCOPE**. | **L–XL** (the deferred two-way-nesting item) |
| **FDDA / DA** | Grid/analysis nudging, obs nudging, spectral nudging — all correctly **OUT_OF_SCOPE** in the catalog. For a literal "every feature represented" they'd be **XL**, but they are a defensible scope boundary (pure forecast-integration port). | OUT_OF_SCOPE (XL if ever in scope) |
| **Stochastic** | SPPT, SKEBS, SPP, rand_perturb — all **OUT_OF_SCOPE** (deterministic port). Seed arrays appear in the reference wrfout but should stay absent. | OUT_OF_SCOPE |
| **Specialty / coupled** | WRF-Chem, WRF-Fire, WRF-Hydro, coupled ocean (`sf_ocean_physics`), urban canopy (UCM/BEP/BEM), windfarm, `sst_update` — all **OUT_OF_SCOPE** in the catalog (documented design decisions). | OUT_OF_SCOPE |
| **I/O** | wrfout coverage 74/368 (→ Part 1 A1/A2/B1/B3 raise it). Auxhist streams (`auxhist*`) **NOT implemented** (no `auxhist` anywhere in src) — **M** to add a second history stream. True `wrfrst` restart exists (`wrfrst_netcdf.py`). Restart-field completeness vs the expanded var set — **M**. | **M–L** |
| **Projections** | Lambert / Mercator / Polar implemented. MISSING: lat-lon (`cylindrical equidistant`, global), rotated-lat-lon. **M each** (init-side geometry only). | **M** |

## 2.3 Honest "what would 100% even mean" framing

A literal "every single WRF v4 option represented" is dominated by the **microphysics (22) + cumulus (10) + PBL (8) + radiation (12) + LSM (6) = ~58 unported scheme kernels**, almost all **M–XL** each. That is the multi-milestone bulk of the road to v1.0.0 and is the right framing for the "untouchable complete port" roadmap: it is a long tail of independent, template-following scheme ports, NOT a small set of blockers. The genuine *architectural* gaps (vs scheme-count gaps) are far fewer: **3-D TKE/LES closures, two-way nesting feedback, GWD, PD/monotonic advection + advection-order options, auxhist streams, lat-lon projection**. Everything coupled/assimilative/stochastic is a defensible OUT_OF_SCOPE boundary that should stay documented, not implemented.

---

# RANKED SUMMARY (for the manager's tonight-vs-v0.13 decision)

### Cheap wins, ranked by value × feasibility:
1. **A1 — route ~30 already-computed vertical-coord + grid-metric static vars to wrfout** (S, no GPU, near-zero risk). Biggest coverage gain per hour; 74→~104 vars. **DO TONIGHT.**
2. **A3/D1/D2 — honesty fixes**: correct the stale "64-variable" claim (it's 85/code, 74/368 emitted), fill the `<<MANAGER-FILL>>` placeholders in KNOWN_ISSUES, mark/fix KI-8 brittle tests (all S, no GPU). **DO TONIGHT** — a skeptical reviewer catches these in minutes.
3. **A2 — trivially-derived diagnostics** (RH2, SNOWC, SR, COSZEN-already-computed, CLAT) (S, no GPU). **DO TONIGHT** alongside A1.
4. **B5 — namelist recognition breadth** (recognize gwd_opt/adv_opt/icloud_bl/cadence keys, fail-closed-with-reason) (M; first batch S). High credibility — a real WRF namelist gets honest per-key verdicts.
5. **B3 — Noah-MP snow-layer diags** (TSNO/SNICE/SNLIQ/ZSNSO) (M, no GPU; carry likely already holds them — verify first). Closes a *named* KI-3 gap.
6. **B1 — radiation flux-boundary diags** (SWDNB/LWUPB/OLR… ~16 vars from existing flux profiles) (M, plumbing only).
7. **C-Dudhia — scan-wire Dudhia SW** (M, GPU) — smallest reference-only scheme; proves radiation-family dispatch generalizes.
8. **B2 — RRTMG taug UV-band fix (KI-6)** (M, GPU, needs re-validation) — fidelity, not a forecast bug.

### MISSING-to-100% groups, ranked by (architectural-importance, then effort):
- **Architectural gaps (do for a *credible* complete port):** two-way nesting feedback (L–XL) · 3-D TKE/Smagorinsky/SMS-3DTKE LES closures (L–XL) · GWD `gwd_opt` (L) · PD/monotonic + advection-order scalar transport (L) · auxhist streams (M) · lat-lon/rotated projections (M).
- **Scheme long-tail (the bulk of v0.13→v1.0.0):** microphysics ×22 (M–XL) · cumulus ×10 (M–L) · PBL ×8 + MYJ-wire (M–L) · radiation ×12 + Dudhia/RRTM-wire + taug (M–L) · surface-layer ×4 + Janjic-wire (M) · LSM ×6 (M–XL). Template-following, parallelizable.
- **Correctly OUT_OF_SCOPE (keep documented, do NOT implement):** WRF-Chem, WRF-Fire, WRF-Hydro, coupled ocean, urban canopy, moving nests, FDDA/DA, all stochastic physics.

**Bottom line for tonight:** the highest-confidence, no-GPU, few-hours push is **A1 + A2 + A3 + D1 + D2** — it raises wrfout coverage from 20% to ~31%, makes the public KI-3/doc claims honest, and removes the two reviewer-bait `<<MANAGER-FILL>>` placeholders, all with zero forecast-correctness risk and no GPU contention. Everything GPU-touching (B1/B2/C-wiring) and every new scheme belongs in v0.13.
