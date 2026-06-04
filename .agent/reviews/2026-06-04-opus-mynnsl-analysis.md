# MYNN surface-layer (sf_sfclay_physics=5) — complete analysis + faithful fix

**Author:** Opus 4.8 (max effort), 2026-06-04
**Branch:** `worker/opus/v090-mynnsl-analysis` (from trunk-0.9.0 @ 7b7c26e)
**Oracle:** UNMODIFIED pristine `/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F`
sha256 `86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3`
**Outcome:** **FIXED** — 8 localized WRF-fidelity deviations corrected; MYNN-SL
flux+similarity+diagnostic now PASS vs the pristine oracle; the one empirical repair
isolated behind an opt-in flag (default OFF = faithful). The daytime over-flux is
**WRF's own MYNN-SL behaviour**, not a port bug, and operationally is owned by Noah-MP
over land (case 4a for the SL module; 4b/4c for the 2-m T2 diagnostic only).

---

## 1. Method — a real WRF oracle, not a self-compare

I built a byte-identical Fortran oracle by compiling the pristine `module_sf_mynn.F`
(verified `cmp`-identical, sha256 above) with gfortran (wrfbuild env) plus a 4-symbol
`module_model_constants` shim (`p1000mb, r_d, r_v, ep_2` — the only externals MYNN
uses) and a thin driver that calls the UNMODIFIED `SFCLAY1D_mynn`. Default build is
REAL*4 (exactly how operational WRF runs); a `-fdefault-real-8` build isolates fp32
roundoff. Config: `ISFFLX=1, isftcflx=0, iz0tlnd=0, spp_pbl=0, COARE_OPT=3.0,
psi_opt=0 (CB05), itimestep=2`.

I also wrote a faithful fp64 NumPy transcription of `SFCLAY1D_mynn` (`mynn_faithful_ref.py`,
NO empirical repair). It matches the fp64 Fortran oracle to ~1e-3 (zolrib fixed-point
seed + fp-table floor) and **HFX to 1.4e-4 relative** — so it is a trustworthy second
witness. Three-way harness: pristine Fortran oracle vs production `surface_layer.py` vs
faithful ref, over daytime-unstable land (the crux, 4 columns), stable-night land (3),
neutral (1), and daytime water (2). Proofs in `proofs/v090/`.

## 2. Root-cause verdict

### The daytime over-flux IS WRF's own MYNN-SL behaviour (not a port bug)
The faithful fp64 ref reproduces WRF MYNN-SL daytime-unstable land HFX of
**270 / 457 / 664 W/m²** (dT=+8/+12/+16 K) to 1.4e-4 relative against the pristine
oracle. So MYNN-SL genuinely produces large midday sensible-heat flux for warm-skin,
light-wind land columns — matching WRF faithfully *reproduces* that flux. This is the
fact the v0.1.0 prescribed-input oracle was circling: its own notes say the residual
land HFX is "the Noah-MP LSM surface-energy-balance coupling, not reproducible by a
standalone surface layer."

**Operational consequence (the part the situation framing under-stated):** in the
coupled GPU model (`noahmp_coupler.py`), over LAND the operational HFX/QFX/ZNT/TSK come
from **Noah-MP** (`nm.hfx`, masked `where(is_land, noahmp, sfclay)`), NOT from MYNN-SL.
MYNN-SL owns the **water** flux and the **momentum + similarity + 2-m diagnostics**
everywhere. So the d03 "midday HFX 505 vs 137" land over-flux is a **Noah-MP land
energy-balance question (case 4c)**, while the MYNN-SL flux that *is* used operationally
(water + similarity) is now WRF-faithful (case 4a, fixed below).

### The production code had 8 real localized fidelity deviations (case 4a)
Before this sprint, production `surface_layer.py` diverged from the pristine oracle by
**zol 57%, br 67%, psih 44%, ust 22%** — all traced to concrete non-faithful choices:

| ID | Field | Production (was) | WRF MYNN | Effect |
|----|-------|------------------|----------|--------|
| B1 | `VCONVC` | 1.0 | **1.25** | unstable wstar/wspd too small → br ~40% too negative |
| B2 | land conv. height | `pblh` | **`min(1.5·pblh,4000)`** | compounds B1 |
| B3 | water z0 | keep seed 2.85e-3 | **`charnock_1955`** (~1e-4) | water zol/psih wrong (z0 33× off) |
| B4 | `CPM` | `cp(1+0.8q)` | **`cp(1+0.84q)`** | flux ~0.4% (synced in coupler) |
| B5 | regime | no regime 2 | **BR>0.2→1, 0<BR≤0.2→2** | diagnostic |
| B6 | BR clamp | `[-250,250]`+MOL-gate | **`[-4,4]`**, MOL-gate is commented-out in WRF | br extremes |
| B7 | viscosity | sfclayrev linear | **Andreas(1989) cubic** | restar→z_t |
| B8 | humidity | single mixing-ratio `qsfc`, +salinity | **QSFC(spec hum, ep_3, ice) vs QSFCMR(mixing ratio)**, no salinity | THVGB / QFX / Q2 |

I reconstructed production's br=-0.8619 exactly from its constants (vs WRF -0.6163),
proving B1+B2+B4 fully explain the dominant unstable-land divergence. The zolrib seed
was also Li-only (cold-start path) where WRF warm steps use the MOL-based first guess.

### The 2-m T2 diagnostic is a separate faithful-vs-skill / missing-LSM matter (4b/4c)
The only empirical repair in the file was `_noahmp_bare_2m_weight_stable`, which over
stable land replaced the WRF MYNN 2-m weight `PSIT2/PSIT` with a bare-ground SFCDIF1
weight, making T2 **+1.4…+2.8 K warmer** than WRF's MYNN diagnostic. This is NOT
MYNN-SL being unfaithful to MYNN-SL — it is a **stand-in for a coupling step real WRF
performs**: over a Noah-MP land point WRF OVERWRITES the MYNN 2-m T2 with the LSM
diagnostic `T2 = FVEG·T2MV + (1-FVEG)·T2MB` (module_sf_mynn.F:1135;
module_surface_driver.F:3470). This port's Noah-MP does not yet compute T2MB/T2MV.
It touches ONLY th2/t2/q2 — never flux, MOL, or the PBL bottom BC.

## 3. The fix (faithful, no new empiricism)

All 8 deviations corrected against the pristine source (every block line-referenced).
`VCONVC_MYNN=1.25` is a new MYNN-only constant; the sfclayrev-family schemes
(`sfclay_revised_mm5`, `sfclay_pleim_xiu`) keep `VCONVC=1.0`. The empirical 2-m repair
is now gated behind `state.lsm_t2_diag` (**default False = faithful**); with the default
the entire MYNN-SL output matches the oracle.

### Parity vs pristine WRF oracle — before → after

| field | before relmax | after relmax | verdict |
|-------|---------------|--------------|---------|
| zol   | 57.1% | 2.06% | PASS |
| br    | 66.5% | 2.07% | PASS |
| psih  | 43.8% | 1.29% | PASS |
| psim  | 45.5% | 1.37% | PASS |
| ust   | 22.4% | 0.17% | PASS |
| mol   | 19.2% | 2.17% | PASS |
| hfx   | 3.83% (and on the WRONG side — too low) | 0.98% | PASS |
| lh    | 11.8% | 1.22% | PASS |
| qsfc  | 5.54% | 0.04% | PASS |
| u10/v10 | 3.41% | 0.02% | PASS |
| t2 (faithful default) | 2.79 K abs | **6.5e-4 K abs** | PASS |

The residual ~2% on zol/br/psi is the zolrib fixed-point convergence-path + fp32-table
floor — the same magnitude as the faithful fp64 ref's own residual vs the oracle, i.e.
at the transcription floor, not a structural bug. **MYNN-SL FAITHFUL VERDICT = PASS**
(`proofs/v090/mynnsl_parity.json`).

Per-case daytime-unstable HFX after fix (W vs production): 270.2/271.5, 457.2/459.5,
664.0/667.4, 186.3/187.2 — within ~1%, on the correct side. Water HFX <1%. T2 on
unstable/neutral/water to ~1e-3 K.

## 4. Operational T2 — the scope choice (surfaced for the principal)

Flipping `lsm_t2_diag` to its faithful default **changes operational land T2**: it
removes the repair that previously cut overnight land-T2 RMSE vs WRF *wrfout* T2 from
1.64 K to 0.72 K. The reason is structural — WRF's wrfout land T2 *is* the LSM-overwritten
value, so matching obs/wrfout over land genuinely needs the LSM overwrite that MYNN-SL
alone does not provide. Three options:

- **(A) Ship faithful (default, recommended for the module):** MYNN-SL is exactly
  module_sf_mynn.F. Land T2 = WRF's MYNN 2-m diagnostic. Honest, no empiricism. May
  re-expose the historical nocturnal land-T2 cold tendency vs obs until Noah-MP T2MB
  exists.
- **(B) Opt-in the LSM stand-in (`lsm_t2_diag=True`):** better obs skill over land, but
  the 2-m T2 is NOT module-faithful — it approximates the LSM overwrite. Use only as a
  labeled operational knob.
- **(C) Implement Noah-MP T2MB/T2MV (the real fix):** route the genuine LSM 2-m
  diagnostic through the coupler. Removes the tension entirely. Out of MYNN-SL scope;
  tracked.

**Recommendation:** ship (A) as the MYNN-SL default (faithful), and decide (B vs C) for
the operational T2-over-land separately — this is a Noah-MP/coupler scope item, not a
MYNN-SL defect. The daytime warm-bias/over-flux over land is likewise a Noah-MP
land-energy-balance question, since operational land HFX is Noah-MP's, not MYNN-SL's.

## 5. Honest unresolved risks

- **Operational coupled re-run not executed.** The fix is proven at savepoint (single-call
  parity vs the pristine oracle) and the 31 surface/MYNN/PBL/coupler unit tests pass, but
  I did not run a coupled d02/d03 forecast to measure the net operational T2/HFX shift
  from B1-B8 + the faithful T2 default. The heavy coupled-step parity test
  (`test_m6b6_column_coupled_step_parity_one_step`) **segfaults (SIGILL) on THIS host on
  both my branch AND the pristine base** — a stale AVX512 XLA-AOT / CPU-codegen
  environment issue, NOT introduced by this change (verified by reproducing on the base
  files). A coupled GPU confirmation run is the recommended next gate.
- **zolrib seed/convergence floor (~2%)** on zol/br/psi: faithful but at the
  fixed-point transcription floor; tightening would require bit-matching the Fortran
  while-head early-stop exactly. Below the operational tolerance.
- **Land daytime over-flux root cause for the *coupled* model** lives in Noah-MP's
  surface energy balance (operational land HFX = Noah-MP), not MYNN-SL — out of this
  sprint's scope but the logical next investigation for the v0.2.0 damped-diurnal-T2.

## 6. Files

- `proofs/v090/module_sf_mynn_pristine.f90` (byte-identical oracle source) + `oracle_source_sha256.txt`
- `proofs/v090/module_model_constants.f90`, `mynn_oracle_driver.f90`, `build_oracle.sh`
- `proofs/v090/mynn_faithful_ref.py` (faithful fp64 ref), `_validate_ref_vs_oracle.py`
- `proofs/v090/mynnsl_parity.py` → `mynnsl_parity.json` (per-field PASS/FAIL verdict)
- `proofs/v090/mynnsl_localization.json` (before/after, 8 deviations)
- FIX: `src/gpuwrf/physics/surface_layer.py`, `surface_constants.py` (+`VCONVC_MYNN`),
  `noahmp_coupler.py` (CPM 0.84 sync), `coupling/physics_couplers.py` (`lsm_t2_diag` field)
