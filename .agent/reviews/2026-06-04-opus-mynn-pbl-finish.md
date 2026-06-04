# v0.9.0 MYNN-PBL FINISH — water-path generalization + entrainment/qke residual closure

Branch: `worker/opus/v090-mynn-pbl-finish` (base `worker/opus/v090-mynn-pbl-mymlength-fix` @ ccfa910)
HEAD: `339d442`. Mode: CPU fp64 (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`), cores 0-3 only.
Oracle: UNMODIFIED pristine WRF v4.7.1 `module_bl_mynnedmf.F`
(`sha256=6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72`),
`/mnt/data/wrf_gpu2/physics_oracle_v090/surface_mynn`, itimestep 1000, 5487 columns × 44 levels.
JAX-vs-WRF, predeclared tolerances UNCHANGED, no clamps/fudge.

## Objective (from the verifier's flags)

Finish the accepted mixlength=1 fix (ccfa910): (1) generalize the two water-path
simplifications faithfully for land+water (Canary is marine); (2) close the el_pbl
PBL-top tail + exch_h/exch_m/qke/tendency residuals faithfully, or report an
honest irreducible residual.

## Key context discovered

The v090 oracle case (CONUS coastal `20260428_18z`) is **90.5% WATER columns**
(xland=2.0; 4965 water / 522 land). So this oracle directly exercises the marine
regime that matters for Canary. All winds < 20 m/s (no hurricane taper active).

## Three WRF-faithful fixes (no clamps)

### 1. Water-path generalization (verifier flag #1) — `_mym_length_option1`
WRF `mym_length` CASE(1) (`module_bl_mynnedmf.F:1801-1805, 1859-1861`) branches on
the land/sea mask `(xland-1.5)>=0`:
- `elt_max`: WATER = `350 + 100*min(1,max(0,ugrid-50)/25)` (<=450); LAND = `400`.
  The kernel hardcoded the land 400. Now faithful for both, with the real WRF
  `xland` threaded through `SurfaceFluxes -> _mym_turbulence -> _mym_length_option1`
  (default 1.0=land for the analytic fixtures; operational coupler + harness pass the
  real per-column mask). Coupler/surface_layer/noahmp_coupler `SurfaceFluxes` now carry xland.
- `el(k)*wt_u1` hurricane taper: now WATER-ONLY (was applied unconditionally).
- **No-op at THIS step** (elt maxes at 121 < 350, U<20 so wt_u1=1), so it does not
  move the proof here — but the branch is now general/faithful for the deep-PBL,
  high-TKE, and >50 m/s marine conditions where it binds. Verified by construction
  + the land/water unit branch.

### 2. `pmz` surface-shear factor (qke root cause) — `_mym_predict_qke`
WRF `mym_predict` line 3072: `pdk1 = 2*ust**3*PMZ/vkz`. The kernel dropped `pmz`
(implicitly =1). `pmz = phim(zet) - zet`, `zet = clip(0.5*dz1*rmol, -20, 20)`,
with the **Puhales-2020** `phim` (`bl_mynn_stfunc=1` default, `:7701-7750`), ported
faithfully (stable Cheng-Brutsaert + unstable Grachev blend). In stable surface
layers (`fltv<0 -> zet>0`) `pmz>1` (~3.1 at the failing cells); dropping it
under-produced surface TKE and collapsed `qke(kts)`. **All 253 qke violations were
k=0, land, fltv<0; porting pmz closes them 253 -> 0** (max_rel 263 -> 0.14).

### 3. Specific humidity in the buoyancy variable — `_virtual_potential`
WRF forms `thv` (and `thlv` for the level-2 buoyancy gradient) from the **specific
humidity** `sqv = qv/(1+qv)` (`:673` `thv1=th1*(1+p608*sqv1)`, `:1006-1008`), NOT
the mixing ratio. The kernel used `qv`. At the well-mixed-layer top the buoyancy
gradient `dthv/dz` is a knife-edge near zero; the ~1% `qv` vs `qv/(1+qv)` difference
**flips its sign** at exactly the inversion-base cells, deciding whether the level-2.5
Sh treats the cell as (un)stable. The mixing-ratio form made the JAX see those cells
as marginally UNSTABLE -> large Sh -> spurious entrainment-zone over-mixing (kh_j~30
vs kh_w~0.3). The faithful specific humidity flips them stable -> Sh suppressed,
matching WRF. **exch_h 1570 -> 721, el_pbl 70 -> 31, exch_m 353 -> 340.**

## Full-parity status per field (vs unmodified WRF oracle, predeclared tols)

| field | base viol | final viol | status |
|---|---|---|---|
| qke (2·TKE) | 253 | **0** | **PASS** — pmz fix closed it (max_rel 263 -> 0.14) |
| exch_h (Kh) | 1570 | 721 | residual: **95% SGS-cloud-coupled**; 37 clear, abs<5.8 m²/s |
| exch_m (Km) | 353 | 340 | residual: **100% SGS-cloud-coupled** |
| el_pbl | 70 | 31 | residual: **100% SGS-cloud-coupled** (median abs 4.75 m, ±1 lvl of cloud) |
| rthblten | 864 | 679 | 90% cloud-coupled; clear residual abs <2.4e-5 K/s |
| rqvblten | 2498 | 2384 | 41% cloud; clear residual abs ~2e-8 (at the 1e-8 abs floor) |
| rublten / rvblten | 200 / 723 | 203 / 776 | small; clear residual abs ~2-5e-5 m/s² |
| pblh | 18 | 18 | unchanged; median \|dpblh\| 0.02 m, 18 shallow cells just over abs 50 m |

Overall proof `pass=false` — the cloud-coupled entrainment residual remains.

## Honest irreducible residual + reason

After the three faithful fixes, the el_pbl/exch_m residual is **100% SGS-cloud-coupled**
and exch_h is **95%** (only 37 of 721 clear, all tiny abs<5.8 m²/s in a field whose
WRF max is 137). Mechanism (verified at the failing cells): WRF's cloud-PDF
(`mym_condensation`) folds the SGS cloud condensate + cloud fraction
(`qc_bl`/`cldfra_bl`) into the liquid-water virtual potential temperature `thlv1`
(`:1006-1008`), which modifies the entrainment-zone buoyancy gradient — changing both
the master length (`dtv -> elb` stable-branch clip) and the level-2.5 Sh at the
marine-stratocumulus cloud top. The JAX kernel is a **dry MYNN2.5 eddy-diffusion
column**: it does not carry the SGS cloud-PDF (qc/qi prognostic mixed scalars; the
unsaturated daytime PBL has qc=qi=0). So the cloud-top entrainment cells are a
**genuine, documented scope boundary, not a transcription bug**. Closing them
requires porting `mym_condensation` + the topdown-cloud-radiation TKE source — a
separate scheme component, out of this lane's scope.

The 37 clear exch_h + the small clear tendency cells are the residual ~14%-too-long
`el` over a handful of water entrainment cells (the in-PBL `el_core -> elt` blend
matches WRF's `elt`/`els` exactly by construction — verified against the WRF
DO-WHILE loop semantics — so this remainder is operator-split / near-neutral
sensitivity, not a missed term).

## Operational impact (negligible)

- Eddy diffusivity exch_h matches WRF to **<1% in the mixing bulk**: median |dKh|
  0.005, mean 0.20 m²/s vs WRF median 24 / max 137 m²/s. Errors confined to thin
  cloud-top layers.
- theta tendency -> equivalent **1-hour dT: median 0.004 K, p99 0.094 K**, max 0.76 K
  (a single cloud-top cell). qv tendency clear residual is at the 1e-8 abs floor.
- PBLH: median |dpblh| 0.02 m. qke now exact.

Conclusion: the operational PBL mixing, TKE, and PBLH are WRF-faithful; the residual
is sub-grid-cloud-top entrainment physics that the dry ED column does not model, with
negligible operational footprint.

## WRF-faithful confirmation

No clamps, no masks, no loosened tolerances, no tuning constants added. All three
fixes are direct ports of WRF `module_bl_mynnedmf.F` lines (1801-1805/1859-1861 water
branch; 879-897+7701-7750 pmz/phim; 673/1006-1008 specific-humidity thv). Predeclared
tolerances in `mynn_pbl_savepoint_parity.py` are byte-identical to ccfa910.

## Risk

- The water `elt_max=350` branch is correct-by-construction but UNTESTED on a case
  where `elt>350` (deep convective marine PBL). Low risk: it is a literal WRF
  transcription and reduces (never increases) `el` over water.
- The SGS-cloud entrainment residual is a real coverage limit for the dry column;
  a stratocumulus-heavy multi-day run would stress it more than this single evening
  step. Tracked as scope, not regression.
- 13/14 MYNN CPU unit tests pass; the 1 failure (`test_mynn_adapter_consumes_..`)
  is a pre-existing GPU-device requirement (`State.zeros requires a GPU`), unrelated
  to these changes (fails before any edited code runs). The analytic Tier-1 fixture
  still passes (loose tols abs el 20 / km,kh 5); its WRF-linked harness is a coarse
  synthetic column and is not the authoritative gate — the real-WRF savepoint oracle is.

## Files changed
- `src/gpuwrf/physics/mynn_pbl.py` — water branch (xland) in `_mym_length_option1`;
  `_phim_puhales`/`_pmz_surface` + pmz in `_mym_predict_qke`; specific-humidity `_virtual_potential`.
- `src/gpuwrf/physics/mynn_constants.py` — `NL_ELT_MAX_WATER`, `CPHM_UNST`.
- `src/gpuwrf/physics/mynn_surface_stub.py` — `xland` field on `SurfaceFluxes` (default land=1.0).
- `src/gpuwrf/physics/surface_layer.py`, `noahmp_coupler.py`,
  `src/gpuwrf/coupling/physics_couplers.py` — thread `xland` through to MYNN.
- `proofs/v090/mynn_pbl_savepoint_parity.py` — load + pass real `xland`.
- `proofs/v090/mynn_pbl_savepoint_parity.json` — regenerated.

## Reproduce
```
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 \
  python3 proofs/v090/mynn_pbl_savepoint_parity.py \
  --oracle-dir /mnt/data/wrf_gpu2/physics_oracle_v090/surface_mynn
```
