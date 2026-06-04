# v0.9.0 Physics Consolidation — merge of the five #1 physics fixes + integrating gate

**Author:** Opus 4.8 (1M context) integration lane, 2026-06-04
**Branch:** `worker/opus/v090-physics-consolidation` (from `worker/opus/trunk-0.9.0` @ `7b7c26e`, the closed-v0.6.0 base)
**Mode:** CPU fp64 for merge + savepoint re-verify (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`, `taskset -c 0-3`); GPU (cuda:0) for the coupled forecast. Cores 4-31 (live 28-rank CPU-WRF backfill, pid 133927) untouched throughout.

## Objective

Consolidate the five #1 physics-fix branches onto trunk-0.9.0, PROVE every fix survived the merge (no regression at its predeclared tolerance), then run the combined-MYNN COUPLED GPU confirmation. WRF-faithful; honesty over green; never weaken a fix or loosen a tol to make the merge "work."

## PART 1 — Consolidation (merge order + conflict resolution)

Merged in the contracted order onto the consolidation branch:

| # | Branch | SHA | Result |
|---|---|---|---|
| 1 | `worker/opus/v090-noahmp-t2mb` | 2268fe6 | clean (first merge) — MYNN-SL faithful + retire stand-in + Noah-MP T2MV/T2MB land-T2 overwrite |
| 2 | `worker/opus/v090-thompson-warmrain-fix` | 5a1ee35 | clean — Thompson warm-rain qr/Nr (thompson_column.py) |
| 3 | `worker/opus/v090-mynn-pbl-finish` | 1305ec7 | auto-merged (ort) — MYNN-PBL CASE-1+BouLac + pmz + specific-humidity-thv + xland water-path |
| 4 | `worker/opus/v090-namelist-compat` | f459bcd | clean — WRF namelist catalog + 3-outcome validation |

### The conflict cluster (surface_layer.py / noahmp_coupler.py / physics_couplers.py)

These three files are modified by BOTH #1 and #3. The contract required resolving by **combining, never dropping a fix**. The `ort` strategy auto-merged all three (the changes live in non-overlapping regions), and I explicitly verified BOTH fixes are co-present (no clobber, no conflict markers):

- **surface_layer.py** — #1's 221-line MYNN-SL faithful rewrite (VCONVC_MYNN; Andreas-1989 viscosity; COARE-3.0 water-z0 Charnock update; QSFC/QSFCMR specific-humidity vs mixing-ratio split; `cpm=CP*(1+0.84*qv)`; warm-step MOL zol1 seed; faithful `w2m = psit2/psit`; **retired** the empirical `_noahmp_bare_2m_weight_stable` stand-in) **AND** #3's `xland=xland` added to the `SurfaceFluxes` constructor (line 797). Grep confirms the stand-in helper + `land_stable`/`w2m_bare` are fully gone, and `xland=xland` is present.
- **noahmp_coupler.py** — #1's `rho_cpm = rhosfc*(CP_D*(1+0.84*qx))` (MYNN moist heat capacity, matches surface_layer.py) **AND** #3's `xland=sf.xland` carry-through. Old `0.8` form gone.
- **physics_couplers.py** — #1's `lsm_t2_diag` RETIRED field on `_SurfaceColumnState` **AND** #3's `xland=jnp.asarray(state.xland,...)` in `_surface_fluxes_from_state`.

Total core-source delta vs base: 16 files, +1150/−153. All merged modules import cleanly under CPU fp64.

## PART 2 — Re-verify every fix survived (CPU fp64, before the GPU run)

Each savepoint parity driver was RE-RUN on the merged tree against its UNMODIFIED pristine-WRF oracle (gfortran oracle rebuilt from byte-identical `module_sf_mynn.F`; Thompson/MYNN-PBL read the `/mnt/data/wrf_gpu2/physics_oracle_v090` savepoint dumps; T2MB reads the compiled Noah-MP savepoints). No tolerance was loosened. Aggregated in `proofs/v090/physics_consolidation_reverify.json`:

| Fix | Predeclared gate | Post-merge result | PASS? |
|---|---|---|---|
| MYNN-SL | `mynnsl_faithful_verdict == PASS` | PASS, 10 cases, T2 to 3 dp vs `module_sf_mynn.F` | ✅ |
| MYNN-PBL | `qke n_violations == 0` | 0 violations (max_abs 0.004715 < 0.005); EXACT reproduction of branch result | ✅ |
| Thompson | `qr.pass and nr.pass` | both PASS; residuals bit-identical to committed branch (qr 4.117e-11, nr 7.151e-4) | ✅ |
| Noah-MP T2MB | `11/11, nfail==0` | 11/11, worst 1.08e-3 K (crux stable-nocturnal land < 1e-3 K) | ✅ |
| v0.6.0 multicfg smoke | `n_run_fail==0, all RUN PASS` | 20/20 RUN PASS + 3/3 FAIL-CLOSED OK | ✅ |
| v0.6.0 consolidation matrix | `overall_consolidation_pass` | True (git_head f1f720d, 22 GPU-operational-wired, 0 unknown) | ✅ |
| namelist + v090 unit tests | pytest all pass | 30 passed | ✅ |

**No merge regression.** The MYNN-PBL and Thompson overall-strict matrices intentionally remain `pass:false` for the fields each branch predeclared as honest scope-limited residuals (MYNN-PBL tendencies/km/kh/pblh in the eddy-diffusion-only column; Thompson qc-autoconversion 5.4e-9 vs 1e-9 + theta fp32-storage band). Those are NOT these fixes' gate criteria and reproduce the branches' own committed state exactly.

Checkpoint committed at `e2bf685`.

## PART 3 — Combined-MYNN coupled GPU confirmation

### Setup and the integrating case

Used a corpus d02 Canary case with a CPU-WRF wrfout reference and the full operational, radiation-ON config: **20260521_18z_l2** (d02 3 km, 66×159×44; MP8 Thompson / BL5 MYNN-EDMF / SF5 MYNN-SL / SURFACE4 Noah-MP / RA_LW4+RA_SW4 RRTMG / CU0). The d02 CPU-WRF **boundary-replay** GPU coupled forecast (init from the CPU-WRF wrfout t0, CPU-WRF hourly side-history as the LBC) exercises the consolidated **MYNN-SL + MYNN-PBL (xland water-path active in `_step_mynn_pbl_impl_with_pblh` → `_mym_turbulence(..., flux.xland)`) + Thompson** in one coupled GPU run. (First-choice case 20260509 was rejected by the harness's hardcoded `(66,159)` drop-in guard — it is `(66,120)` — so the `(66,159)` 20260521 case was used.)

**Predeclared pass:** finite/stable over the window; final-lead RMSE vs CPU-WRF within the frozen Tier-4 bars (T2 ≤ 3.0 K, U10 ≤ 7.5, V10 ≤ 7.5 m s⁻¹); report daytime-lead T2 RMSE/bias vs CPU-WRF and the pre-fix comparison.

### Result — coupled run blocked by a PRE-EXISTING base-level dycore/LBC instability (NOT a consolidation regression)

The merged-branch coupled forecast went **non-finite after forecast hour 1** (`NONFINITE_STATE`): a dynamics-first blow-up (u→~10¹², v→~10¹⁴, θ→~10³⁷, μ/φ→~10¹⁸¹) with the surface fields (ustar/fltv/τ/θ-flux) NaN *downstream* of the exploded winds.

To attribute it honestly I ran the **identical replay on the UNMERGED trunk-0.9.0 base (7b7c26e)** as a control. The base blows up **identically**: same failure hour (1), same mode (`NONFINITE_STATE`), the **same exact non-finite-field set**, and the same dynamics-first magnitude (base u→~10¹⁸, θ→~10³⁷, μ→~10¹⁷⁸). The consolidation has **ZERO** dynamics/boundary/driver delta vs base (`git diff --stat 7b7c26e..HEAD` over `src/gpuwrf/dynamics/`, `boundary_apply.py`, `driver.py`, `boundary_replay.py` is empty — it is a physics-only merge of 16 files). 

**Conclusion:** this is a pre-existing d02-replay dycore/LBC instability on this `(66,159)` case at the trunk-0.9.0 base, independent of the physics consolidation. The coupled-confirmation via this harness on this case is **infeasible** (the core blows up before any physics-fix skill signal can be measured), and the consolidation is exonerated (no regression, no worse than base). The daytime-T2 warm-bias question cannot be answered from this coupled run; the M7-era pre-fix d02 baseline (a different `20260525` case/grid, now purged so not re-runnable) showed T2 RMSE ≈ 4.07 K with both winds failing — a directional, not same-case, reference.

**Fallback evidence (binding):** the savepoint re-verify is the authoritative proof that the consolidated physics is WRF-faithful and regression-free — MYNN-SL T2 to 3 dp vs `module_sf_mynn.F` (incl. daytime-unstable land), Noah-MP T2MB 11/11 to < 1.1e-3 K (incl. the stable-nocturnal land crux), Thompson qr/Nr PASS, MYNN-PBL qke 0-viol — plus the v0.6.0 multicfg smoke (20/20 RUN PASS) which IS a multi-step coupled integration of all merged schemes (sfclay_mynn/bl5, land_noahmp, mp8) and ran finite/stable. Proof: `proofs/v090/combined_mynn_coupled_confirm.json`.


## Risk / honesty notes

- The Noah-MP **T2MB land-T2 overwrite** (#1) is wired in `compute_m9_diagnostics` gated on `use_noahmp + noahmp_land`. The d02 boundary-replay coupled path runs `use_noahmp=False`, so over land the coupled T2 comes from the (now faithful) MYNN-SL 2-m diagnostic, NOT the LSM overwrite. The T2MB fix is therefore SAVEPOINT-PROVEN (11/11) but not exercised in this particular coupled run. The marine d02 domain is >majority water, where the MYNN-SL/water path governs T2 regardless.
- The clean drop-in replay harness hardcodes the d02 mass grid to (66,159); the 20260509 case is (66,120) and was rejected by that guard, so the (66,159) 20260521 case was used (same operational config: MP8/BL5/SF5/SURFACE4/RA4, CU0, 3km).
