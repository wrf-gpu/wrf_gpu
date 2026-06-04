# Faithful Noah-MP 2-m T2 LSM diagnostic (T2MV / T2MB / combined T2)

**Author:** Opus 4.8 (max effort), 2026-06-04
**Branch:** `worker/opus/v090-noahmp-t2mb` (from `worker/opus/v090-mynnsl-analysis` @ 26d94f8)
**Oracle:** UNMODIFIED pristine `/home/enric/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F`
sha256 `74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b`
**Outcome:** **PASS** — the genuine Noah-MP T2MV/T2MB 2-m LSM diagnostic is implemented
faithfully (savepoint parity 11/11 vs pristine WRF, worst 1.08e-3 K), wired into the
operational land-T2 path, and the opt-in empirical `lsm_t2_diag` bare-ground stand-in is
RETIRED. The stable-nocturnal land T2 the stand-in was patching now resolves to within
3.2e-4 K of WRF — **with no empiricism**.

---

## 1. What real WRF does (the target)

Over a Noah-MP LAND point WRF OVERWRITES the surface-layer (MYNN/sfclay) 2-m temperature
with the LSM diagnostic (module_surface_driver.F:3469-3473):

- vegetated (`FVEG>0`): `T2 = FVEG*T2MV + (1-FVEG)*T2MB`
- bare / urban / `FVEG=0`: `T2 = T2MB`
- water / sea-ice: SFCDIAG form from TSK/HFX (owned by MYNN-SL, not Noah-MP)

with the two tile diagnostics computed inside NOAHMP_SFLX:

- **T2MV** (VEGE_FLUX, :4148-4163):
  `CAH2 = FV*VKC/(LOG((2+Z0H)/Z0H) - FH2)`;
  `T2MV = TAH - (SHG + SHC/FVEG)/(RHOAIR*CPAIR) * 1/CAH2` (else `TAH` if `CAH2<1e-5`).
- **T2MB** (BARE_FLUX, :4461-4474):
  `EHB2 = FV*VKC/(LOG((2+Z0H)/Z0H) - FH2)`;
  `T2MB = TGB - SHB/(RHOAIR*CPAIR) * 1/EHB2` (else `TGB` if `EHB2<1e-5`).

`FH2`/`FV`/`Z0H` are the SFCDIF1-converged values (FH2 = the 2-m Monin-Obukhov heat
adjustment, :4691-4729; Z0H = Z0M over bare ground, CZIL commented out at :4354-4356).
`SHG` is the post-loop2/clamp under-canopy ground sensible heat; `SHC` is the FVEG-weighted
canopy flux (divided back out in the T2MV formula); `SHB` is the bare-ground flux.

## 2. Method — extend the existing real-WRF oracle, not a self-compare

The S0b Noah-MP oracle harness (`proofs/noahmp/`) already links the COMPILED pristine
`module_sf_noahmplsm.o` and calls the exact `NOAHMP_SFLX` orchestrator on real Canary d03
land columns (`noahmp_offline_driver.F90`, scope dveg=4/opt_sfc=1/opt_alb=2/opt_stc=1/...).
`T2MV`/`T2MB`/`Q2V`/`Q2B` are already `NOAHMP_SFLX` OUT args — they were simply not being
written to the savepoint. I:

1. extended the driver to emit a `T2DIAG t2mv t2mb t2m q2v q2b` line (with the
   driver-level FVEG combine mirroring module_surface_driver.F:3470/3467), rebuilt the
   driver against the unmodified `.o`, and regenerated the savepoints;
2. implemented T2MV/T2MB in the JAX port from the SAME converged FH2/FV/Z0H/SHG/SHC/SHB
   the energy balance already produces (the energy gate proves those match WRF to
   <1e-3 K), plus the FVEG combine in `noahmp_energy_canopy`;
3. built `proofs/v090/noahmp_t2mb_parity.py` that feeds each savepoint column through the
   JAX port and compares t2mv/t2mb/t2 to the Fortran OUT values (same WRF parameter
   tables + same column state → external oracle, NOT a self-compare).

Coverage (the savepoint columns): vegetated (veg 5/9/10, FVEG 0.26-0.35) + bare/urban
(veg 13/16, FVEG=0 → T2=T2MB) + daytime-unstable (cosz~0.94, soldn 770-1070) + the
**stable-nocturnal** land columns (soldn=0; the +2.8 K crux) + a Teide snow column.

## 3. Result — savepoint parity vs pristine WRF (`proofs/v090/noahmp_t2mb_parity.json`)

**11 / 11 columns PASS**, predeclared fp64 tol 0.05 K, **worst |jax−wrf| = 1.08e-3 K**.

| column            | case   | fveg | t2mv Δ (K) | t2mb Δ (K) | t2 Δ (K) |
|-------------------|--------|------|-----------|-----------|----------|
| daytime_veg5      | day    | 0.346| — (uses below) | -1.3e-5 | +1.3e-5 |
| daytime_veg9      | day    | 0.259| +5.5e-5 | -1.1e-5 | +1.8e-5 |
| daytime_veg10     | day    | 0.293| -3.4e-5 | +3.2e-5 | +2.5e-5 |
| daytime_veg13     | day    | 0.000| (bare)  | -1.7e-5 | -1.7e-5 |
| daytime_veg16     | day    | 0.000| (bare)  | +1.0e-5 | +1.0e-5 |
| **nighttime_veg5**  | night | 0.346| -2.2e-4 | -1.2e-5 | **-7.3e-5** |
| **nighttime_veg9**  | night | 0.259| +7.6e-4 | -1.1e-5 | **+1.9e-4** |
| **nighttime_veg10** | night | 0.293| +1.1e-3 | -7.4e-6 | **+3.2e-4** |
| **nighttime_veg13** | night | 0.000| (bare)  | -1.7e-5 | **-1.7e-5** |
| **nighttime_veg16** | night | 0.000| (bare)  | +9.8e-6 | **+1.0e-5** |
| teide_snow        | snow   | 0.000| (bare)  | +3.5e-6 | +3.5e-6 |

The residual (~1e-3 K worst, on a vegetated-nocturnal T2MV) is at the fp64 transcription
floor — T2MV/T2MB are direct algebraic functions of TAH/TGB/SHG/SHC/SHB/FV/FH2, all of
which the energy gate already matches to <1e-3 K. **The stable-nocturnal land T2 (the +2.8 K
crux) resolves to ≤3.2e-4 K vs WRF.** Energy gate unchanged: 11/11 PASS (no regression).

WRF reports `T2MV=0.0` for `FVEG=0` columns (VEGE_FLUX skipped, T2MV stays at its :2047
init); the JAX `_vege_flux` always runs so its t2mv is a live but DISCARDED value there —
the combine `where(use_veg, …)` takes only t2mb when FVEG=0, exactly the driver else-branch,
so T2 is faithful regardless. The gate therefore compares t2mv only where FVEG>0.

## 4. Wiring + stand-in retirement

- `NoahMPEnergyFluxes` / `NoahMPFluxes` gain `t2mv/t2mb/t2` (ADDITIVE, default None — same
  patch-protocol pattern as the S1 `pahv..foln` amendment).
- `noahmp_surface_hook.overlay_noahmp_land_diagnostics`: with `bulk_t2` supplied it returns
  `(hfx, lh, tsk, t2)` and routes `nm.t2` over land / `bulk_t2` over water (the WRF
  overwrite); `bulk_t2=None` keeps the legacy 3-tuple for un-wired callers.
- `runtime.operational_mode` (recompute M9): passes `bulk_t2=surf.t2`, reports the
  land-overwritten T2 in `M9Diagnostics`. Water keeps the bulk surface-layer T2.
- `surface_layer.py`: REMOVED `_noahmp_bare_2m_weight_stable` and the `lsm_t2_diag` branch.
  The MYNN-SL 2-m diagnostic is now the pure module_sf_mynn.F `psit2/psit` everywhere (the
  land overwrite is owned by the coupler). `physics_couplers.lsm_t2_diag` is now inert
  (kept so legacy constructors don't break; surface_layer no longer reads it).

The empirical stand-in is **retired**; the faithful T2MB/T2MV path is the default.

## 5. Confirmation status (honest)

- **Savepoint (PASS):** T2MV/T2MB/T2 vs pristine WRF, 11/11, worst 1.08e-3 K, including all
  stable-nocturnal columns. This is the binding physics proof.
- **Operational-hook (PASS):** `tests/test_v090_noahmp_t2_overwrite.py` confirms the
  operational hook routes the Noah-MP LSM T2 over land and the bulk T2 over water (and keeps
  the legacy 3-tuple). 33 passed / 1 skipped across the surface/Noah-MP/MYNN test sweep.
- **Coupled forecast (NOT executed):** I did not run a coupled d02/d03 forecast to measure
  the net operational land-T2 shift. The savepoint + hook tests prove the diagnostic and the
  masking; a coupled GPU confirmation run measuring overnight land-T2 RMSE vs wrfout/obs is
  the recommended next gate (and is where the v0.1.0 stand-in's 1.64→0.72 K RMSE benefit
  should now be recovered faithfully, since the wrfout land T2 *is* this LSM-overwritten
  value).

## 6. Honest risks / caveats

- **Q2 land overwrite not wired.** WRF also overwrites land Q2 with the LSM
  `Q2 = FVEG*Q2V + (1-FVEG)*Q2B`. I implemented and oracle-captured Q2V/Q2B in the driver
  but did NOT compute Q2V/Q2B in the JAX port or route them — out of the T2 scope and the
  operational Q2-over-land lever is weaker than T2. The hook comment notes this as the
  symmetric follow-up. Surface-layer Q2 remains the faithful MYNN `psiq2/psiq`.
- **Coupled re-run pending** (see §5). The net obs-skill change over land is unmeasured here.
- The T2MV/T2MB use the SFCDIF1-converged FH2/FV frozen per-column at each column's last
  sweep (matching WRF's finite-iteration LITER semantics); the 11/11 PASS confirms the
  freeze is correct, but it inherits the same ~1e-3 K SFCDIF1 fixed-point floor the energy
  gate carries — well below any operational tol.

## 7. Files

- IMPL: `src/gpuwrf/physics/noahmp/energy.py` (T2MV/T2MB + combine),
  `src/gpuwrf/physics/noahmp/types.py`, `src/gpuwrf/contracts/noahmp_state.py`,
  `src/gpuwrf/physics/noahmp/noahmp_driver.py`
- WIRING: `src/gpuwrf/coupling/noahmp_surface_hook.py`,
  `src/gpuwrf/runtime/operational_mode.py`
- RETIRE: `src/gpuwrf/physics/surface_layer.py`, `src/gpuwrf/coupling/physics_couplers.py`
- ORACLE: `proofs/noahmp/noahmp_offline_driver.F90` (+T2DIAG),
  `proofs/noahmp/build_noahmp_savepoints.py`, `proofs/noahmp/savepoints_energy.json`,
  `proofs/noahmp/savepoints_all.json`
- PROOF: `proofs/v090/noahmp_t2mb_parity.py` → `proofs/v090/noahmp_t2mb_parity.json`
- TEST: `tests/test_v090_noahmp_t2_overwrite.py`
