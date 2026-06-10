# V0.14 Fable — Noah-MP Land-Tile Energy Closure

Date: 2026-06-10 WEST · Owner: Fable/Mythos · Branch: `worker/gpt/v013-close-manager`
Base: `43accdc6` (manager committed the prior sprint as `203be499`)
Contract: `.agent/sprints/2026-06-10-v014-fable-noahmp-energy-closure/sprint-contract.md`

## Verdict

`NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING`

The prior blocker (`...NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`) is **refuted and
resolved**. The JAX Noah-MP energy solve is exact; the real defect was a missing
moist-θ→dry-T decoupling in the air-temperature fed to Noah-MP, which I **found
and fixed in production**. Strict Step-1 remains RED, now precisely localized to
two out-of-scope lanes (RRTMG radiation forcing; the same decoupling in the
sfclay/MYNN water path) — this is the contract's acceptable fallback, with the
fix delivered on top.

## Objective

Close the strict Step-1 Noah-MP land-tile energy blocker, or return an exact
WRF-anchored blocker narrower than "Noah-MP land-tile energy".

## What I did

1. **Per-column WRF `noahmplsm` energy hook** (NMPIN/NMPOUT) on the rmol-pinned
   tree (`module_sf_noahmpdrv.F`): full energy in/out per land cell — FVEG/LAI/SAI,
   CM/CH, two-stream SAV/SAG/FSA/FSR/SALB, SH/EV/GH/FIRA/TRAD/T2MV/T2MB, EFLXB,
   STC. Env-gated `WRFGPU2_V014_NOAHMP_ENERGY_HOOK`. Recompiled the object with
   the wrf-build `mpif90` (the exact original toolchain), relinked `wrf.exe`,
   emitted in ONE run (~13 s). The surface-handoff hook re-emitted in the SAME run
   is **byte-identical** to the prior pinned-onerun handoff → the energy
   instrumentation does not perturb WRF physics.

2. **Energy-algorithm exoneration.** Fed WRF's EXACT per-column NMPIN into the JAX
   `physics.noahmp` chain (phenology → two-stream radiation → energy). Every output
   matches WRF NMPOUT on land cells to ~1e-3 W/m²: **FSH rmse 7.7e-4** (ref max
   370.6), SSOIL rmse 2.2e-4, TRAD rmse 2.0e-5 K, SAV/SAG ~2.5e-3, SALB ~5e-7. The
   solve is correct — the prior narrowing was wrong.

3. **ROOT CAUSE found + FIXED (production).** The lowest-level air temperature fed
   to Noah-MP was **+4.06 K too warm** (rmse 4.12). `state.theta` is the WRF MOIST
   potential temperature `θ_m = θ_dry·(1 + R_v/R_d·q_v)` (use_theta_m=1; the dycore
   prognostic — `operational_mode` `conv_t_tendf_to_moist` divides `before.theta`
   by the same `1+_RVRD·qv`). `assemble_noahmp_forcing` converted θ_m → T with a
   **naive Exner**, treating moist θ as dry, so the air temperature was high by
   exactly the `(1+R_v/R_d·q_v)` factor (verified to machine precision: identity
   error 2.2e-16). WRF feeds `noahmplsm` the dry sensible temperature
   `T3D = θ_dry·(p/p0)^κ` (`module_sf_noahmpdrv.F:755`). **Fix:** decouple
   θ_m → θ_dry before the Exner conversion. sfctmp vs WRF T_ML: **+4.06 K bias →
   0.0033 K rmse**.

4. **Causal split confirms the fix + isolates the remainder.** With the fix, the
   real-overlay land-tile HFX residual is **rmse 7.6 W/m²** (bias +6.5); swapping
   WRF's EXACT SWDOWN/GLW in **collapses it to rmse 0.097 W/m²** (TSK 0.0014 K).
   So the entire remaining land-tile residual is the RRTMG radiation forcing:
   **GLW +14.7 W/m²** (clear-sky), **SWDOWN +3.6 W/m²** — an out-of-scope lane
   (RRTMG production frozen this sprint).

5. **Propagation to the contract proof.** Re-ran `noahmp_step1_closure.py` with the
   fixed coupler: post-overlay MYNN-boundary land theta_flux rmse **0.062 → 0.0076**,
   and the WRF-radiation swap now COLLAPSES it (0.0076 → 0.0013). Strict rmse
   **13.20 → 12.15**; max_abs unchanged at 1489.5 because the worst cell
   (i=66, j=37, k=3; WRF −2457.6 vs JAX −968) is a **WATER** column — Noah-MP does
   not run there. That cell is sfclay/MYNN + RTHRATEN, where `surface_layer.py`
   uses the SAME naive θ_m→T conversion (out of this sprint's ownership).

## Strict gate (honest)

`after_conv T_TENDF` vs JAX dry source leaf, vs pinned one-run truth:
- max_abs **1489.51** (worst cell i=66 j=37 = water), rmse **12.15** (was 13.20),
  p95 2.01, p99 48.0. Pass target max_abs ≤ 1e-3 / rmse ≤ 1e-5 → **RED**.
- Not closable within this sprint's file ownership: the residual is now (a) RRTMG
  GLW/SWDOWN forcing and (b) the sfclay/MYNN moist-θ decoupling over water.

## Files changed

- PRODUCTION (fix, not default-inert — corrects a real +4 K warm bias on the
  Noah-MP path):
  - `src/gpuwrf/physics/noahmp_coupler.py` — `assemble_noahmp_forcing` decouples
    θ_m → θ_dry (`/(1+R_v/R_d·q_v)`, `RVOVRD = 461.6/R_D`) before the Exner
    conversion of the lowest-level air temperature; new module constant `RVOVRD`.
- TEST:
  - `tests/test_noahmp_coupler.py` — `test_forcing_decouples_moist_theta_to_dry_air_temperature`
    (pins the moist-θ_m convention; asserts sfctmp recovers dry T3D and the bug
    factor is exactly `1+R_v/R_d·q_v`).
- PROOF:
  - `proofs/v014/noahmp_land_tile_energy_closure.{py,json,md}` (NEW) — the energy
    closure proof (hook provenance + non-perturbation byte-cmp, energy-algorithm
    exoneration, forcing-vs-WRF, decoupling fix, flux closure + radiation swap,
    ranked lanes).
  - `proofs/v014/noahmp_step1_closure.{py,json,md}` — re-run on the fixed coupler;
    refreshed ranked hypotheses / verdict narrative (energy lane closed; RRTMG +
    sfclay-water lanes).
- WRF truth instrumentation (in `/tmp`, not in-repo): the per-column energy hook in
  the pinned `module_sf_noahmpdrv.F`; truth at
  `/tmp/wrfgpu2_v014_noahmp_energy_pinned_onerun/`.

## Commands run

```bash
# instrument + rebuild + one-run emit (WRF, wrf-build mpif90 toolchain)
#   3 env-gated blocks in module_sf_noahmpdrv.F; recompiled object; relinked wrf.exe
cd /tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/run && env \
  WRFGPU2_V014_NOAHMP_ENERGY_HOOK=1 \
  WRFGPU2_V014_NOAHMP_ENERGY_ROOT=/tmp/wrfgpu2_v014_noahmp_energy_pinned_onerun \
  WRFGPU2_V014_SURFACE_HANDOFF_HOOK=1 \
  WRFGPU2_V014_SURFACE_HANDOFF_ROOT=/tmp/wrfgpu2_v014_surface_handoff_energyhook_xcheck \
  taskset -c 0-3 ./wrf.exe
# gates
python -m py_compile proofs/v014/noahmp_step1_closure.py proofs/v014/noahmp_land_tile_energy_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_land_tile_energy_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.validated.json
python -m json.tool proofs/v014/noahmp_land_tile_energy_closure.json >/tmp/noahmp_land_tile_energy_closure.validated.json
git diff --check
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q \
  tests/test_noahmp_coupler.py tests/test_v014_mynn_surface_layer_regressions.py \
  tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py \
  tests/test_v014_mynn_coldstart_init.py
```

## Proof objects

- `proofs/v014/noahmp_land_tile_energy_closure.json` — verdict
  `NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING`;
  energy FSH rmse 7.7e-4 (WRF inputs); sfctmp 4.06 K → 0.0033 K; HFX 7.6 → 0.097
  on radiation swap; GLW +14.7 / SWDOWN +3.6.
- `proofs/v014/noahmp_step1_closure.json` — strict rmse 12.15 (was 13.20), land
  flt 0.0076 (was 0.062), swap collapses, verdict
  `..._NARROWED_TO_RADIATION_FORCING_INTO_NOAHMP`.

## Unresolved risks / next decision

- **Strict stays RED** by the two out-of-scope lanes:
  1. **RRTMG step-1 GLW +14.7 / SWDOWN +3.6 W/m² forcing** — needs an RRTMG
     longwave/shortwave forcing-parity hook; RRTMG production is frozen this sprint.
  2. **`surface_layer.py` (sfclay/MYNN) moist-θ→dry-T decoupling over WATER** — the
     SAME bug I fixed for Noah-MP also lives in
     `surface_layer._potential_to_temperature`; it drives the strict worst (water)
     cell. `surface_layer.py` is outside this sprint's ownership AND it is the
     validated MYNN path — a "fix" there must be paired with d02 MYNN
     re-validation (the bulk scheme may partially cancel the offset in
     potential-temperature differences). **Recommend a dedicated sprint.**
- The energy hook + truth tree live in `/tmp` (volatile); reproducible via the
  documented one-run command + the archived instrumentation.
- The decoupling fix assumes `use_theta_m=1` (the Canary/WRF default, and the only
  config Noah-MP runs under here); it is consistent with the unconditional moist
  coupling already in `operational_mode._physics_step_forcing`.

## Gate status

py_compile ✅ · energy-closure proof ✅ (verdict above) · step1 proof ✅ ·
json.tool ×2 ✅ · `git diff --check` ✅ · pytest subset **17 passed, 1 skipped**
(pre-existing skip). Working tree intentionally UNCOMMITTED; manager reviews /
commits / merges.
