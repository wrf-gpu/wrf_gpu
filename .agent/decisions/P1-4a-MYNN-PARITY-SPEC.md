# P1-4a — MYNN / HFX full surface-layer parity (implementation spec)

Status: SPEC ONLY (prep). Closes GPT review findings **G1** and **G3**
(`.agent/reviews/2026-05-31-gpt-hfx-and-proof-review.md`, findings #1 and #3).
**Do not implement until after the v0.1.0 tag.** v0.1.0's d02/d03 proofs are pinned to the
current `src/gpuwrf/physics/surface_layer.py` (commit `d1c373b`); editing it now un-validates the
release. This document is the exact change-list + WRF refs; the code edit + GPU oracle-parity run
are the 0.1.1 task.

## Scope

The current land `z_t` heat/moisture roughness block in `surface_layer.py` (lines 487–571,
landed `d1c373b`) is an **empirical partial** MYNN repair: it ports the Zilitinkevich-1995 land
thermal roughness onto the heat/moisture profiles and collapses the +3.6 K midday T2 warm bias to
~+1 K, but it is **not** a faithful `module_sf_mynn.F` port. Three concrete algorithmic mismatches
remain. This spec specifies the WRF-faithful form of each.

### Reference Fortran (pristine WRF v4)

- `/home/enric/src/wrf_pristine/WRF/phys/module_sf_mynn.F` — subroutine `SFCLAY1D_mynn`
  (decl `370`, body `~440`–`1207`); helper functions `zolrib` (`1984`), `zilitinkevich_1995`
  (`1209`), `psim_stable`/`psih_stable`/`psim_unstable`/`psih_unstable` (`2197`–`2271`).
- Cross-confirmed identical in `/home/enric/src/wrf_pristine/WRF/phys/physics_mmm/sf_mynn.F90`
  (`zolrib` `1894`; land `psih2` on momentum baseline `613`/`626`/`697`/`708`;
  `zilitinkevich_1995` `999`; the `zolrib(... z_t, gz1oz0, gz1ozt ...)` call `590`/`675`).

The Canary L3 corpus ran `sf_sfclay_physics=5` = this MYNN surface layer. **The GPU port must
reproduce the MYNN algorithm, not sfclayrev with a thermal-roughness graft.**

### Target GPU file

`src/gpuwrf/physics/surface_layer.py`, function `surface_layer_with_diagnostics` (the land-path
block lines ~405–571, plus the `_zolri` helpers at lines 174–235).

---

## Mismatch 1 — `zol` (Richardson → z/L) must be solved with the THERMAL roughness `z_t` in the heat term

### What WRF does

MYNN computes `z_t`/`z_q` **before** the stability solve (`module_sf_mynn.F:671–760`: `restar`,
the `zilitinkevich_1995` call, then `GZ1OZt = LOG((ZA+ZNTstoch)/z_t)` at `756`). The z/L solve is
then `zolrib(br, ZA, ZNTstoch, z_t, GZ1OZ0, GZ1OZt, ZOL, psi_opt)` for **both** the stable
(`module_sf_mynn.F:804`) and unstable (`889`) regimes. Inside `zolrib`
(`module_sf_mynn.F:1984–2048`) the iteration is:

```
zol20 = zolold*z0/za        ! z0 (momentum) / L
zol3  = zolold + zol20      ! (za+z0)/L
zolt  = zolold*zt/za        ! z_t (thermal) / L
psit2 = MAX(logzt - (psih(zol3) - psih(zolt)), 1.0)   ! logzt = GZ1OZt, uses z_t
psix2 = MAX(logz0 - (psim(zol3) - psim(zol20)), 1.0)  ! logz0 = GZ1OZ0, uses z0
zolrib = ri * psix2**2 / psit2
```

i.e. the **heat** residual term `psit2` is built from the thermal-roughness log `GZ1OZt` and the
`psih` difference taken about `zolt = z_t·zol/za` (NOT `z0`). The momentum term `psix2` keeps the
momentum roughness. Convergence: `|zolold - zolrib| <= 0.01`, `nmax = 20`; on non-convergence,
fall back to `Li_etal_2010(zolrib, ri, za/z0, z0/zt)` (`module_sf_mynn.F:2039`).

Note the first guess differs by regime (`module_sf_mynn.F:792–799`, `877–884`): for `itimestep<=1`
use `Li_etal_2010`; otherwise `ZOL = ZA·KARMAN·G·MOL/(TH1D·max(UST²,…))` clamped to `[0,20]`
(stable) / `[-20,0]` (unstable). The brute-force `zolrib` then refines from that guess. The first
guess only sets `zol1`; the converged result is dominated by `zolrib`, so the GPU port may keep its
existing endpoint seeding (`x1`/`x2` = `-5/0` unstable, `0/5` stable) **provided** the `zolrib`
residual itself is corrected to the `z_t` heat term (the residual, not the seed, sets the root).

### What the GPU does today (wrong)

`surface_layer.py:405–417` solves `zol` via `_zolri(br, za, znt)` — momentum roughness only, and
**before** the land `z_t` block (which is computed later at `509–529`). The `_zolri2` residual
(`174–192`) builds **both** `psix2` and `psih2` from `log((z+z0)/z0)` (momentum) — there is no
thermal-roughness heat term at all. So z/L is solved as if heat and momentum shared the roughness;
WRF solves z/L with a split heat (`z_t`) / momentum (`z0`) residual.

### Required change

1. **Move the `z_t` (and `z_q`) computation BEFORE the z/L solve.** Compute `restar`, `z_t_land`,
   `z_q_land` (and the water `z0t`/`z0q`) and the logs `gz1ozt`, `gz2ozt`, `gz10ozt` first, then
   solve z/L. This means the friction-velocity `ustar` used for `restar` must also be available
   before the solve (see Mismatch 2 — WRF uses the prior-step `UST`, which is an input, so this
   ordering is consistent: `restar` does not depend on the in-step `ustar` update).

2. **Add a thermal-roughness-aware residual.** Introduce a `_zolrib(ri, za, z0, zt, logz0, logzt)`
   helper that mirrors `module_sf_mynn.F:1984–2048` exactly:
   - `zol20 = zol*z0/za`, `zol3 = zol+zol20`, `zolt = zol*zt/za`.
   - `psit2 = max(logzt - (psih(zol3) - psih(zolt)), 1.0)` (use `_psih_unstable` for `ri<0`,
     `_psih_stable` for `ri>=0`).
   - `psix2 = max(logz0 - (psim(zol3) - psim(zol20)), 1.0)`.
   - residual / fixed-point update `zolrib = ri * psix2**2 / psit2`.
   WRF's `zolrib` is a **fixed-point iteration on `zolrib` itself** (`zolold = zolrib`,
   recompute), not the secant method `_zolri` uses. Two faithful options:
   - **(preferred)** Port the fixed-point loop directly: 20 vectorized trips of
     `zol_new = ri*psix2(zol_old)**2/psit2(zol_old)`, freezing the endpoint once
     `|zol_new - zol_old| <= 0.01`, seeded as WRF seeds (`zol1` from the first-guess block). This
     is the literal `zolrib`.
   - **(acceptable if it matches the oracle to tol)** keep the secant framework but feed it the
     thermal-aware `psit2` (the root of `ri·psix2²/psit2 - zol = 0` is identical). Must be
     validated bit-for-bit against the fixed-point `zolrib` on the oracle columns before adoption.

   **Land** uses `z_t` for the heat term; **water** uses the Fairall `z0t` as `zt` in `zolrib`
   (the water path also calls `zolrib(... z_t ...)` — `module_sf_mynn.F:804`/`889` are common to
   both land and water; only the `z_t`/`z_q` formula upstream differs). So the GPU port should
   solve z/L per-cell with `zt = where(is_land, z_t_land, z0t_water)`.

3. **Non-convergence fallback.** Mirror `Li_etal_2010(zolrib, ri, za/z0, z0/zt)` for cells that do
   not converge in 20 trips. (Li_etal_2010 is `module_sf_mynn.F:1831`+.) For the Canary operational
   columns the iteration converges; the fallback is for robustness parity, not the common path.

4. **Clamp** `zol` to `[0,20]` (stable) / `[-20,0]` (unstable) per `module_sf_mynn.F:805–806`,
   `890–891`. The existing `jnp.clip(zol, -20, 20)` at `surface_layer.py:417` is close but applies
   one symmetric clamp; WRF clamps per-sign (stable→[0,20], unstable→[-20,0]). Match per-sign.

---

## Mismatch 2 — `restar` must use the PRIOR-STEP `ustar`, not a blended/look-ahead value

### What WRF does

`module_sf_mynn.F:675` (water) and `725` (land): `restar = MAX(ust(i)*ZNTstoch(i)/visc, 0.1)`,
computed using `UST(i)` **as it enters the column** — i.e. the prior timestep's friction velocity
(an INTENT(INOUT) array carried across steps). The in-step update
`UST(i) = 0.5*UST(i) + 0.5*KARMAN*WSPD/PSIX` happens much later, at `module_sf_mynn.F:949`, AFTER
`restar`/`z_t` and AFTER the z/L solve. `OLDUST = UST(I)` is saved at `948` for reference but
`restar` already used the pre-update value. So `z_t` lags `ustar` by exactly one timestep — this is
intentional in WRF.

### What the GPU does today (wrong)

`surface_layer.py:505–507` diagnoses a *fresh* `ustar` (`ust_fresh = KARMAN*wspd/psix`) and forms
`ustar = where(cold_start, ust_fresh, 0.5*ust_in + 0.5*ust_fresh)` **before** the land `z_t` block,
specifically so `restar_l` (line 526) sees the spun-up value. The accompanying comment
(`496–504`) argues this removes a "one-step lag." That is exactly the look-ahead GPT finding #1
flags: WRF *keeps* that lag. The blended/fresh `ustar` feeding `restar` is not WRF-faithful.

### Required change

1. **`restar` uses `ust_in` (the prior-step `ustar` input), floored at `0.1`:**
   `restar_l = max(ust_in * znt / visc_l, 0.1)` — matching `module_sf_mynn.F:725`. Remove the
   `ust_fresh`/blend feeding `restar`.
2. **Compute `restar`/`z_t` (Mismatch 1 ordering) using `ust_in`** — this is consistent: `z_t` is a
   prior-step-`ustar` quantity in WRF, so moving it before the z/L solve does not need the in-step
   `ustar`.
3. **Keep the in-step `ustar` update at its WRF position** (`module_sf_mynn.F:945–962`), i.e. after
   the z/L solve and PSIM are known: `ustar = 0.5*ust_in + 0.5*KARMAN*wspd/psix`, then the land
   floor `ustar = max(ustar, 0.005)` (`959`) — note WRF's land floor is **0.005**, the current code
   uses `max(ustar, 0.001)` at `surface_layer.py:666`; align to `0.005`. The cold-start
   `where(ust_in<=0.001, ust_fresh, blend)` special-case (`506–507`) is **not** in WRF; WRF always
   blends. Removing it changes the cold-start (`itimestep<=1`) friction velocity. **Decision for
   0.1.1:** the operational replay path is warm-started (real `UST` carried from the corpus
   wrfout), so `ust_in` is physical and the blend is correct; drop the cold-start branch to match
   WRF, and verify the oracle (which feeds same-step `UST`) still matches — if the oracle's UST
   input represents "prior step," the plain blend is what WRF would do. Document the chosen
   semantics in the 0.1.1 commit.

---

## Mismatch 3 — `PSIH2`/`PSIH10` use the MOMENTUM-roughness baseline; only `PSIH` uses the thermal baseline

### What WRF does

In every regime block, `PSIH` is taken about the **thermal** roughness `zolzt` but `PSIH2` and
`PSIH10` are taken about the **momentum** roughness `zolz0`:

- Stable (`module_sf_mynn.F:823–827`):
  ```
  psih(I)  = psih_stable(zolza) - psih_stable(zolzt)    ! zolzt = thermal
  psih10(I)= psih_stable(zol10) - psih_stable(zolz0)    ! zolz0 = momentum
  psih2(I) = psih_stable(zol2)  - psih_stable(zolz0)    ! zolz0 = momentum
  ```
- Unstable (`module_sf_mynn.F:907–911`, `918–922`): identical structure with `psih_unstable`.
- Thin-layer caps (`module_sf_mynn.F:931–935`): `PSIH = MIN(PSIH, 0.9*GZ1OZt)`,
  `PSIH2 = MIN(PSIH2, 0.9*GZ2OZt)`, `PSIH10 = MIN(PSIH10, 0.9*GZ10OZt)`,
  `PSIM = MIN(PSIM, 0.9*GZ1OZ0)`, `PSIM10 = MIN(PSIM10, 0.9*GZ10OZ0)`. NOTE the cap on `PSIH2`/
  `PSIH10` uses the THERMAL log (`GZ2OZt`/`GZ10OZt`) even though the PSIH2/PSIH10 *value* was built
  on the momentum baseline `zolz0` — the cap and the baseline disagree on purpose in WRF.

Then the resistances (`module_sf_mynn.F:969–977`, repeated `1020–1025`):
```
GZ1OZt = LOG((ZA+ZNTstoch)/z_t)
GZ2OZt = LOG((2.0+ZNTstoch)/z_t)
PSIT   = MAX(GZ1OZt - PSIH , 1.)
PSIT2  = MAX(GZ2OZt - PSIH2, 1.)
PSIQ   = MAX(LOG((ZA+ZNTstoch)/z_q) - PSIH , 1.0)
PSIQ2  = MAX(LOG((2.0+ZNTstoch)/z_q) - PSIH2, 1.0)
PSIQ10 = MAX(LOG((10.0+ZNTstoch)/z_q) - PSIH10, 1.0)
```
So: `PSIT`/`PSIQ` carry the thermal-baseline `PSIH`; `PSIT2`/`PSIQ2` carry the **momentum-baseline**
`PSIH2`; `PSIQ10` carries the momentum-baseline `PSIH10`. The `log((…)/z_t|z_q)` numerators always
use the thermal/moisture roughness. The 2 m / 10 m diagnostics (`CHS2 = UST·K/PSIT2`,
`CQS2 = UST·K/PSIQ2`, `Q2`, `TH2`) therefore mix a thermal numerator with a momentum-baseline
`PSIH2`.

### What the GPU does today (wrong)

`surface_layer.py:531–571`: the helper `_psih_zt(z0x, height)` subtracts the **thermal** baseline
`zol*z_t/za` for **all** heights (line 535: `z0 = zol*z0x/za` with `z0x = z_t_land`), then
`psih_zt`, `psih2_zt`, `psih10_zt` are all thermal-baseline (`545–547`), and
`psih2 = where(is_land, psih2_zt, …)` / `psih10 = where(is_land, psih10_zt, …)` (`570–571`). That
makes the GPU `PSIH2`/`PSIH10` use the thermal baseline — wrong; WRF uses momentum (`zolz0`) for
those two.

### Required change

For the **land** path, split the heat-psi baselines:

- `psih` (lowest level): thermal baseline — `psih_stable/unstable(zolza) - psih_…(zolzt)`,
  where `zolzt = zol*z_t/za`. (This is what `_psih_zt(z_t, za)` already does for the lowest level —
  keep it.)
- `psih2` / `psih10`: **momentum** baseline — `psih_…(zol2) - psih_…(zolz0)` and
  `psih_…(zol10) - psih_…(zolz0)` with `zolz0 = zol*znt/za`. These are **the same `psih2_s/psih2_u`
  and `psih10_s/psih10_u` already computed for the water/momentum path at
  `surface_layer.py:437–438,448`,`436,447`** — so the land `psih2`/`psih10` should reuse those
  momentum-baseline values rather than the thermal `_psih_zt` ones. Concretely: do **not** overwrite
  `psih2`/`psih10` with `psih2_zt`/`psih10_zt` over land (delete the `where(is_land, …)` at
  `570–571`); keep only `psih = where(is_land, psih_zt, psih)` at `569`.
- Caps: apply `psih2 = MIN(psih2, 0.9*GZ2OZt)` and `psih10 = MIN(psih10, 0.9*GZ10OZt)` using the
  THERMAL logs (`gz2ozt`, `gz10ozt`) as WRF does at `module_sf_mynn.F:933,935`, even though the
  value is momentum-baseline. (The lowest-level `psih` cap uses `0.9*GZ1OZt` — already correct at
  `surface_layer.py:550`.) The momentum-baseline `psih2`/`psih10` already received the `0.9*gz2oz0`/
  `0.9*gz10oz0` cap in the generic unstable block (`456–458`); the THERMAL re-cap (`0.9*GZ2OZt`)
  must REPLACE that for land, matching WRF.
- Resistances: then `PSIT2 = MAX(GZ2OZt - psih2, 1.)`, `PSIQ2 = MAX(LOG((2+ZNT)/z_q) - psih2, 1.)`,
  `PSIQ10 = MAX(LOG((10+ZNT)/z_q) - psih10, 1.)` (`module_sf_mynn.F:973,976,977`). Numerators keep
  the thermal/moisture roughness log; only the subtracted PSIH switches to momentum baseline for the
  2 m / 10 m terms.

### Also align the top log-argument (sfclayrev-vs-MYNN nit, GPT finding #1 tail)

GPT noted the current `_psih_zt` uses `log((height+znt)/z_t)` (`surface_layer.py:542–544`). MYNN's
`GZ?OZt = LOG((height + ZNTstoch)/z_t)` (`module_sf_mynn.F:756–760`) — i.e. the numerator is
`height + z0(momentum)`, divided by `z_t`. **The current GPU code is already correct on this point**
(`(za+znt)/z_t_land`, etc.), matching MYNN — keep it. (The sfclayrev `iz0tlnd>=1` variant uses
`(height+z0t)`; we are matching MYNN, not that variant, so no change.) Record this so the 0.1.1
reviewer does not "fix" it back to sfclayrev.

---

## Out of scope (explicitly NOT ported in P1-4a — Canary release path)

- **Snow/ice thermal roughness — `Andreas_2002`** (`module_sf_mynn.F:731–732,1559`). Triggers only
  for `SNOWH >= 0.1`. The Canary MAM corpus is no-snow; the land branch stays on
  `zilitinkevich_1995`. Do not port `Andreas_2002`.
- **Stochastic roughness perturbations (`spp_pbl=1`)** (`module_sf_mynn.F:664–669,714–719`;
  `zilitinkevich_1995:1267–1273`). The corpus runs `spp_pbl=0`, so `ZNTstoch = ZNT` and the
  perturbed `z_t` floors (`MAX(Zt,0.0001)`) never activate. Do not port.
- **`IZ0TLND` alternatives `Yang_2008` / `garratt_1992` and the `IZ0TLND=1` CZIL law**
  (`module_sf_mynn.F:738–743,1255–1256`). The default `IZ0TLND<=1` with `CZIL=0.085` is what the
  corpus ran; keep the single Zilitinkevich `CZIL=0.085` branch.
- **`isftcflx>0` dissipative-heating term over water** (`module_sf_mynn.F:1067–1072`). Default
  `isftcflx=0`; not added.

### Remove the non-MYNN land floor (GPT finding #9)

`surface_layer.py:529` applies `z_t_land = max(z_t_land, 2.0e-9)`. WRF's default land
`zilitinkevich_1995` (`module_sf_mynn.F:1261–1265`) applies **only** `Zt = MIN(Zt, 0.75*Z_0)` with
NO lower floor (the `2e-9` floor exists only on the WATER branch, `1239/1242/1246/1249`, and on the
`spp_pbl=1` land path as `MAX(Zt,0.0001)`). Drop the `2e-9` land floor to match WRF; keep the
`MIN(z_t, 0.75*znt)` cap (already present at `528`).

---

## Acceptance for 0.1.1 (when implemented)

1. The land path solves z/L with the `z_t` heat residual (`_zolrib`), uses prior-step `ust_in` for
   `restar`, and uses the momentum baseline for `PSIH2`/`PSIH10` (thermal for `PSIH` only).
2. External-oracle parity (NOT self-compare): `proofs/v010_validation/sfclay_hfx_oracle_parity.py`
   extended to validate **HFX, LH, QFX, Q2, MOL, PSIT, PSIQ, U10, V10, PBLH** vs same-step WRF over
   land/water × stable/unstable (see `sfclay_mynn_full_parity.py`). The standalone-vs-Noah-MP land
   HFX residual is expected and is the LSM surface-energy-balance coupling, not a scheme bug — the
   binding scheme metrics are T2/Q2/U10/V10/MOL/PSIT/PSIQ and the water HFX/LH.
3. No regression on the d02 3-day validation (T2, Q2/LH, U10/V10, PBLH) and the d03 24 h run.
4. Claim downscoped/upgraded: only after parity passes may the proof say "WRF-faithful MYNN land
   thermal-roughness"; until then it stays "empirical partial MYNN repair."

## Files this 0.1.1 task will touch (NOT touched by this prep)

- `src/gpuwrf/physics/surface_layer.py` — the changes above. (PREP MUST NOT touch this.)
- `proofs/v010_validation/sfclay_mynn_full_parity.py` — the harness (created by this prep, run
  post-tag).
