# diff_opt=1 + km_opt=4 (2-D Smagorinsky horizontal diffusion) — implementation review

**Branch:** `worker/opus/v090-diffopt1-smagorinsky` (base `worker/opus/trunk-0.9.0` @ 7b7c26e)
**Date:** 2026-06-04
**Author:** Opus 4.8 (1M)
**Reference (oracle):** unmodified WRF v4 `dyn_em/module_diffusion_em.F` +
`dyn_em/module_big_step_utilities_em.F` + `dyn_em/module_em.F` at
`/home/enric/src/wrf_pristine/WRF`.

## Objective

Wire WRF's **recommended real-data default** horizontal diffusion — `diff_opt=1`
(horizontal diffusion evaluated along coordinate/eta surfaces) + `km_opt=4`
(2-D Smagorinsky horizontal eddy viscosity from the horizontal deformation) — as
an additional selectable option, without breaking the existing
`diff_opt=2`/`km_opt=1` constant-K path that the idealized Straka case depends on.
Before this change a stock WRF real-data namelist (`diff_opt=1`, `km_opt=4`)
fail-closed in `namelist_check.py`.

## What WRF actually does (the spec I transcribed)

1. **`km_opt=4` (`smag2d_km`, `module_diffusion_em.F:1934-2044`)** computes the
   horizontal eddy viscosity from the horizontal deformation:
   - `def2 = 0.25*(D11-D22)^2 + tmp^2`, where `tmp` is the 4-corner average of
     `D12` onto the mass cell (`:2004-2007`).
   - `mlen_h = sqrt(dx/msftx * dy/msfty)` (`:2015`) → `sqrt(dx*dy)` for msf=1.
   - `xkmh = c_s^2 * mlen_h^2 * sqrt(def2)`, then `xkmh = min(xkmh, 10*mlen_h)`
     (`:2018-2019`). `c_s` default 0.25 (Registry).
   - `xkhh = xkmh / prandtl`, `prandtl = 1/3` (`:2021`) → heat K is exactly 3× momentum K
     (matches the WRF `khdq = 3*khdif` convention).
   - The slope-factor reduction (`:2023-2039`) is gated on `diff_opt==2`, so the
     `diff_opt=1` path carries **no** slope reduction.
2. **Deformations (`cal_deform_and_div`)**, flat slab (zx=zy=0, msf=1):
   `D11 = 2 du/dx`, `D22 = 2 dv/dy` (mass points); `D12 = du/dy + dv/dx`
   (vorticity/corner points), eqns 13a/13b/13d.
3. **`diff_opt=1` application (`module_em.F:802-878`, `rk_step==1`)** calls
   `horizontal_diffusion` for u/v/w (with `xkmhd`) and `horizontal_diffusion_3dmp`
   for theta (with `xkhh`, on the perturbation `t - t_init`). VERTICAL diffusion is
   gated on `bl_pbl_physics==0` (`:842`) — so the 2-D Smagorinsky adds **horizontal
   mixing only**; vertical mixing is the PBL scheme's job.
4. **`horizontal_diffusion(_3dmp)` (`module_big_step_utilities_em.F:2715-3060`)** is
   a variable-K mass-weighted flux divergence on coordinate surfaces; with unit map
   factors the face flux uses the arithmetic-averaged K and `mass = c1*MUT+c2`.

## Implementation

`src/gpuwrf/dynamics/explicit_diffusion.py` (new, additive — existing operators
untouched):
- `horizontal_deformation_2d(u, v)` → `(D11, D22, D12)` (flat periodic slab).
- `smag2d_horizontal_km(D11, D22, D12)` → `(xkmh, xkhh)` (literal `smag2d_km`).
- `horizontal_diffusion_coord_scalar_tendency(field, xkhh, mass, base_3d=...)`
  (`horizontal_diffusion_3dmp` perturbation form).
- `horizontal_diffusion_coord_momentum_tendency(u, v, w, xkmh, ...)` (u/v/w branches).
- `PRANDTL = 1/3`, `C_S_DEFAULT = 0.25`.

`src/gpuwrf/runtime/operational_mode.py`:
- New `c_s` namelist field (threaded through `from_grid` / `tree_flatten` /
  `tree_unflatten`).
- New `diff_opt==1 and km_opt==4` branch in `_augment_large_step_tendencies`,
  placed **after** the const-K `nu>0` block. It is a SEPARATE branch — the two
  diffusion paths never run together.

`src/gpuwrf/io/namelist_check.py` (minimal, to stay merge-clean with the parallel
consolidation lane):
- `diff_opt` accepts `{0, 1, 2}`; `km_opt` accepts `{0, 1, 4}`. `km_opt=2/3`
  (full 3-D TKE / 3-D Smagorinsky) still rejected.

## Verification (CPU fp64, no GPU)

Proof object: `proofs/v090/diffopt1_smagorinsky_parity.json` (generator
`proofs/v090/diffopt1_smagorinsky_parity.py`). **Method = analytic oracle**: every
operator is compared to a literal NumPy transcription of the WRF formula on smooth
periodic analytic fields. Max-abs residuals (verdict **PASS**):

| Check | Residual |
|---|---|
| D11 vs analytic `2 du/dx` | 4.9e-19 |
| D22 vs analytic `2 dv/dy` | 2.2e-18 |
| D12 vs analytic `du/dy+dv/dx` | 0.0 |
| `xkmh` vs WRF `smag2d_km` | 0.0 |
| `xkhh` vs WRF `smag2d_km` | 0.0 |
| `xkhh == 3*xkmh` (prandtl=1/3) | 3.6e-15 |
| coordinate-surface flux divergence vs WRF | 0.0 |
| mass-weighted integral conservation (per-level sum) | 4.2e-15 |

Tests (all PASS, `tests/dynamics/`):
- `test_diffopt1_smagorinsky.py` — 12 analytic-oracle unit tests (deformations,
  smag Kh + cap + prandtl, flux divergence + conservation + down-gradient,
  momentum staggers, jittability).
- `test_diffopt1_smagorinsky_integration.py` — 3 CPU integration tests:
  (a) **baseline bit-identical** when smag not selected (diff_opt=0 == diff_opt=0 with
  km_opt=4 set) → no regression; (b) finite down-gradient diffusion added to u/theta
  when selected; (c) full augment step jit-traceable.
- `test_deformation_momentum_diffusion.py` (existing) — 4 tests still PASS.
- Full `tests/dynamics/` + `tests/test_namelist_check.py` = **28 passed**.

## Answers to the gate questions

- **diff_opt=1/km_opt=4 implemented + verified?** Yes — analytic-oracle parity to
  machine epsilon + CPU integration.
- **Smagorinsky Kh matches WRF formula?** Yes, exactly (residual 0.0 vs the literal
  `smag2d_km` def2/mlen_h/c_s²/cap/prandtl formula).
- **Existing diff_opt=2 idealized cases still PASS (no regression)?** Structurally
  guaranteed (new branch is inert: idealized cases use `const_nu_m2_s`, leave
  `diff_opt`/`km_opt`=0) and **proven bit-identical** by the baseline-unchanged test.
  NOTE: the GPU-only idealized close gates (`run_warm_bubble_case` /
  `run_density_current_case`) were NOT re-run because the GPU was claimed by the
  parallel consolidation lane (one-GPU-job rule) — see Risks.
- **Namelist accepts the real-data default?** Yes — `diff_opt=1`+`km_opt=4` no longer
  fail-closed; unsupported `km_opt=2/3` still rejected.
- **WRF-faithful, no clamps?** Yes. The only ceiling is the literal WRF
  `min(xkmh, 10*mlen_h)` stability cap (`:2019`), faithfully transcribed — not a
  masking/tuning clamp.

## Scope / risks

- **Scope = flat periodic slab (msf=1, zx=zy=0).** This matches the documented scope
  of the existing const-K deformation path. For real terrain the WRF coordinate-slope
  terms (`zx du^/dpsi` etc. in the deformation) and the map-factor ratios in
  `horizontal_diffusion` are dropped; these are zero on flat terrain / unit msf but
  NONZERO on real Canary terrain. **A full-terrain (sloped-eta + map-factor) extension
  is required before this is exercised on a real-data run; it is not yet wired for
  curved/sloped grids.** This is the honest limitation.
- **The momentum u/v/w branches use the cell-face-averaged K** (vs WRF's u/v-point
  four-corner K averaging); on the unit-msf slab with smooth K these agree to the
  operator's 2nd order. Documented in the function docstring.
- **No GPU idealized-gate re-run** (GPU lock held by consolidation lane). No-regression
  rests on the structural inertness proof + bit-identical baseline test, which is
  strong but not the GPU dynamical gate. Re-run `run_density_current_case` /
  `run_warm_bubble_case` on GPU when the lock frees to close that empirically.

## Files changed

- `src/gpuwrf/dynamics/explicit_diffusion.py` (additive operators)
- `src/gpuwrf/runtime/operational_mode.py` (c_s field + dispatch branch)
- `src/gpuwrf/io/namelist_check.py` (accept diff_opt∈{0,1,2}, km_opt∈{0,1,4})
- `tests/dynamics/test_diffopt1_smagorinsky.py` (new, 12 tests)
- `tests/dynamics/test_diffopt1_smagorinsky_integration.py` (new, 3 tests)
- `proofs/v090/diffopt1_smagorinsky_parity.{py,json}` (proof object)
- `.agent/reviews/2026-06-04-opus-diffopt1-smagorinsky.md` (this file)
