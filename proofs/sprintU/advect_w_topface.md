# Sprint U / P1-5 — WRF advect_w top-face (lid) flux contribution

Date: 2026-05-29
Branch: `worker/opus/f7d-pressure-mass-fix`

## Finding being closed (GPT pre-close P1)

> For `vert_order == 3`, WRF `advect_w` computes the top vertical flux and adds a
> top-face tendency `tend(i,kde) = tend + 2*rdzu(ktf)*vflux(i,kde)`.  The JAX
> `_vertical_flux_div_w` filled interior `tend[1:nz]` and left the top-face
> tendency zero.  The idealized cases use `top_lid=True`, which masks this; the
> real path is not proven to be lid-only.

## WRF source (pristine v4.7.1)

`dyn_em/module_advect_em.F`, `advect_w`, `vert_order == 3` block (`:5996-6031`):

```
DO k=kts+2,ktf                              ! interior flux3 faces
  vel=0.5*(rom(i,k,j)+rom(i,k-1,j))
  vflux(i,k) = vel*flux3( w(k-2),w(k-1),w(k),w(k+1), -vel )
k=kts+1: vflux = 0.25*(rom(k)+rom(k-1))*(w(k)+w(k-1))      ! 2nd order
k=ktf+1: vflux = 0.25*(rom(k)+rom(k-1))*(w(k)+w(k-1))      ! TOP FACE, 2nd order  (:6014-6015)
DO k=kts+1,ktf
  tendency(i,k,j) = tendency - rdzu(k)*(vflux(k+1)-vflux(k))
k = ktf+1                                                  ! lid pickup           (:6025-6028)
  tendency(i,k,j) = tendency + 2.*rdzu(k-1)*vflux(i,k)
```

`rdzu` is passed `grid%rdn` by the caller (`module_em.F:594`), matching the JAX
`rdn=metrics.rdn` already used in `advect_w_flux`.

Index mapping (JAX 0-based; w faces 0..nz, kde=nz+1 in 1-based, ktf=nz):
* interior flux3: WRF `kts+2..ktf` → JAX `2..nz-1` (already implemented).
* top face: WRF `ktf+1=kde` → JAX `nz`; flux `vflux(nz)=vel_face(nz)*0.5*(w(nz)+w(nz-1))`.
* lid pickup: `tend(nz) += 2*rdn(ktf)*vflux(nz)` with `rdn(ktf)=rdn[nz-1]`.

## Implementation

`src/gpuwrf/dynamics/flux_advection.py::_vertical_flux_div_w` now takes
`top_lid: bool`:

* `top_lid=True` (rigid-lid idealized config): top-face flux stays 0, no lid
  pickup → byte-identical to the closed F7 idealized path.
* `top_lid=False` (open/top-damped real config): sets `vflux[nz]` (2nd order),
  includes it in the interior face-(nz-1) divergence (via `vflux(k+1)`), and adds
  the lid pickup `tend[nz] += 2*rdn[nz-1]*vflux[nz]`.

`advect_w_flux` threads `top_lid` through; `_augment_large_step_tendencies` passes
`top_lid=bool(namelist.top_lid)`. The real-case path (`_build_real_case`) sets
`top_lid=False`, so the open-top branch is active for real runs (P0-1).

## Validation

`tests/dynamics/test_advect_w_topface.py` (4 tests, all PASS):

* `test_topface_rigid_lid_zero_top_tendency` — `top_lid=True` gives EXACTLY zero
  top-face tendency (the idealized path is unchanged).
* `test_topface_open_top_matches_wrf_formula` — `top_lid=False` reproduces the
  WRF lid pickup `2*rdn(ktf)*vflux(kde)` to round-off on a hand-built column.
* `test_topface_interior_face_nz_minus_1_uses_top_flux` — the open-top vflux(nz)
  enters the interior face-(nz-1) divergence and nowhere below.
* `test_advect_w_flux_topface_flag_threaded` — `advect_w_flux` propagates the flag.

The idealized warm bubble (rigid lid) still PASSES 6/6 after this change,
confirming the gate is byte-unaffected.
