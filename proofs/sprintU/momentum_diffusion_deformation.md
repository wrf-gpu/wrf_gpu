# Sprint U / P0-2 — WRF deformation-tensor momentum diffusion wired + validated

Date: 2026-05-29
Branch: `worker/opus/f7d-pressure-mass-fix`

## Finding being closed (GPT pre-close P0-2)

> WRF `diff_opt=2/km_opt=1` does NOT diffuse momentum by applying the scalar
> Laplacian independently to u, v, w.  WRF calls `horizontal_diffusion_u/v/w_2`
> and `vertical_diffusion_u/v/w_2` using deformation/stress terms (defor13,
> defor23, defor33).  The JAX runtime added the scalar flux-divergence to u, v,
> w; `constant_k_deformation_momentum_tendency` existed but was UNWIRED.

## What WRF actually does (source)

`module_diffusion_em.F`:

* `cal_deform_and_div` builds the deformation tensor:
  - `defor11 = 2 du/dx`        (`:215`)
  - `defor33 = 2 dw/dz`        (`:373`)
  - `defor13 = dw/dx + du/dz`  (`:820` first term, `:885/:902` second term)
* `cal_titau_11_22_33` / `cal_titau_13_31`: `titau_ij = -rho * xkm * defor_ij`
  (`:5428, :5441`).
* `horizontal_diffusion_u_2` (`:3308`) flat-slab: `tend += g*dz/dnw * rdx *
  d/dx(titau11)`; `vertical_diffusion_u_2` (`:4552`): `+ g/dnw * d/dz(titau13)`.
* `horizontal_diffusion_w_2` flat-slab: `g*dz/dn * rdx * d/dx(titau13)`;
  `vertical_diffusion_w_2` (`:4779`): `+ g * d/dz(titau33) / dn`.

So the WRF momentum operator has (a) a **factor 2** on the diagonal (D11/D33)
and (b) **off-diagonal cross terms** — the `dw/dx` term enters the u tendency
and the `du/dz` term enters the w tendency. The scalar Laplacian / scalar
flux-divergence has neither.

## Implementation

`src/gpuwrf/dynamics/explicit_diffusion.py::wrf_deformation_momentum_tendency`
implements the flat-slab reduction. For the uniform-z hydrostatic slab the WRF
mass-coordinate weight `g*dz/dnw` times the `rho` in `titau` reduces (via
`|dnw| = rho*g*dz/mu`) to the dry-mass face weight, so the density-weighted
stress divergence

```
d(u)/dt = K*( 2 u_xx + u_zz + w_xz )        (D11 diag + D13 cross)
d(w)/dt = K*( w_xx + 2 w_zz + u_xz )        (D33 diag + D13 cross)
```

is computed UNCOUPLED and then multiplied by the field face mass
(`mass_u`/`mass_f`) in `operational_mode._augment_large_step_tendencies`, exactly
as the scalar diffusion enters the coupled tendency space. Theta keeps the
conservative scalar flux-divergence (WRF `horizontal_diffusion_s`); v on the
one-row slab is degenerate (D22/D12 = 0) so it uses the scalar form (identical to
the deformation v-diffusion for ny=1).

Wiring: gated by `OperationalNamelist.use_deformation_momentum_diffusion`, active
only when `const_nu_m2_s > 0`. Default OFF (the F7N close default — conservative
scalar flux-divergence — is preserved bit-for-bit).

## Validation (analytic oracle)

`tests/dynamics/test_deformation_momentum_diffusion.py` (4 tests, all PASS):

* `test_deformation_matches_fd_oracle_interior` — `du` matches an INDEPENDENT
  closed-form finite-difference oracle of `K(2u_xx+u_zz+w_xz)` to **round-off
  (1e-9)**; `dw` matches `K(w_xx+2w_zz+u_xz)` to **~1%** (the WRF flux-divergence
  assembles `titau13` at w-faces then differences in x — a different but equally
  2nd-order stencil for the `w_xx` cross piece).
* `test_deformation_dw_cross_term_converges_second_order` — that ~1% stencil
  difference **converges at 2nd order** under grid refinement (halving dx drops
  the error >3x), confirming both are consistent discretizations.
* `test_deformation_is_down_gradient` — a single-mode bump has negative
  `<du/dt, u>` (dissipative).
* `test_deformation_zero_on_uniform_flow` — uniform u / zero w → zero tendency.

## Idealized gate re-run with the deformation operator

The deformation operator is wired and analytically validated. The default close
path keeps the conservative scalar flux-divergence (the operator validated in the
F7N close that yields the 6/6 Straka PASS). See `straka_deformation_gate.md` for
the A/B of the Straka gate with `use_deformation_momentum_diffusion=True`.

## Honest scope

This is the **flat-slab** reduction (zx=zy=0, msf=1, ny=1): the terrain-slope
(`zx`/`zy`) cross-coordinate stress terms and the multi-row D22/D12/D23 paths are
Phase-B (3D-terrain) gates. For the dry idealized/real-flat operational path the
operator is the WRF momentum operator; for full 3D terrain the slope terms remain
to be wired.
