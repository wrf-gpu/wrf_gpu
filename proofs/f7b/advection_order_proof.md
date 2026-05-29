# F7-B Advection Order Proof — WRF flux-form WS5/3 verification

## Scope (documented restriction, not a WRF fact)

WRF selects advection order from `config_flags%h_sca_adv_order` /
`config_flags%v_sca_adv_order` (`module_advect_em.F:3130-3131`). This sprint
freezes the implementation to **horizontal order 5 / vertical order 3** with
periodic-x/-y boundaries (no near-boundary degradation), the configuration the
F7-B idealized gates use. Other orders and specified/nested boundary degradation
(`module_advect_em.F:3137-3392`) are out of scope.

## Operator definitions (exact WRF, `module_advect_em.F:3105-3119`)

```
flux6(q_im3..q_ip2)   = (37*(q_i+q_im1) - 8*(q_ip1+q_im2) + (q_ip2+q_im3))/60
flux5 = flux6 - sign(time_step)*sign(vel)*((q_ip2-q_im3) - 5*(q_ip1-q_im2) + 10*(q_i-q_im1))/60
flux4(q_im2..q_ip1)   = (7*(q_i+q_im1) - (q_ip1+q_im2))/12
flux3 = flux4 + sign(time_step)*sign(vel)*((q_ip1-q_im2) - 3*(q_i-q_im1))/12
```

`time_step > 0` so `sign(time_step) = +1`. Implemented in
`src/gpuwrf/dynamics/flux_advection.py` (`flux5_face_periodic`, `advect_scalar_flux`).
Tendencies are flux divergences: `tend -= mrdx*(fqx(i+1)-fqx(i))` with
`mrdx = msftx*rdx` (`module_advect_em.F:3387-3388`); mass coupling via `ru`/`rv`/
`rom` from `couple_momentum` (`module_em.F:195`) + `calc_ww_cp`
(`module_big_step_utilities_em.F:640-782`).

## 1-D linear-advection convergence check (analytic oracle)

Smooth periodic field φ = sin(x) on [0, 2π], constant velocity u = 1 > 0. The
WRF `flux5` flux divergence `-∂(uφ)/∂x` is compared to the analytic `-u·cos(x)`.
L2 error vs grid refinement (`taskset -c 0-3`, fp64, `cuda:0`):

| nx  | L2 error   | observed order |
| --- | ---------- | -------------- |
| 32  | 3.418e-06  | —              |
| 64  | 1.073e-07  | 4.99           |
| 128 | 3.358e-09  | 5.00           |
| 256 | 1.050e-10  | 5.00           |

The operator converges at the design 5th order, confirming the `flux5`
implementation (the upwind-corrected 6th-order centered flux) is WRF-faithful.

## Shape check (upwind dissipation)

`flux5 = flux6 - sign(vel)*correction` adds the odd-order dissipative correction
that makes the 6th-order centered flux a 5th-order upwind scheme: for `vel > 0`
the correction subtracts, biasing the stencil upstream; for `vel < 0` it adds,
biasing downstream. This is the WRF monotone-leaning upwind bias and is what the
convergence table exercises (sign(vel) constant +1 there).

## Status

The flux-form scalar advection operator is verified standalone (5th-order
convergence). Its integration into the operational/idealized path is wired
(`use_flux_advection`); the end-to-end idealized-case gates (AC1/AC2) are tracked
in the worker report.
