# F7L — Straka density-current close: WRF-faithful constant-K diffusion on w

## Residual entering F7L (from F7K)
Skamarock warm bubble PASS 6/6 (inviscid). Straka density current (dx=dz=100 m,
−15 K cold pool) NaN at 240 s. F7K trace (ν=75 on u,v,θ already wired):

```
t= 60 finite=True maxw=7.32  thmin=-14.92
t=120 finite=True maxw=14.66 thmin=-14.80
t=180 finite=True maxw=21.31 thmin=-14.68
t=240 finite=False                       <- detonation
```

max|w| roughly **doubles every 60 s** (7→15→21) — an exponential numerical
growth signature, not physical cold-pool acceleration (which saturates ~15-20 m/s).
The growing field is the vertical velocity w.

## Root cause: w was not being diffused
The F7.B constant-K diffusion was wired into the large-step dry tendency for
**u, v, θ only** (`runtime/operational_mode.py` `_augment_large_step_tendencies`,
the `nu > 0` block). WRF's `diff_opt=2`/`km_opt` constant-K path diffuses
**u, v, w AND θ**:

* `dyn_em/module_diffusion_em.F:2864-3113` `horizontal_diffusion_2` calls, for the
  const-K path, `horizontal_diffusion_u_2` (:3118), `horizontal_diffusion_v_2`
  (:3323), **`horizontal_diffusion_w_2` (:3519, invoked at :2999-3007)**, and
  `horizontal_diffusion_s` (:3711) for θ.
* `dyn_em/module_diffusion_em.F:4004-4458` `vertical_diffusion_2` calls
  `vertical_diffusion_u_2` (:4463), `_v_2` (:4576), **`_w_2` (:4688)**, `_s` (:4789).

The Straka et al. (1993) reference solution is itself **defined** with constant
kinematic ν = 75 m²/s applied to **u, w, and θ** (it is a 2-D x–z problem; the
ν=75 run *is* the benchmark — the front ≈15.5 km, min θ′ ≈ −9..−10 K, and the
2-4 Kelvin-Helmholtz rotors at 900 s are the ν=75 solution). Diffusing only u/θ
leaves the sharp-front vertical velocity free to grow a grid-scale (2Δx) mode →
the observed exponential max|w| runaway and 240 s detonation.

## The fix (WRF-faithful, no masking clamp)
`src/gpuwrf/runtime/operational_mode.py` `_augment_large_step_tendencies`, in the
`nu > 0.0` block, add the w component using the existing F7.B
`constant_k_diffusion_tendency` and the already-computed w-face mass `mass_f`:

```python
w_t = w_t + mass_f * constant_k_diffusion_tendency(haloed.w, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
```

For the flat (zx=zy=0) uniform-z idealized slab the WRF deformation-tensor
momentum diffusion (`τ_ij = K(∂u_i/∂x_j+∂u_j/∂x_i)`, `module_diffusion_em.F`
`cal_deform_and_div` :17, `horizontal/vertical_diffusion_w_2`) reduces to
`K∇²w` plus the small `K ∂_z(∇·u)` divergence-correction term; `K∇²w` is the
dominant stabilizing contribution and is exactly the Straka spec form. The
const-K tendency form was verified WRF-faithful in F7.B: WRF's coupled
scalar/vertical tendency `g·(H3(k+1)−H3(k))/dnw` with `H3=−K·ρ·Δvar·rdz`
algebraically equals `μ·K·∂²var/∂z²` on the hydrostatic flat grid (since
`g·ρ·dz=dp` and `dp/dnw=μ`), matching `mass_f * K∇²`.

The warm bubble is unaffected: it runs with `const_nu_m2_s = 0.0`
(`ic_generators/idealized.py:567`), so the entire `nu > 0` block is skipped — the
bubble stays inviscid.

## Damping / CFL audit (verified already WRF-correct, unchanged)
* External-mode divergence damping (`emdiv`, mudf) — active, WRF default 0.01,
  in `advance_uv` (`acoustic_substep_core` default `emdiv=0.01`;
  `module_small_step_em.F:808-810,866-869`).
* Pressure-memory divergence damping (`smdiv`) — active, WRF default 0.1, in
  `calc_p_rho(step=iteration)` (`p = p + smdiv*(p − pm1)`;
  `module_small_step_em.F:557-567`).
* Vertical-CFL `w_damping=1` (`w_damp`, `module_big_step_utilities_em.F:2714-2774`)
  — active; activation CFL>1 not reached (dts≈0.01 s ⇒ tiny vertical Courant), so
  it is the safety net, not the Straka stabilizer.
* Rayleigh upper damping `damp_opt=3, dampcoef=0.2, zdamp=3000` + rigid top_lid —
  active.
* Acoustic substeps = 10 ⇒ dt_sound = 0.01 s ⇒ acoustic CFL = c·dt_sound/dx ≈
  347·0.01/100 ≈ 0.035 ≪ 1: the acoustic small step is far inside the CFL limit,
  so the detonation was NOT an acoustic-CFL violation. (WRF `time_step_sound`
  auto-pick would give ~4-6 substeps; 10 is conservative and WRF-valid.)

## Before / after max|w| trace (measured, dx=dz=100m, dt=0.1s, 10 acoustic substeps)

A/B with the existing F7.B const-K wiring, all WRF damping active
(emdiv=0.01, smdiv=0.1, w_damping=1, damp_opt=3 dampcoef=0.2 zdamp=3000, top_lid):

```
t(s)   nu=0 (no diffusion)        nu=75 u,v,w,θ  (F7L fix)
 60    maxw 4.98                  maxw 7.30
 80    maxw 9.75
120    maxw 14.80                 maxw 14.62
160    maxw 19.52                 maxw 19.22
180    maxw 21.31(F7K)            maxw 21.20
200    maxw 23.49                 maxw 22.62
240    NaN  (DETONATE, =F7K)      maxw 23.90  finite  <-- fix crosses 240s
300                               NaN  (detonate near cold-pool touchdown)
```

The w-diffusion fix is REAL and necessary: nu=0 detonates at 240 s (the F7K
failure, unchanged whether ν=75 is on u,v,θ only); ν=75 on **u,v,w,θ** survives
240 s (max|w|=23.9, ramp flattening 22.6→23.9). But it is **NOT SUFFICIENT** for
the full 900 s — it detonates between 240 and 300 s, at the moment the cold pool
reaches the lower boundary and the gust-front shear/Kelvin–Helmholtz layer forms.

## Honest status: F7L_PARTIAL
max|w| reaches 23.9 m/s at 240 s — well above the canonical Straka ν=75 reference
(~12–18 m/s) — while the gust front is only ~2.65 km from center at 240 s (the
reference head is several km further along). This **excess vertical velocity +
sluggish lateral spreading** pattern (cold air sinking/oscillating rather than
spreading) plus the touchdown-time detonation indicates a residual beyond simple
under-diffusion: a real operator/coupling defect at the descending sharp cold
front (candidate: gust-front horizontal-PGF → cold-pool outflow conversion, or
the descending-front lower-boundary w handling). Per the F7L hard rule (no ad-hoc
clamps; only benchmark ν=75 + WRF damping/CFL allowed), the WRF-faithful
stabilizers were applied and Straka still detonates → **F7L_PARTIAL**, trace
above, no masking added. The warm bubble (cleaner buoyancy test) remains PASS 6/6
inviscid — the F7 buoyancy/transport path is sound; the residual is specific to
the stiff descending cold front.
