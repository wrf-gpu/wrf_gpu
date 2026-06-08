# F7N — Straka touchdown close: vertical-momentum-advection sign fix

## Status: F7N_COMPLETE — the dry dynamical core is DONE

Straka density current **PASSES 6/6** at 900 s and the Skamarock warm bubble
**STILL PASSES 6/6** with no regression. The decisive per-acoustic-substep WRF
ground-truth diff at the cold-pool touchdown column localized the residual to a
single sign error in the vertical momentum advection; the WRF-faithful fix closes
it.

## How the residual was localized (per-acoustic-substep WRF ground truth)

Instrumented pristine WRF v4.7.1 `dyn_em/solve_em.F` (the `WRFGPU2_TOUCHDOWN`
block, after the in-loop `calc_p_rho`) to dump, **per acoustic substep**, the
touchdown column `i=256` (= `nxc` = domain center) and its x-neighbours `i=255,257`,
`j=2`, for `itimestep 170..205` (t≈170–205 s, the touchdown window): `w_2, ph_2,
p, rw_tend, ph_tend, ww (omega), u_2` (faces i and i+1), `v_2, t_2save, muave,
muts, mut`. Built the matching JAX per-substep dump
(`scripts/f7n_jax_touchdown_dump.py`, the operational RK3/acoustic cadence with a
Python substep loop) at the JAX center mass index (249, x≈0). Diffed them
(`scripts/f7n_touchdown_diff.py` → `proofs/f7n/touchdown_substep_diff.json`).

**Decisive find.** Through t≈180 s the bulk dynamics agree (as F7M reported), but
the JAX center-column **`ww` (omega, the vertical mass flux) develops a 2Δz
vertical sawtooth at faces ~22–37 (z≈2200–3700 m)** while WRF's `ww` stays smooth
and bounded. Tracing it back, the JAX **physical `u`** carries a growing 2Δz
vertical mode in the cold-pool descent layer (z≈2000–4000 m): the 2nd-difference
`|u(k+1)-2u(k)+u(k-1)|` grows 0.04 (t=120) → 0.28 (t=150) → 4.2 (t=170) → 26.5
(t=180) → 137 (t=190) → NaN, while `max|u|` runs away 17 → 47 → 135 → 523 m/s and
`max|w|` 21 → 44 → 407 m/s, all with the **front frozen at 2350–2450 m** (the
cold pool never spreads). This is exactly the F7M signature ("descending w not
converted to horizontal u-outflow; front crawls while |u| runs away"), now
resolved to a **growing 2Δz vertical mode in `u`**, not a coupling deficit.

**Bisection (empirical).** Disabling the vertical momentum advection
(`_vertical_flux_div_3`) removed the mode entirely (u 2Δz 26.6 → 0.7 at t=180,
finite past 240 s); a 20× stronger vertical diffusion also suppressed it. So the
vertical momentum advection was *generating* the mode (anti-dissipative), not
merely failing to damp it.

## Root cause + WRF-faithful fix

`src/gpuwrf/dynamics/flux_advection.py::_vertical_flux_div_3` (the `advect_u` /
`advect_v` vertical 3rd-order flux) applied the upwind correction with the
**wrong sign**.

WRF `advect_u` (`dyn_em/module_advect_em.F:1474-1480`) computes the vertical flux
at face *k* as `vflux = vel*flux3(u(k-2..k+1), -vel)` with
`vel = 0.5*(rom(i-1,k)+rom(i,k))`. WRF's `flux3` (`:202-204`) is
`flux4 + sign(time_step)*sign(ua)*corr`, and the argument passed is `ua = -vel`,
so the assembled flux is

    vflux = vel*flux4 + sign(-vel)*vel*corr = vel*flux4 - |vel|*corr   (DISSIPATIVE)

The JAX code used `velz = +romq` and `flux3 = flux4 + sign(velz)*corr`, then
`vflux = romq*flux3`, i.e.

    vflux = romq*flux4 + |romq|*corr   (ANTI-dissipative)

— the **opposite sign** on the upwind correction, turning the dissipative
3rd-order upwind scheme into an anti-dissipative (mode-amplifying) one. The
2Δz-in-z mode the corrector is meant to damp was instead pumped, and the
cold-pool descent column (large `|rom|`, sharp vertical `u` shear) excited it into
a runaway.

The scalar path (`advect_scalar_flux`) and the `w` path (`_vertical_flux_div_w`)
already negate the velocity correctly; only the `u`/`v` vertical flux had the flip.

**Fix** (`_vertical_flux_div_3`): use the WRF sign,
`flux3 = flux4 + sign(-rom_k)*corr`, `vflux = rom_k*flux3` →
`rom_k*flux4 - |rom_k|*corr`. WRF source cited inline.

### Secondary (mass conservation) fix

Once the touchdown instability was removed and Straka ran to 900 s, the
dry-column mass drifted 3.4e-8 (vs the ≤1e-8 gate) — traced (diffusion-off A/B:
drift 0 without ν) to the **non-conservative `mass*K*∇²field`** const-K diffusion
(F7L). Replaced with the **WRF-faithful flux-divergence form**
`d/dx_j( mass*K*d field/dx_j )` (`conservative_constant_k_diffusion_tendency`;
WRF `horizontal_diffusion_s`/`vertical_diffusion`,
`module_diffusion_em.F:2999-3018`), which conserves the mass-weighted integral to
round-off. Drift → **2.25e-9** (PASS), and θ′min improved to the canonical
−9.97 K. The bubble is inviscid (ν=0) so this is a no-op there.

## Gates

- **AC1 — Straka PASS (6/6):** finite to 900 s; front **14.15 km** (gate 15±2);
  θ′min **−9.97 K** (gate −9..−10); max|w| **14.57** (gate 12–18); **4 rotors**
  (gate 2–4); mass drift **2.25e-9** (gate ≤1e-8). All PASS.
- **AC2 — warm bubble STILL PASS 6/6:** thermal_rise 1924.3 m, max|w| 11.68,
  θ′max 1.92, h-drift 1.8e-12, mass drift 0. Identical to F7K/F7L/F7M (no
  regression).
- **AC3 — touchdown per-substep parity DELIVERED:** `touchdown_substep_diff.json`.
  Post-fix the center-column `ww` low-level stays bounded (≈0, matching WRF) where
  pre-fix it ran to −68/−165; coupled `w_min` bounded (~−3000) vs the pre-fix
  −2.4e6 runaway; `max|u|` 16→20 m/s through t=205 vs pre-fix 16→523.
- **AC4 — no regression:** m4 10/10; flat-rest machine-zero, conservation, all
  prior F7 operators intact; no masking clamps; only WRF-faithful sign + flux-form
  fixes.

## Files changed

- **M** `src/gpuwrf/dynamics/flux_advection.py::_vertical_flux_div_3` — WRF sign
  on the 3rd-order vertical momentum-advection upwind correction (the close).
- **M** `src/gpuwrf/dynamics/explicit_diffusion.py` — add
  `conservative_constant_k_diffusion_tendency` (flux-divergence, mass-conserving).
- **M** `src/gpuwrf/runtime/operational_mode.py::_augment_large_step_tendencies`
  — wire the conservative diffusion for u/v/w/θ.
- **NEW** WRF instrumentation `~/src/wrf_pristine/WRF/dyn_em/solve_em.F`
  (`WRFGPU2_TOUCHDOWN` block) + `recompile_grav2d_incremental.sh`.
- **NEW** `scripts/f7n_jax_touchdown_dump.py`, `scripts/f7n_touchdown_diff.py`,
  `scripts/f7n_official_run.py`.
- **NEW** ground truth `/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt`
  (per-substep WRF) and `..._touchdown_substeps.json` (JAX).
