# AC4 — Nonzero analytic acoustic oracle (hydrostatic adjustment)

This is the primary physics proof for F7.A: a nonzero test whose sign and order
of magnitude are known analytically, exercising the implicit vertical solve
(`advance_w`), the buoyancy coupling via `cqw`/`c2a`, and the pressure refresh
(`calc_p_rho`). Flat-rest (AC3) only proves the trivial fixed point; this proves
the operator responds correctly to a real forcing.

Harness: `scripts/f7a_oracles.py::run_analytic_acoustic`. It drives the
production `acoustic_substep_core` directly (no operational replay, no
JAX-vs-JAX self-compare) on a constructed hydrostatically-balanced, dry,
constant-θ, flat (pure-sigma `hybrid_opt=0`) column.

## Setup

* Column: `nz=20` mass levels, `θ = T0 = 300 K`, dry, flat terrain, no map-factor
  distortion. Surface pressure 1000 hPa, model top 100 hPa.
* Base state is exactly hydrostatic, so every perturbation work array is zero and
  `advance_w` produces no tendency (verified separately by AC3 = machine zero).
* Perturbation: a single mid-column mass level (`k = nz/2`) is warmed by
  `θ' = +1 K`. Nothing else is perturbed.
* One acoustic substep at `dts = 2 s`, `epssm = 0.5`.

## Analytic expectation (sign + magnitude)

A warm potential-temperature perturbation reduces the local density and produces
an upward buoyancy force. In the WRF small-step `w` equation
(`module_small_step_em.F:1477-1489`) the perturbation θ enters the implicit-`w`
RHS through the buoyancy term `c2a·alt·t_2ave`, with magnitude (per unit mass)

    b = g · θ' / θ0  =  9.81 · 1 / 300  ≈  0.0327 m s⁻².

Over a single substep the leading-order physical vertical-velocity scale is

    dw ≈ b · dts  ≈  0.0327 · 2  ≈  0.065 m s⁻¹.

Known qualitative signature of a warm bubble in a stratified column:

1. The face **above** the warm layer is accelerated **upward** (`w > 0`).
2. The face **below** the warm layer is accelerated **downward** (`w < 0`)
   (the layer expands about its centre — a buoyant dipole in `w`).
3. The column geopotential **rises** above the warm layer (`ph' > 0` aloft).
4. The decoupled `|w|` is within ~2 orders of magnitude of `b·dts` after one
   substep (the implicit column solve couples the whole column, so an O(1–10×)
   amplification of the single-level estimate is expected and bounded).

## Measured result (committed code)

From `analytic_acoustic_oracle.json`:

* buoyancy accel `b` = 0.0327 m s⁻²; expected `dw` order = 0.065 m s⁻¹.
* `w` face **above** warm level = **+0.323 m s⁻¹** (upward — correct sign).
* `w` face **below** warm level = **−0.307 m s⁻¹** (downward — correct sign).
* geopotential above the warm layer (`ph_top`) = **> 0** (column rises — correct sign).
* decoupled `|w|_max` = 0.323 m s⁻¹, i.e. ≈ 5× the single-level `b·dts`
  estimate — same order of magnitude, well inside the 2-order band.

All four expectations are met: `sign_dipole_ok = True`, `ph_rise_ok = True`,
`magnitude_ok = True` → **AC4 PASS**.

Note on coupling: `advance_w` returns the *coupled* work `w` (≈ `(c1f·mut+c2f)·w/msfty`),
so the raw returned array is O(mut·w) ≈ 3×10⁴. The harness decouples by
`(c1f·mut+c2f)` before the physical sign/magnitude checks; the values above are
the decoupled physical `w` in m s⁻¹.
