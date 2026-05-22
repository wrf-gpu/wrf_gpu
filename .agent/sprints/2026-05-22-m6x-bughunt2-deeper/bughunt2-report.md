# M6.x Bug-Hunt #2 (Deeper) — What Bug-Hunt #1 Missed

**Date**: 2026-05-22
**Reviewer**: Claude Opus 4.7 xhigh (bughunt2)
**Anchor**: `/tmp/wrf_gpu2_m6x/` after worker A2 applied bug-hunt #1's 3 fixes (PH g, removed `PRESSURE_IMPLICIT_RELAXATION` + asymmetric mask, alpha cap 0.02→5.0, mu inside acoustic scan)
**Reference artifacts**:
- `m6x_failed_6h_direct_probe.json` — pre-fix baseline (76% sanitize fire, `nonfinite_count=0`, deterministic drift to clip bounds)
- `m6x_a2_fix2_6h_direct_probe.json` / `fix2b` / `fix3` — post-fix probes (99.95% fire, `nonfinite_count = 3.89 BILLION`, sanitize value-rate 43%)

---

## 1. Re-analysis of failure signature: why did bug-hunt #1's fixes make it WORSE?

Bug-hunt #1's three fixes were each individually defensible:
- PH update WAS missing `g` (real dimensional bug); after fix, units balance.
- `PRESSURE_IMPLICIT_RELAXATION = 0.05` had no WRF citation; after fix, pressure absorbs divergence at physical rate.
- `MAX_INVERSE_DENSITY = 0.02` was 42× too tight for the troposphere (α = R T/p ≈ 0.83 at surface); after fix, horizontal pressure gradient force is at physical magnitude.

But the post-fix probe shows the failure *changed character*: from "deterministic drift into clip bounds with `nonfinite_count = 0`" (76% fire) to "broad nonfinite explosion with 3.89 BILLION NaN/Inf values" (99.95% fire, 43% of state values clipped per step). That is **strictly worse**.

The interpretation that fits is: the original 0.02 alpha cap and the 0.05 pressure relaxation were **neutering the horizontal pressure gradient force and the acoustic pressure response by factor 42× and 20× respectively**. With both factors near-zero, the dycore's only dynamics was Eulerian advection + boundary forcing + sanitize. The system was effectively frozen-acoustic, deterministically drifting due to advection-only forcing + boundary mismatch + sanitize asymmetry. That looked like a contained 76% fire rate.

Bug-hunt #1's fixes **unfroze the acoustic mode** without fixing the deeper bugs that the freezing was hiding. Once acoustic propagation is at physical strength, three structural problems become dominant. Bug-hunt #1 examined `acoustic_once` lines 185–198 but missed the FORMULATION level — the question of whether the equations themselves match WRF's dycore at all.

---

## 2. Operators bug-hunt #1 skipped — audit findings

Spot-checks I performed (READ-ONLY): `_grad_x_to_u`, `_grad_y_to_v`, `_grad_z_to_w`, `_mass_to_u/v/w_face`, `_layer_thickness_m`, `_inverse_density`, `_horizontal_divergence_cgrid`, `_vertical_divergence_cgrid`, `compute_mu_tendency`, `apply_lateral_boundaries`, the physics-coupler `_mass_to_u_face`, `compute_advection_tendencies`, RK3 mu plumbing.

**Operators that are dimensionally and structurally correct**:
- `_grad_x_to_u`, `_grad_y_to_v`, `_grad_z_to_w` (acoustic.py:109-141) — stencils and zero-padding match a non-periodic specified-boundary C-grid. Sign convention checks against ∂p/∂x positive → ∂u/∂t negative. `_grad_z_to_w` correctly uses spacing at the w-face (the inter-mass distance).
- `_layer_thickness_m` (acoustic.py:125-126) — `(ph[1:] - ph[:-1])/g`; positive by `jnp.maximum(..., 1.0)` floor.
- `_horizontal_divergence_cgrid` (acoustic.py:98-101) and `_vertical_divergence_cgrid` (acoustic.py:104-106) — divergence at mass points from face-normal differences. Shapes work out.
- `compute_mu_tendency` (tendencies.py:43-54) — column-integrated mass-coupled horizontal flux divergence. Sign, weights, face interpolation correct.
- `apply_lateral_boundaries` and `_relaxed_slice` (boundary_apply.py) — specified-zone at offset 0 + relaxation at offsets 1..3 with WRF-style fcx/gcx weighting; staggered shapes handled correctly (u-faces 0 and nx, v-faces 0 and ny). I see no sign or stencil error.

**Operators that are technically wrong but probably not the dominant bug**:
- `coupling/physics_couplers.py:98-109` — `_mass_to_u_face` and `_mass_to_v_face` use `jnp.roll` (PERIODIC interpolation) before producing the staggered face array. For d02 (limited-area), this corrupts u-face 0, u-face nx, v-face 0, v-face ny with a periodic-wrap mix of opposite boundary values. **However**, `apply_lateral_boundaries` is called immediately after physics in the timestep order (`driver.py:777-797`) and overwrites exactly those boundary faces with the specified Gen2 forcing. The internal cells are correctly interpolated. So this leaks at most one cell per step pre-overwrite. Not the failure mode, but worth fixing for correctness.
- `advection.py:41-45` `_dz_from_state` returns the *mean* dz across all layers as a single scalar passed to vertical 3rd-order upwind. WRF eta levels have non-uniform dz (~30 m near surface vs ~1000 m aloft). Using the mean (~300 m) makes near-surface vertical gradients ~10× too weak and aloft gradients ~3× too strong. This is a systematic bias in scalar advection — contributes to slow theta/qv drift but is not the explosion source.

**Operators that are the real problem** — see §3.

---

## 3. Top 3 bug hypotheses DIFFERENT from bug-hunt #1

### Hypothesis A — Missing buoyancy term in the vertical momentum (w) equation (acoustic.py:189)

**File:line**: `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/acoustic.py:189-191`

```python
w_explicit = state.w - dt_sub * _mass_to_w_face(alpha) * _grad_z_to_w(p_dynamic, state)
w_next = state.w + _vertical_implicit_weight(state, dt_sub) * (w_explicit - state.w)
ph_next = state.ph + dt_sub * GRAVITY_M_S2 * w_next
```

The w equation contains only the **perturbation pressure gradient** `-α ∂p'/∂z`. There is **no buoyancy term**. In a compressible non-hydrostatic atmosphere, the vertical momentum equation in perturbation form is

```
∂w/∂t = -α ∂p'/∂z  +  g (ρ_base - ρ)/ρ   +  ...   (buoyancy = the second term)
```

Equivalently, in WRF's mass-coordinate split-explicit dycore the buoyancy enters via the (w, ph) tridiagonal coupling that ties w to perturbation geopotential AND to perturbation theta (t_2ave). Specifically, `module_small_step_em.F:1481-1486`:

```fortran
w(i,k,j) = w(i,k,j) + ... [pressure-gradient part using c2a, rhs, ph] ...
         + dts*g*msft_inv*(rdn(k)*(c2a(i,k,j)*alt(i,k,j)*t_2ave(i,k,j)
                                  - c2a(i,k-1,j)*alt(i,k-1,j)*t_2ave(i,k-1,j))
                          - (c1f(k)*muave(i,j)))
```

The second chunk — products of `t_2ave` (perturbation theta) and `c2a*alt` (≈ γp×α = γ R T) — is the discrete buoyancy. The worker's `acoustic_once` has *no analogue* of this term. Theta never enters the acoustic substep at all.

**Why this matches the failure signature**:
- Before the alpha-cap relaxation, `α ≤ 0.02` neutered the pressure-gradient term too; w stayed near boundary forcing values. Missing buoyancy didn't matter because no vertical mode was excited.
- After the cap relaxation to 5.0, the perturbation-pressure-gradient drive is at physical magnitude (250× stronger), and there's no buoyancy to restore the gravity-wave mode. Any small vertical perturbation grows monotonically (no stiffness against vertical expansion/compression), driving `ph` to drift, which drives `_layer_thickness_m` to drift, which corrupts `_vertical_divergence_cgrid` and `_grad_z_to_w` for every subsequent substep.
- The bipolar mu pattern (cells running to both 1000 Pa and 120000 Pa clip bounds) is consistent with unbounded gravity-wave growth: some columns expanding (ph rising → mu falling), others compressing (mu rising), depending on local divergence sign.

**WRF citation**: `module_small_step_em.F:1481-1486` (advance_w buoyancy term), plus Skamarock 2008 ARW Tech Note Eqs. 2.20-2.21 (the buoyancy coefficient in the linearized vertical momentum equation).

**Minimal discriminator test** (idealized warm-bubble Skamarock-Klemp 1994):

```python
def test_warm_bubble_rises_then_oscillates():
    """A warm thermal must rise (buoyancy) then oscillate (gravity wave restore).
    Without the buoyancy term the bubble just sits or drifts unbounded.
    """
    grid = make_ideal_grid(nz=40, ny=8, nx=80, dx_m=100.0, dy_m=100.0)
    state = isothermal_atmosphere(grid, T=300.0)
    state = add_warm_perturbation(state, dT=2.0, x_center=4000.0, z_center=2000.0, radius=2000.0)
    # 600s of integration with dt=1.0s; ascending phase ~250s
    out = run_dycore(state, dt=1.0, steps=600)
    w_max = float(jnp.max(out.w))
    z_of_bubble = bubble_centroid_height(out)
    assert w_max > 5.0          # buoyancy lifted the bubble
    assert z_of_bubble > 2300.0 # moved up at least 300 m
    # Without buoyancy: w_max ~ 0 from acoustic noise only; bubble stays at z=2000.
```

**Likelihood**: HIGH. This is a *formulation* gap, not a typo. Adding a `g*(ρ_base - ρ)/ρ` term costs ~5 lines but requires `state.theta_base` (or perturbation theta) which the State pytree may not currently expose. A real fix is roughly the depth of one acoustic-pass rewrite.

---

### Hypothesis B — Pressure is integrated prognostically and drifts off hydrostatic balance with (theta, ph) (acoustic.py:185)

**File:line**: `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/acoustic.py:185`

```python
p_next = state.p - c2 * dt_sub * div
```

The worker treats `p` as an independent prognostic with `dp/dt = -c² ∇·v`. In WRF's mass-coordinate split-explicit dycore, **pressure is not prognostic** — it is **diagnosed** from `theta`, `ph`, `mu` every small step via the linearized equation of state and hydrostatic relation. `module_small_step_em.F:494-528`:

```fortran
! linearized equation of state — p is computed from t_2 (theta) and al (perturbation alpha)
p(i,k,j) = c2a(i,k,j)*(alt(i,k,j)*(t_2(i,k,j)-(c1h(k)*mu(i,j))*t_1(i,k,j))
                      /((c1h(k)*Mut(i,j)+c2h(k))*(t0+t_1(i,k,j)))-al(i,k,j))

! and al is computed from mu and ph (hydrostatic):
al(i,k,j) = -1./(c1h(k)*Mut(i,j)+c2h(k))*(alt(i,k,j)*(c1h(k)*mu(i,j))
                                          + rdnw(k)*(ph(i,k+1,j)-ph(i,k,j)))
```

In the worker's formulation, three things evolve **independently**: theta (via dycore advection + Thompson/MYNN/RRTMG physics in `_candidate_before_boundary` driver.py:800-816), `ph` (via `ph += dt*g*w` in acoustic.py:191), and `p` (via the prognostic equation above). After even one step they are no longer hydrostatically self-consistent, and the inconsistency compounds. By 6h the (theta, ph, p) triplet is unrelated to a physical atmospheric state — and `mu`, which in WRF is `p_sfc - p_top`, has no clean relationship to anything.

A linked symptom: the `_pressure_perturbation(p_next, state.pb)` extraction (acoustic.py:186) gives a "perturbation pressure" that no longer corresponds to (theta - theta_base) × hydrostatic — it is the integrated divergence error. The gradient of THAT then drives momentum at acoustic.py:187-189, injecting unphysical accelerations.

**Why this matches the failure signature**: it is the *first-principles* reason why the worker's dycore can never be stable to 6h regardless of damping tuning. The 76% sanitize fire rate at the original constants was the symptom of slow `p`-vs-(theta,ph) drift. Removing the damping (bug-hunt #1's fix #2) merely accelerated the drift into immediate NaN territory.

**WRF citation**: `module_small_step_em.F:527-528` for the diagnostic perturbation pressure; `module_small_step_em.F:506-509` for the hydrostatic al computation. The companion document is Skamarock 2008 ARW Tech Note §3 (eqs. 3.7-3.13) — the "small step equations" are written as prognostic in (u, v, w, μ, ph) but **diagnostic in p**.

**Minimal discriminator test**:

```python
def test_hydrostatic_consistency_preserved_after_n_acoustic_substeps():
    """After acoustic loop, p_state should equal p_diagnosed_from_theta_ph_mu within roundoff."""
    state = realistic_init_from_gen2()
    state_after = forward_backward_acoustic(state, grid, dt=10.0, n_acoustic=4)
    p_diag = diagnose_pressure_from(state_after.theta, state_after.ph, state_after.mu, state_after.pb)
    rel_err = jnp.max(jnp.abs(state_after.p - p_diag) / jnp.maximum(state_after.p, 1.0))
    assert rel_err < 1e-3   # current code: drift will be visible after 1 dt and explode by 6h
```

**Likelihood**: HIGH (this is the structural issue c1 is rewriting). Bug-hunt #1's hypothesis #3 ("mu temporal decoupling") was a symptom of this — the mu update IS temporally coupled (worker A2 fixed that), but mu still drifts because the (theta, p, ph) triplet that mu lives in is itself drifting.

---

### Hypothesis C — Prognostic pressure equation is missing a factor of ρ (= 1/α) (acoustic.py:185)

**File:line**: same as B, but a different framing — a "what if you keep the prognostic-pressure approach but fix it?" view.

The linearized continuity equation gives `∂p/∂t = -ρ c² ∇·v`, equivalently `∂p/∂t = -γ p ∇·v`. The worker uses

```python
p_next = state.p - c2 * dt_sub * div     # this is dp/dt = -c² ∇·v, missing ρ
```

The error factor is `code / correct = c² / (ρ c²) = 1/ρ = α`. At α=1 (surface, ρ≈1 kg/m³) this is roughly right. With α capped at 5.0 (post-fix), upper-tropospheric cells where ρ ≈ 0.2 see code pressure-response **5× too strong**. Combined with the symmetric `α` factor on the velocity side (`u_next = state.u - dt_sub * α * grad p`, acoustic.py:187), the effective sound-wave equation is

```
∂²p/∂t² = α · c² · ∇²p     (in the worker)
∂²p/∂t² = c² · ∇²p          (correct: α from velocity cancels 1/α from pressure)
```

So the effective sound speed is `sqrt(α) × c_true`. With α=5, that's ~2.24× the true sound speed (~780 m/s vs 348 m/s). The CFL diagnostic at acoustic.py:75-89 uses `c_true`, so it reports CFL ≈ 0.29 (subcritical) while the effective CFL is ≈ 0.65. Still nominally subcritical, but the dispersion relation is wrong — sound waves with the wrong restoring stiffness — and the asymmetry can pump the acoustic mode.

This bug is **co-resident** with Hypothesis B: if you choose to keep prognostic pressure (rather than rewrite to diagnostic), you still need to fix this factor. If you go the diagnostic-pressure route (B), this bug becomes moot.

**Why I rate this separate from B**: the c1 worker's pre-scoped redesign is explicitly diagnostic (Klemp 2007 §3a-c). But there is an alternative cheaper patch — keep prognostic pressure but at least make the wave equation isotropic by changing `c2 * dt * div` to `c2 / alpha * dt * div` (= `γ p * dt * div`). That is a **one-line edit** and would restore the dispersion symmetry without rewriting the formulation. It will NOT fix the missing buoyancy (Hypothesis A) and will NOT make (theta, p, ph) hydrostatically self-consistent (Hypothesis B), but it should reduce the post-fix nonfinite explosion to (at most) the original 76% fire rate by removing the runaway acoustic energy injection.

**Minimal discriminator test**:

```python
def test_acoustic_dispersion_relation_independent_of_alpha():
    """Effective sound speed from a 1D plane-wave should equal c_true, not sqrt(alpha)*c_true."""
    grid = make_ideal_grid(nz=4, ny=4, nx=128, dx_m=300.0)
    # set up at high altitude where alpha is large (cap-binding)
    state = state_at_pressure(p=10_000.0, theta=240.0)  # alpha = 287*240/1e4 = 6.9, capped at 5.0
    # impose a sinusoidal pressure perturbation, measure phase speed over 100 substeps
    c_measured = measure_sound_phase_speed(state, grid, n_substeps=100)
    c_true = jnp.sqrt(1.4 * 287.0 * 240.0)   # ~290 m/s
    assert abs(c_measured - c_true) / c_true < 0.05   # current code: ~sqrt(5)*c_true
```

**Likelihood**: MEDIUM as a *standalone* bug — but HIGH as the explanation for why the alpha-cap relaxation (bug-hunt #1 fix #2) was the proximate cause of the post-fix nonfinite explosion. Pre-cap-relaxation, α was clamped at 0.02 so the wave was just over-damped, not wrong-speed. Post-relaxation, the wrong dispersion can pump energy.

---

## 4. Recommendation

**Keep c1 (Klemp-Skamarock clean-room) running.** It is on a path that addresses all three hypotheses above:
- The Klemp 2007 tridiagonal w-ph solve carries the buoyancy term natively (Hypothesis A).
- Pressure becomes diagnostic from (theta, ph, mu) (Hypothesis B) — the c1 scope explicitly cites `module_small_step_em.F:527` as the diagnostic.
- The ρ factor is built into the formulation (Hypothesis C).

The 5-9 day estimate is consistent with the depth of these issues — they are not single-line edits.

**Cheap fix-hint to consider for the existing M6.x branch in parallel** (low cost, would NOT close M6.x but may convert the catastrophic post-fix probe back to "drift, not nonfinite explosion" and let the worker iterate more cheaply):
1. Apply Hypothesis C's one-line patch: change `acoustic.py:185` to `p_next = state.p - c2 / jnp.maximum(alpha, 1e-3) * dt_sub * div` (= `γ p × div`). This restores acoustic-wave dispersion symmetry.
2. Revert the alpha cap to a moderate value (0.5 or 1.0) until buoyancy is added. The 5.0 cap allows the worst dispersion error.

Neither (1) nor (2) addresses Hypothesis A (missing buoyancy) or Hypothesis B (prognostic-vs-diagnostic p) — those need c1's rewrite. But they could buy a stable enough "interim mode" to keep validation tooling exercised while c1 is in flight.

**Honest uncertainty**: I cannot run probes from this worktree (READ-ONLY on `/tmp/wrf_gpu2_m6x/`). The three hypotheses above are mechanism-of-failure analyses; the "minimal discriminator" tests are written to be runnable but I have not verified them against the current code. I am ~80% confident Hypothesis A (missing buoyancy) and Hypothesis B (prognostic-p drift) are real and correct. I am ~60% confident Hypothesis C explains the *quantitative* jump from 76% → 99.95% post-fix-2; it could also be that fix #2's removal of the asymmetric mask exposed an aliasing pattern I haven't traced.

What would change my mind: a probe in which Hypothesis C's one-line patch is applied and the failure stays at 99.95% (in which case C is wrong and A or B alone explains it), or a probe in which a buoyancy term is added (even with rough perturbation theta) and the failure drops back to drift-not-NaN (which would confirm A).

---

## 5. Files referenced (READ-ONLY)

- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/acoustic.py` (whole file)
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/advection.py:41-45, 175-280`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/rk3.py:35-72`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/tendencies.py:43-69`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/coupling/boundary_apply.py` (whole file)
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/coupling/driver.py:777-816, 819-905`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/coupling/physics_couplers.py:98-115, 270-281`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/contracts/state.py:35-49`
- `/tmp/wrf_gpu2_m6x/artifacts/m6/performance/m6x_failed_6h_direct_probe.json` (76% fire, drift)
- `/tmp/wrf_gpu2_m6x/artifacts/m6/performance/m6x_a2_fix3_6h_direct_probe.json` (99.95% fire, 3.89B nonfinite)
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:225-260, 485-560, 1444-1597` (advance_w with buoyancy, calc_p_rho diagnostic, divergence damping)
- `/tmp/wrf_gpu2_c1/.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/design.md` (c1 plan §2 — confirms diagnostic-p path)

No files in `/tmp/wrf_gpu2_m6x/` were modified by this review.
