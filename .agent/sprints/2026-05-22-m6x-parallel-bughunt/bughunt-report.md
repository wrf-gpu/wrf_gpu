# M6.x Parallel Bughunt — Code Review (Claude Opus 4.7 xhigh)

**Date**: 2026-05-22
**Reviewer**: Claude Opus 4.7 xhigh
**Worker under review**: codex on `/tmp/wrf_gpu2_m6x` (NOT taking over; orthogonal review)
**Anchor commit**: WIP, modified files in `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/` as of 2026-05-22 ~00:09 local
**Failure under investigation**: `full_domain_batching_m6x_failed_6h.outputs.json` (6h coupled probe failed; 1h probe had PH overflow warning per role-prompt)
**Direct probe artifact**: `artifacts/m6/performance/m6x_failed_6h_direct_probe.json` (manager-supplied)

---

## 0. Failure signature (from probe)

`m6x_failed_6h_direct_probe.json` gives the smoking gun. Headline numbers:

| step | h | changed | theta_min | theta_max | mu_min | mu_max | v_abs_max |
|------|---|---------|-----------|-----------|--------|--------|-----------|
| 360  | 1 |       0 | 290.6     | 492.8     |  68924 | 106149 |  10.24    |
| 720  | 2 |     580 | 288.1     | 492.7     |  52763 | 120000⚠ | 10.73    |
| 1080 | 3 |    1809 | 286.9     | 500.8     |  14535 | 120000⚠ | 11.78    |
| 1440 | 4 |    4730 | 284.9     | 550.0⚠    |   1000⚠ | 120000⚠ | 12.47    |
| 1800 | 5 |    8003 | 280.0     | 550.0⚠    |   1000⚠ | 120000⚠ | 12.93    |
| 2160 | 6 |   54923 | 150.0⚠    | 492.1     |   1000⚠ | 120000⚠ | 12.93    |

`nonfinite_count = 0`, `clip_count = 14_880_761`, sanitize firing rate 76% of steps. **This is not NaN explosion — it is slow, deterministic drift into the clip bounds.** Mu spreads bipolarly (low side runs to floor, high side to ceiling); theta overshoots high through ~h4 then collapses below the floor by h6 as the floor-mu columns get compressed. v_abs_max climbs monotonically. This signature is **mass-conservation failure** + **acoustic-mode under-damping**, not a one-cell instability blowup.

That rules out, with high confidence, several of the role-prompt's a-priori candidates:
- "CFL diagnostic not actually clamping `n_acoustic`" — diagnostic is correct (see `test_m6x_cfl_diagnostic.py` and `acoustic_cfl_diagnostic` body); failure does not look CFL-limit-grade explosion. CFL at default state: `347.2 * 2.5 / 3000 ≈ 0.29` — comfortably below 1.
- "Sign error in mu tendency that integrates over many steps" — sign convention is verified by `test_mu_tendency_matches_column_flux_divergence_oracle` (expects `tendency == -80000/1000`, i.e., negative for divergent flow). The oracle is correct in sign.
- "Missing top/bottom boundary condition in vertical" — top and bottom are explicitly zeroed in `_grad_z_to_w` (acoustic.py:137-139) and `_layer_thickness_m` clamps at 1.0 m floor (acoustic.py:124). Not silent BC violation.

What IS consistent with the signature: a small, systematic, every-step error in the (u,v,w,p,ph) ↔ mu coupling that compounds. Three concrete bugs follow.

---

## 1. Code review (per file, with line citations)

### 1.1 `dynamics/acoustic.py` (the main suspect)

Constants and conversions (`acoustic.py:24-31`):

```
GAMMA_DRY_AIR = 1.4
R_DRY_AIR = 287.0
P0_PA = 100000.0
KAPPA_DRY_AIR = R_DRY_AIR / CP_DRY_AIR
GRAVITY_M_S2 = 9.80665
PRESSURE_IMPLICIT_RELAXATION = 0.05      # <-- magic number, no WRF citation
MAX_INVERSE_DENSITY = 0.02
```

`PRESSURE_IMPLICIT_RELAXATION = 0.05` is not derivable from `module_small_step_em.F`. WRF's only multiplicative coefficient on the small-step pressure update is `smdiv` (divergence damping, line 562) and that gates a *correction*, not the main update. **No WRF source citation in the file justifies 0.05** — it appears to be a hand-tuned damping factor. This is bug 2 below.

Pressure-update line (`acoustic.py:192`):

```
p_next = state.p - PRESSURE_IMPLICIT_RELAXATION * c2 * dt_sub * div
```

`p_next` is total pressure (since `state.p = P + PB` per driver.py:89). Two things wrong here:

(i) The 0.05 prefactor reduces acoustic pressure response by 20×. Velocity divergence is *not* absorbed back into the pressure field at the rate physical sound waves require → divergence accumulates step-on-step → mu drifts via the mass-continuity equation.

(ii) `div` is *masked* by `_vertical_implicit_mass_weight(state, dt_sub)` (acoustic.py:189-191). The same mask is NOT applied when computing the pressure gradient force on u, v, w (acoustic.py:194-196). The two sides of the acoustic mode are then inconsistent: velocity feels full `grad p_dynamic`, but `p` is updated from a damped divergence. **This is non-conservative by construction.** Energy/mass can leak through the asymmetry. See bug 2 below.

**PH update (acoustic.py:198) — single most likely root cause:**

```
ph_next = state.ph + dt_sub * w_next
```

Units check:
- `state.ph` is geopotential, m²/s² (state.py:160; loaded as `PH + PHB`, driver.py:90).
- `dt_sub * w_next` is `[s] * [m/s] = [m]`.
- `[m²/s²] + [m]` is dimensionally inconsistent.

The geopotential equation in non-hydrostatic eta-coordinate dycores (Skamarock 2008 ARW Tech Note, eq. 2.5) is `∂φ/∂t = -[advection terms] + g·w`. The simplest (advection-free) acoustic update is `ph += dt * g * w`. WRF canonical small-step PH update is more elaborate (line 1583):

```
ph(i,k,j) = rhs(i,k) + msfty * 0.5 * dts * g * (1.+epssm) * w(i,k,j) / (c1f(k)*muts(i,j)+c2f(k))
```

— note the `g`, the off-centering `0.5*(1+epssm)`, and the `mut` coupling. Even stripped of the off-centering and map factors, the **`g`** is mandatory. The worker's line 198 is missing it.

**Consequence**: PH evolves at 1/g ≈ 1/9.81 of the rate it should. `_layer_thickness_m = (ph[1:] - ph[:-1]) / g` (acoustic.py:124) therefore drifts almost not at all even when the vertical wind says it should. That stale `dz` then enters:
- `_vertical_divergence_cgrid` (acoustic.py:104) — denominator of `(w[k+1]-w[k])/dz`, so vertical divergence is wrong magnitude;
- `_grad_z_to_w` (acoustic.py:132-139) — `dz_face` denominator, so vertical pressure gradient force is wrong magnitude;
- `_vertical_implicit_weight` and `_vertical_implicit_mass_weight` — `vertical_cfl` based on `dz_face` is wrong, so the implicit damping factor is wrong.

The bug compounds because PH is the only state variable that mediates "how thick is this layer right now," and every vertical operator in `acoustic.py` reads it.

The pressure perturbation reconstruction (acoustic.py:157-160) is correct in form:

```
def _pressure_perturbation(total_pressure, base_pressure):
    return total_pressure - base_pressure
```

`state.pb` is the dry base loaded once from Gen2 and never updated, which is the right convention (base state is by definition time-invariant in WRF). So the role-prompt's candidate "Incorrect pressure-perturbation reference state (state.pb baseline)" is not the failure mode — what is wrong is *upstream*: `state.p` itself is drifting because the pressure-divergence loop is under-damped (bug 2) and the PH-feedback is wrong (bug 1). `p - pb` then drifts mechanically, and the pressure gradient force at `acoustic.py:194-196` injects spurious horizontal acceleration → v_abs_max climbs.

Other notes on acoustic.py:
- `_inverse_density` (acoustic.py:163-166) caps `alpha` at `MAX_INVERSE_DENSITY = 0.02`. For a 1000-hPa surface column with T=290K: `alpha = R*T/p ≈ 287*290/1e5 ≈ 0.83`. The cap at 0.02 is **42× too tight** — clamps every cell in the troposphere. Whether this is a stabilizer or a bug is unclear from the comments, but it does make the pressure-gradient force on (u,v,w) at acoustic.py:194-196 systematically 42× too weak.
- `_vertical_implicit_weight` is `1/(1 + cfl⁴)` (acoustic.py:174). For CFL=1.0 this is 0.5; for CFL=2.0 this is 0.06; it goes to 0 as CFL grows. That is heuristic, not the Klemp-Skamarock tridiagonal Thomas solve. Functions but introduces an unphysical w-damping that may also contribute to drift.

### 1.2 `dynamics/tendencies.py`

`compute_mu_tendency` (tendencies.py:43-54) is **mathematically correct** for the column-integrated mass continuity. Sign, layer weights, face interpolation, and divergence stencil all check out against the WRF formula at `module_em.F` line 1094-1099 (after the necessary substitutions for the worker's flux convention without map-scale factors). The unit tests in `tests/test_m6x_mu_continuity.py` cover (a) zero-divergence trivial case, (b) analytic divergent-flow oracle, (c) RK3 monotonicity. All pass.

The **wiring** is correct (advection.py:279 adds `compute_mu_tendency(haloed, grid)` into `base.mu`). The PROBLEM is not in `compute_mu_tendency` itself — see bug 3.

### 1.3 `dynamics/rk3.py`

`rk3_step` (rk3.py:42-62) calls `forward_backward_acoustic` only at stages 2 and 3 (line 52, 58), with `dt/2` and `dt`. mu is *not* updated inside the acoustic loop. mu is only updated through `add_scaled_tendencies(origin, tendencies, dt_stage)` at `rk3_stage` (rk3.py:38-39). The acoustic substeps run on a state whose `mu` is held fixed for the duration of that RK stage's small-step loop.

WRF canonical small-step (module_small_step_em.F:1102-1108) updates mu **every small step** from the small-step-evolved velocities:

```
DO i=i_start, i_end
  MUAVE(i,j) = MU(i,j)
  MU(i,j) = MU(i,j)+dts*(DMDT(i)+MU_TEND(i,j))
  ...
ENDDO
```

In the worker, mu sees a coarsened approximation: `compute_mu_tendency(stage_state, grid)` is evaluated once per RK stage on the pre-acoustic velocity field, and `mu = origin.mu + dt_stage * mu_tendency`. After the acoustic loop runs, mu has not absorbed the divergence the acoustic substeps just generated. The next RK stage's `compute_mu_tendency` sees velocities that have been acoustic-modified but mu that was integrated against the OLD velocities. Mismatch accumulates. Bug 3 below.

### 1.4 `dynamics/step.py`

Looks fine — thin jit wrapper around `rk3_step`. Static-argnames are correct (`grid`, `dt`, `n_acoustic`, `debug` all static). No issues found.

### 1.5 `coupling/driver.py`

The lateral-boundary-replay, sanitize logic, and segmentation around `run_forecast_segment` (driver.py:281-388) look fine. Sanitize bounds `mu ∈ [1000, 120000]` Pa and `theta ∈ [150, 550]` K (driver.py:826, 830) are the **clip bounds** the probe is hitting — they are the *symptom*, not the cause. Sanitize is the seatbelt; the dycore is the crash.

One operational note: `state.pb` is in the State pytree but is NOT in any sanitize / boundary-replay code path (good — pb is supposed to be invariant). It is also NOT in the halo `DYCORE_HALO_FIELDS` tuple (advection.py:13) — also correct since pb is a static base profile.

### 1.6 `contracts/state.py`

State pytree definition is consistent with what the dycore consumes. `pb` is present (`state.py:84`) and zero-initialized in `State.zeros` (state.py:315), populated by `driver.build_initial_state` (driver.py:88, 96-105). No issue.

---

## 2. Top 3 bug hypotheses (ranked by likelihood)

### Hypothesis #1 — PH update is missing the gravitational acceleration g (acoustic.py:198)

**File:line**: `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/acoustic.py:198`

```python
ph_next = state.ph + dt_sub * w_next        # WORKER (wrong units)
```

**Should be** (simplest canonical form, dropping off-centering and mass-coupling for the M6.x scope):

```python
ph_next = state.ph + dt_sub * GRAVITY_M_S2 * w_next
```

**WRF canonical citation**: `dyn_em/module_small_step_em.F:1583-1584`:

```fortran
ph(i,k,j) = rhs(i,k) + msfty(i,j)*.5*dts*g*(1.+epssm) * w(i,k,j)/(c1f(k)*muts(i,j)+c2f(k))
```

— the `dts*g*w` term is universal across all WRF dycore versions; the `0.5*(1+epssm)` is the off-centering only.

**Mechanism by which this produces the observed signature**: `state.ph` is unitfully geopotential (m²/s²), `dt*w` is meters. Adding them is dimensionally wrong. PH increments are too small by ~g≈9.81. `_layer_thickness_m = (ph[1:]-ph[:-1])/g` then doesn't update at the rate physical vertical motion requires. Consequence: every operator that reads `_layer_thickness_m`, including the vertical divergence and the vertical pressure gradient (acoustic.py:102-104, 132-139) sees a stale `dz`. Vertical and horizontal modes desynchronize; mass continuity (which depends on velocity divergence) drifts; mu spreads to clip bounds.

**Minimal verification test** (single-line python to add to `tests/test_m6x_dycore_completion.py`):

```python
def test_ph_evolves_with_g_factor_under_uniform_w():
    grid = make_ideal_grid(nz=4, ny=4, nx=4, dx_m=3000.0, dy_m=3000.0)
    state, _ = _physical_state(nx=4, ny=4, nz=4)
    state = state.replace(w=jnp.ones_like(state.w) * 1.0)  # 1 m/s vertical wind
    out = acoustic_once(state, grid, dt_sub=0.25)
    interior = jnp.mean(out.ph[1:-1] - state.ph[1:-1])
    # Expected: dt_sub * g * w = 0.25 * 9.80665 * 1.0 = 2.4517
    assert float(interior) == pytest.approx(0.25 * 9.80665 * 1.0, rel=0.05)
```

This test currently FAILS for the worker code (would yield ~0.25 instead of ~2.45). One-line fix at acoustic.py:198 (multiply by `GRAVITY_M_S2`) makes it pass.

**Likelihood**: VERY HIGH. This is a unit error in a single line that participates in every grid cell every acoustic substep; the failure pattern (slow drift compounding over hours, no NaN explosion) is exactly what a low-amplitude systematic mis-scaling produces.

---

### Hypothesis #2 — Asymmetric implicit damping breaks acoustic-mode mass conservation (acoustic.py:189-196)

**File:line**: `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/acoustic.py:189-196`

```python
div = _vertical_implicit_mass_weight(state, dt_sub) * (
    _horizontal_divergence_cgrid(state, grid) + _vertical_divergence_cgrid(state)
)
p_next = state.p - PRESSURE_IMPLICIT_RELAXATION * c2 * dt_sub * div         # ←  div is masked
p_dynamic = _pressure_perturbation(p_next, state.pb)
u_next = state.u - dt_sub * _mass_to_u_face(alpha) * _grad_x_to_u(p_dynamic, grid)   # ← grad p_dynamic is unmasked
v_next = state.v - dt_sub * _mass_to_v_face(alpha) * _grad_y_to_v(p_dynamic, grid)
w_explicit = state.w - dt_sub * _mass_to_w_face(alpha) * _grad_z_to_w(p_dynamic, state)
```

**Two compounding problems on these lines:**

(2a) `PRESSURE_IMPLICIT_RELAXATION = 0.05` is a 20× under-damping with no WRF source. The canonical Klemp-Skamarock prognostic pressure equation `dp/dt = -c² · div(ρv) + ...` has no such prefactor. The 0.05 makes the pressure response too soft, so velocity divergence is never absorbed at sound-wave timescale; mass distortion accumulates.

(2b) `_vertical_implicit_mass_weight` masks the divergence used to update `p`, but **the same mask is not applied** to `_grad_x/_grad_y/_grad_z(p_dynamic)`. The two sides of the acoustic oscillation see inconsistent coefficients — velocity is forced by full `∇p'` while pressure is updated from a softened `div`. Numerical energy/mass is not conserved by construction.

**WRF canonical citation**: `dyn_em/module_small_step_em.F:527-528` (diagnostic perturbation pressure from theta + ph and ideal gas law, no relaxation factor), and `module_small_step_em.F:562` (divergence damping is `p = p + smdiv*(p-pm1)`, a symmetric *correction* applied identically across the wave; not an asymmetric prefactor on the velocity divergence). The worker is doing something WRF does not.

**Mechanism**: Acoustic modes are how the system enforces mass conservation in a split-explicit dycore. If you under-respond to divergence by 20× (2a) and asymmetrically damp the pressure side but not the velocity side (2b), residual divergence builds, mu must absorb it via mass continuity, and mu drifts to both clip bounds (bipolar drift, exactly what the probe shows).

**Minimal verification test**:

```python
def test_acoustic_substep_conserves_column_mass_to_round_off():
    """Column-integrated divergence should drive a numerically matched pressure response."""
    grid = make_ideal_grid(nz=8, ny=8, nx=8, dx_m=3000.0, dy_m=3000.0)
    state, _ = _physical_state(nx=8, ny=8, nz=8)
    # Set up a single-mode divergence; advance 100 substeps; column mass should drift < 0.1 Pa
    u_div = 1.0e-3 * jnp.cos(2*jnp.pi*jnp.arange(grid.nx+1)/grid.nx)[None, None, :]
    state = state.replace(u=jnp.broadcast_to(u_div, state.u.shape))
    sub = state
    for _ in range(100):
        sub = acoustic_once(sub, grid, dt_sub=0.5)
    drift = float(jnp.max(jnp.abs(jnp.sum(sub.p - state.p, axis=0))))
    assert drift < 1.0   # current code: drift will scale with the asymmetric damping
```

Currently the worker likely produces a drift much larger than 1 Pa over 100 substeps due to (2a)+(2b).

**Likelihood**: HIGH. The bipolar mu drift in the probe (mu running to both 1000 Pa floor and 120000 Pa ceiling) is harder to explain by hypothesis #1 alone (which would predict a coherent bias). Asymmetric acoustic damping is a natural producer of *both* over- and under-mass columns within the same field, depending on local divergence sign.

---

### Hypothesis #3 — `mu` integration is temporally decoupled from acoustic substeps (rk3.py + tendencies.py)

**File:line**: `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/rk3.py:42-62` (specifically lines 51-58) + `dynamics/tendencies.py:67`

In the worker:
- `compute_mu_tendency` evaluates at the *stage state* (pre-acoustic).
- `add_scaled_tendencies` applies `state.mu + dt_stage * tendencies.mu` at the start of each RK stage (tendencies.py:67).
- `forward_backward_acoustic` then runs `n_acoustic` small steps that modify (u, v, w, p, ph) but **never touch mu**.
- mu does not see the acoustic-evolved divergence until the next RK stage's `compute_advection_tendencies` re-reads it on a fresh stage_state.

**WRF canonical**: `dyn_em/module_small_step_em.F:1102-1108` updates mu **inside** the small-step loop using small-step velocities and `dts`, every small step. Comments at line 1071-1088 explicitly note that the column-integrated mass continuity `DMDT` accumulates across all vertical layers and `MU = MU + dts*(DMDT + MU_TEND)` is applied as a small-step prognostic.

**Mechanism**: Over 2160 large steps × 3 stages × 4 substeps = 25,920 acoustic substeps, mu sees velocity divergence at coarsened (RK-stage) frequency instead of small-step frequency. With acoustic CFL ≈ 0.29 this is a small per-substep error, but it integrates over 6h into the kind of slow drift the probe shows. Combined with bug #2's asymmetric damping, the small per-substep mu error is also biased per cell (not unbiased noise), so it doesn't cancel.

**WRF canonical citation**: `dyn_em/module_em.F` `advance_mu_t` is the equivalent of `module_small_step_em.F:969-1175` (it's called from inside the small-step loop, not from outside). The worker calls the equivalent (`compute_mu_tendency`) once per RK stage in `compute_advection_tendencies`. Architecturally different.

**Minimal verification test** (run-time, not unit): re-run the 6h direct probe with an instrumented branch that updates mu inside the acoustic scan body. Diff the per-step mu drift against the current worker. Expected: mass conservation tightens by 1-2 orders of magnitude. This is more involved than a unit test (requires a code branch) but should be cheap.

**Likelihood**: MEDIUM. Real algorithmic mismatch with WRF canonical, but on its own it would produce a *bias* in mu drift, not the *bipolar* spread we see. Probably a contributor but not the leading order term until #1 and #2 are fixed.

---

## 3. Verification suggestions summary

| # | One-line discriminator | What it tells you |
|---|------------------------|-------------------|
| 1 | `acoustic_once(state, grid, 0.25)` with `state.w = 1.0` uniform → check `ph` increment in interior ≈ 0.25 * 9.81 ≈ 2.45 m²/s² (currently ≈ 0.25) | Confirms #1 (g missing) |
| 2 | 100 substeps with a fixed-amplitude horizontal divergence → check column-integrated `p` drift stays bounded | Confirms #2 (asymmetric damping) |
| 3 | Branch that updates `mu` inside the acoustic `lax.scan` body using small-step `compute_mu_tendency`; re-run 6h direct probe | Confirms #3 (temporal decoupling) |

Fix #1 first (5 character edit at acoustic.py:198), re-run the 6h direct probe, see if mu drift is reduced. If yes, also remove the `0.05` factor (or set to ~1.0) and re-run. If the probe is still failing, do #3.

---

## 4. Algorithmic-alternatives assessment (c1 Klemp-Skamarock vs fix-current)

Reading `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/design.md` §2 (option c1), the c1 approach is a clean-room implementation of Klemp et al. 2007 §3a-c with a **per-column vertical-implicit tridiagonal Thomas solve** replacing the current `_vertical_implicit_weight`. Estimated 5-9 wall-days.

**My recommendation: do NOT invoke c1 yet.** The current worker's code has three concrete, citable defects (above) that are 1-line to many-line fixes. The failure signature (slow drift, no NaN, mu hitting both clip bounds) is *consistent* with these bugs and *inconsistent* with "the algorithm is fundamentally broken" — a fundamentally wrong algorithm typically diverges in hours-to-minutes, not in a clean 6h linear drift. Option c1 should remain insurance, not the next step.

**However**, if dispatching the fixes for #1+#2+#3 above to the worker still fails the next 6h probe, c1 becomes the correct pivot. The hard part of c1 (the per-column tridiagonal Thomas solve via `jax.lax.linalg.tridiagonal_solve` or hand-rolled vmapped Thomas) replaces exactly the heuristic `_vertical_implicit_weight` (acoustic.py:169-181) that I called out as unphysical. If three rounds of fix-hint don't stabilize the current code, the cleanest path forward is to throw out `_vertical_implicit_weight` and adopt the Klemp-2007 vertical-implicit acoustic damping wholesale.

---

## 5. Recommendation to manager

**DISPATCH FIX-HINT TO WORKER**, do not take over, do not invoke c1 yet.

Concretely, the message I would send the worker:

> Three suspected bugs found in code review (cite this report). Try fixes in this order, re-running the 6h direct probe between each:
>
> 1. `acoustic.py:198`: `ph_next = state.ph + dt_sub * GRAVITY_M_S2 * w_next` — add the missing g factor. This is the highest-confidence single-line fix. Re-run probe.
> 2. `acoustic.py:30`: change `PRESSURE_IMPLICIT_RELAXATION = 0.05` to `1.0` (or remove the factor entirely from `acoustic.py:192`). Also remove the `_vertical_implicit_mass_weight` mask from `acoustic.py:189` so the divergence used to update `p` matches the gradient used to update (u,v,w). Re-run probe.
> 3. If still failing, restructure `forward_backward_acoustic` so mu is updated inside the `lax.scan` body using small-step `compute_mu_tendency`. This is more involved but matches WRF canonical (module_small_step_em.F:1102-1108).
>
> Tests to add as you go: `test_ph_evolves_with_g_factor_under_uniform_w` and `test_acoustic_substep_conserves_column_mass_to_round_off` (see bughunt report §2 for full code).
>
> Also: `MAX_INVERSE_DENSITY = 0.02` (acoustic.py:31) clamps `alpha = R*T/p ≈ 0.83` for normal tropospheric cells, making the horizontal pressure gradient force 42× too weak everywhere. Audit this — either it has a citation I missed, or it's a stabilizer that's now masking a real bug and should be relaxed (e.g., to 5.0 or removed).

Manager-side: keep c1 fallback warm (the design doc is ready); revisit if the worker comes back red after the three-step fix sequence. The c1 wall estimate (5-9 days) is real and you don't want to spend it on a bug that's a 5-character edit away.

---

## 6. Files referenced (read-only)

- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/acoustic.py:24-217`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/step.py:1-52`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/tendencies.py:1-69`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/rk3.py:1-82`
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/dynamics/advection.py:1-281` (read for tendency wiring; correct)
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/coupling/driver.py:72-1125` (no issue found in coupling path)
- `/tmp/wrf_gpu2_m6x/src/gpuwrf/contracts/state.py:1-382`
- `/tmp/wrf_gpu2_m6x/tests/test_m6x_dycore_completion.py:1-62`
- `/tmp/wrf_gpu2_m6x/tests/test_m6x_mu_continuity.py:1-50`
- `/tmp/wrf_gpu2_m6x/tests/test_m6x_cfl_diagnostic.py:1-44`
- `/tmp/wrf_gpu2_m6x/artifacts/m6/performance/full_domain_batching_m6x_failed_6h.outputs.json`
- `/tmp/wrf_gpu2_m6x/artifacts/m6/performance/m6x_failed_6h_direct_probe.json`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:200-290, 510-568, 1050-1112, 1340-1400, 1570-1600, 1940-1960`
- `/tmp/wrf_gpu2_m6x_bughunt/.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/design.md:1-100`

No files in `/tmp/wrf_gpu2_m6x/` were modified by this review.
