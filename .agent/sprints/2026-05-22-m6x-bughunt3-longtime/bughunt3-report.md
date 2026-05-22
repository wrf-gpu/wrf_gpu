# M6.x Bug-Hunt #3 — Long-Time Integration Instability

**Date**: 2026-05-22
**Reviewer**: Claude Opus 4.7 (1M context)
**Anchor**: `/tmp/wrf_gpu2_c1/` (READ-ONLY) — post-c1-A2 advection/coupling-fix landing
**Failure under investigation**: `artifacts/m6x-fallback-c1/c1_a2_post_fixes_1h.json`
- 18-step probe (`c1_a2_post_fix4_0p05h.json`): `fired_steps=0`, all leaves finite, dycore + couplers clean
- 60-step acoustic-only probe (per role prompt): stable in isolation
- 360-step 1h coupled probe: `fired_steps=319/360 (88.6%)`, `nonfinite_count=793 817 606`, `clip_count=274 320 190`, final state saturated at sanitize clip bounds (`theta∈{150,550} K`, `|u|,|v|=150 m/s`, `|w|=50 m/s`, `p∈{1000,120000} Pa`), `all_state_leaves_finite=true` after sanitize.

This is **not** an operator-correctness bug. The previous opus reviews (bug-hunts #1 and #2) and the c1 + c1-A2 worker iterations already audited every operator. What I am hunting is **what accumulates over 360 large steps × 3 RK stages × 86 acoustic substeps (≈ 92 880 forward-backward iterations / hour) once the operators are individually correct.**

---

## 1. Failure-signature analysis: why 18 steps clean but 1 h dirty

`requested_n_acoustic=2` was promoted to `n_acoustic=86` by `rk3.py:49` via `max(int(n_acoustic), required_n_acoustic(grid, dt))`. So one coupled step does:

- RK stage 1: advection only.
- RK stage 2: advection on stage-1 state, then 86 forward–backward acoustic substeps at `dt_acoustic = 5/86 ≈ 0.058 s`.
- RK stage 3: advection on stage-2 state, then 86 forward–backward acoustic substeps at `dt_acoustic = 10/86 ≈ 0.116 s`.
- Thompson → MYNN → surface (and RRTMG on the cadence).
- **One** `apply_lateral_boundaries` at the end.
- Sanitize.

18-step probe = 18 × (3 RK + 2×86 substeps) ≈ 4 640 substep advances, with the boundary refresh enforced 18 times. 1 h probe = 360 × ≈ 92 880 substep advances, with the boundary refresh enforced 360 times. That is a 20× longer cumulative substep horizon, and several effects below scale super-linearly with substep count.

Critically, `fired_steps=319` ≠ `nonfinite_count/total≈37%`. The pattern is "after some number of clean steps, sanitize starts firing on virtually every step and a sizable fraction of cells go non-finite each step". That is consistent with **one slow drift driving a normally-conditioned linear system into singular regime**, not with "this step's operator returned NaN".

The single biggest piece of probe forensics is **`changed_count = 1.07 B` against `total_checked_values = 2.89 B`** ⇒ 37 % of *all* sanitized values per step are being touched on average. Of those, `nonfinite` (793 M) > `clipped` (274 M); i.e., most touches are NaN/Inf, not over-range. NaN comes from a small number of operations that produce Inf: division by zero, sqrt of negative, log of zero, or 0×Inf. In this dycore the only obvious source is the implicit acoustic tridiagonal when `dz_face → 0` or negative (see Hypothesis #2 below).

---

## 2. Per-suspect audit (role-prompt suspects #1–#7)

### Suspect #1 — Sanitize itself injects shocks
**Status**: Plausible but secondary; sanitize-disable probe still needed to discriminate.

`coupling/driver.py:819-854` clips `u,v∈[-150,150]`, `w∈[-50,50]`, `theta∈[150,550]`, `mu,p∈[1000,120000]`, `qv∈[0,0.05]`, etc. Once a cell hits a clip bound, the *next* step's stencil at neighbours sees a piecewise-constant jump that drives spurious gradient kicks (acoustic horizontal grad, advection 5-point flux). Once started, this is self-reinforcing.

But the 18-step probe has `fired_steps=0` — no clip event in the first 18 steps. So sanitize is NOT the **initiator**; it is the **accelerator** once something else first crosses a bound. Removing sanitize would not produce a clean run; it would produce earlier non-finite states.

### Suspect #2 — Physics coupler timing vs new c1 acoustic
**Status**: Audited, no smoking gun, but it is the path where theta perturbations are injected without the dycore's pressure re-diagnosis.

`coupling/driver.py:800-816` runs dycore → Thompson → MYNN → surface → RRTMG. Thompson updates `theta` and `qv`; RRTMG updates `theta`; MYNN updates `u,v,w,theta,qv,qke`. None of these touches `p`, `pb`, `ph`, or `mu`. After physics, `state.p` is unchanged but `state.theta` has been re-mixed by PBL and re-warmed by RRTMG. The **next** large step's RK1 reads this `theta` and produces an advection tendency from it — but the acoustic substep equations in `acoustic.py:315-344` are *independent of theta* (see Hypothesis #1 below). So theta-physics → no immediate inconsistency in the acoustic substep itself.

The c1-A2 "perturbation-pressure-advection accepted" change (advection.py:411-419) advects `p_perturbation = p − pb` and adds the tendency to `p`. After physics that updates only theta, `p − pb` is still consistent (`p` unchanged, `pb` fixed). So the timing is internally consistent.

### Suspect #3 — Boundary replay timing under c1
**Status**: Real concern; partially audited.

`coupling/boundary_apply.py:apply_lateral_boundaries` replaces `u, v, theta, qv, ph, mu` at edges from the Gen2 forcing. It does **NOT** apply `p` or `pb` (only u, v, theta, qv, ph, mu — see `boundary_apply.py:39-45`). So the *total* and *base* pressure at the boundary edge are whatever the c1 dycore last produced; the *geopotential* at the boundary is forced to Gen2 PH+PHB. Internal column hydrostatic relation `(ph[k+1] − ph[k])/g · (-dp/dz) ≈ ρg·dz` is consistent in Gen2 but not necessarily in the c1 dycore (see Hypothesis #1: c1 uses reference α and reference ρc² everywhere, so it cannot reproduce real-state hydrostatic relations exactly). After 360 steps of boundary forcing on ph but not p, the edge cells carry a ph that says "this is a baroclinic atmosphere" while p drifted to a barotropic-ish value driven by reference-α acoustic damping.

Boundary apply at end of step also overwrites `theta`, but the **next** RK stage 1 reads this `theta` and recomputes the advection tendency on a discontinuous edge — a 5th-order flux stencil hates this.

### Suspect #4 — Halo handling under c1
**Status**: **CRITICAL**. The halo is a no-op AND every horizontal stencil in the dycore uses `jnp.roll` (periodic) on a non-periodic d02 grid. See Hypothesis #A below.

`contracts/halo.py:28-32`:
```python
def apply_halo(state: State, halo: HaloSpec) -> State:
    del halo
    return state
```
The "halo" exchange is literally a no-op — it does not copy interior cells into a halo region nor enforce any boundary condition. Meanwhile:
- `acoustic.py:139-147` (`_grad_x_to_u`, `_grad_y_to_v`): `jnp.roll` periodic gradient
- `acoustic.py:162-170` (`_mass_to_u_face_2d`, `_mass_to_v_face_2d` for μ flux): `jnp.roll` periodic interpolation
- `advection.py:218-229` (`_periodic_flux5_faces`): `jnp.roll` 5-point periodic flux interpolant
- `advection.py:303-312` (`_mass_to_u_face`, `_mass_to_v_face`): `jnp.roll` periodic mass-to-face interpolation

So every dycore-internal horizontal stencil treats the grid as periodic. The c1-A2 worker fixed `physics_couplers._mass_to_u_face/_mass_to_v_face` to be non-periodic (`physics_couplers.py:98-109`), but did NOT touch the corresponding routines in the dycore. **The dycore wraps east-edge values into the west boundary inside every acoustic substep.**

### Suspect #5 — Vertical advection at upper boundary
**Status**: Mild concern; not the dominant accumulator.

`advection.py:40-45` `_dz_from_state` floors `dz` at 1.0 m and falls back to flat-column dz where `raw <= 0`. `acoustic.py:82-91` `_layer_thickness_m` has NO 1 m floor and falls back to `_flat_dz(grid) = top_pressure_pa / nz` (≈ 125 m for nz=40, top=5000 Pa) only when `dz <= 0`. So in the regime where ph drift makes `dz` small-positive (say 0.5 m at one face), `acoustic` uses 0.5 m and `advection` uses 1.0 m — **inconsistent dz across the two modules at the SAME cell**. This is a real defect, see Hypothesis #2 below.

### Suspect #6 — Energy / mass conservation across the coupled timestep
**Status**: Not separately tested.

The advection scalar-conservation oracle (passes at 1e-10 per c1-A2 worker report) only covers periodic-domain mass-flux-form scalar advection. The 1 h coupled step has 360 × (3 RK + 2 × 86 acoustic + boundary apply + physics) ≈ 372 240 invariant-eligible operations. Nothing currently asserts that *over* a coupled step (a) integrated μ does not drift, (b) integrated dry air mass does not drift, (c) total water is bounded. Empirically the probe shows μ saturating at both clip bounds — the bipolar-spread signature already noted by bug-hunt #1, but it persists in c1 because the conservation guarantee from Klemp §3c (small-step μ accumulation) does not extend to (a) horizontal wrap pollution, (b) physics updates, or (c) boundary replay.

### Suspect #7 — physics_couplers.py order in driver
**Status**: Audited; order matches WRF for split-explicit non-hydrostatic dycores. Not the bug.

---

## 3. Top 3 hypotheses NEW to bug-hunt #3 (different from #1 and #2)

### Hypothesis #A — Periodic-wrap horizontal stencils on a non-periodic limited-area grid

**File:line**:
- `contracts/halo.py:28-32` — `apply_halo` is a no-op for all `edge_type`
- `dynamics/advection.py:19` — `halo_spec(grid)` declares `edge_type="periodic"` for d02 (limited-area)
- `dynamics/acoustic.py:139-147` — `_grad_x_to_u`, `_grad_y_to_v` use `jnp.roll`
- `dynamics/acoustic.py:162-170` — `_mass_to_u_face_2d`, `_mass_to_v_face_2d` use `jnp.roll`
- `dynamics/advection.py:218-229` — 5th-order scalar flux uses `jnp.roll` for stencils 3 cells deep
- `dynamics/advection.py:303-312` — mass-to-face interpolation uses `jnp.roll`

**Mechanism**:
Inside one coupled step, the dycore runs the equivalent of ≈ 260 substep advances (3 RK stages × ≈ 86 acoustic) with periodic east↔west and north↔south wraps. The boundary replay then overwrites the outermost 5 cells per side. With `spec_zone=1, relax_zone=4`, that means cells `[0]` are pinned and cells `[1..3]` are relaxation-blended.

For a non-periodic d02, the east edge can carry a large-amplitude gradient (e.g., African Atlantic boundary against the North-East trades). `jnp.roll(p, 1, axis=2)` at column 0 returns the column from the **far east edge**. The 5th-order flux at column 5 uses cells `[2,3,4,5,6,7]` — all in or just outside the relax zone — fine for that step. But at column 1 (which the relax zone will overwrite anyway), the flux uses cells `[N-2, N-1, 0, 1, 2, 3]` — i.e., **the far east side of the domain** is mixed into the west relax zone within every acoustic substep before boundary replay restores it.

The bias is small per substep (≈ 5th-order error in a wrap), but ≈ 92 880 substeps per hour generate a coherent periodic-mode standing wave whose amplitude grows monotonically. The 18-step probe sees ≈ 4 640 substeps — under the threshold; 1 h crosses it.

**Why bug-hunts #1 and #2 missed it**: bug-hunt #2 (§2 line 38-39) flagged the SAME pattern in `physics_couplers._mass_to_u_face/_mass_to_v_face`. The c1-A2 worker fixed *those* (`physics_couplers.py:98-109`), but did not touch the corresponding `_mass_to_u_face/_mass_to_v_face` inside `dynamics/advection.py:303-312`, nor `_grad_x_to_u/_grad_y_to_v` inside `dynamics/acoustic.py`. The wrap is still live in the dycore itself.

**WRF citation**: `dyn_em/module_em.F` uses explicit `i_start, i_end` tile bounds and `bdyzone` halo widths; it does not roll horizontally. The non-periodic specified-zone treatment is in `solve_em.F` boundary subroutines.

**Minimal discriminator**:
Replace `jnp.roll` in `_grad_x_to_u`, `_grad_y_to_v`, `_mass_to_u_face_2d`, `_mass_to_v_face_2d`, `_periodic_flux5_faces`, `_mass_to_u_face`, `_mass_to_v_face` with edge-mirrored extrapolation (the c1-A2 pattern from `physics_couplers.py`). Re-run the 1 h probe. If `nonfinite_count` drops by ≥ 1 order of magnitude, this is the dominant contributor.

**Likelihood**: **HIGH**. It explains both (a) 18-step clean / 1 h dirty (substep-count threshold), (b) bipolar μ saturation (wrapped east values systematically over- or under-shoot west targets depending on local field), (c) why the c1 "60-step acoustic-only" probe is stable in isolation — that probe presumably uses an idealized periodic-friendly grid where wrap is benign.

---

### Hypothesis #B — Inconsistent `dz` floor between `acoustic.py` and `advection.py` drives a runaway through the implicit tridiagonal

**File:line**:
- `dynamics/acoustic.py:82-91` — `_layer_thickness_m` returns raw `dz = (ph[1:]-ph[:-1])/g` whenever `dz > 0`, with NO floor; fallback `_flat_dz(grid)` is used only on non-positive `dz`
- `dynamics/advection.py:40-45` — `_dz_from_state` returns `jnp.maximum(raw, 1.0)` for positive `dz`, i.e., a hard 1 m floor
- `dynamics/acoustic.py:265-294` — `prepare_vertical_implicit_coefficients`:
  ```python
  beta = (REF_INV_DENSITY · REF_RHO_C2 · dt_sub² / dz_face)
  lower = -beta / dz_lower
  upper = -beta / dz_upper
  diag  = 1.0 + beta * (1.0/dz_lower + 1.0/dz_upper)
  ```
  scales as `~1/dz²`. If even one cell has `dz_face` collapse to (say) 0.1 m, that row's lower/upper grow by 10⁴×, the implicit diagonal still has `1.0 + 10⁴×β_ref` — but the off-diagonals dominate, conditioning collapses, and the Thomas solve (`tridiag.py`) returns a `w_interior` value whose recovery contains a few orders of cancellation.

**Mechanism (the "ignition spark")**:
1. ph drifts subtly under any one of the long-time accumulators (periodic-wrap horizontal pressure-gradient injection, no-divergence-damper acoustic mode growth, see Hypothesis #C). After ≈ 200-300 coupled steps, somewhere in the domain a single (k, j, i) cell sees `ph[k+1] - ph[k]` collapse from ~30 m to <1 m, transiently.
2. `advection.py:_dz_from_state` floors that cell at 1 m → advection stays defined. But `acoustic.py:_layer_thickness_m` uses 0.7 m at that cell.
3. The next acoustic substep's tridiagonal coefficients for that cell explode → `w_next` for that cell is large and incorrect.
4. `ph_next = state.ph + g·dt·w_next` then *increases* `ph[k+1]` by a huge amount → `dz` snaps back, **overshooting** to a large positive value.
5. The neighbouring cell now sees a discontinuous `ph` gradient → its acoustic substep over-responds.
6. Wave propagates horizontally via the periodic-wrap stencil (Hypothesis #A).

The 88.6 % step-firing rate is consistent with this once it starts — most cells in the domain are within an order of magnitude of the bad cell, and the next sanitize call catches the explosion at this step's level but not before the failure has propagated to thousands of cells.

**Why bug-hunts #1 and #2 missed it**: bug-hunt #1 flagged the missing `g` in ph update (real bug, fixed); bug-hunt #2 flagged missing buoyancy (which the role prompt asserts c1 fixed via tridiagonal — though I cannot find a buoyancy term in `_vertical_implicit_w` at `acoustic.py:297-312`; this is a separate concern, see §4). Neither audited the cross-module dz floor consistency.

**WRF citation**: `dyn_em/module_em.F` and `dyn_em/module_small_step_em.F` both use eta-coordinate `rdnw, rdn` precomputed once at init; layer thickness is bounded by the static eta partition, never derived per-step from a drifting ph.

**Minimal discriminator**:
Two parts.
(i) Add to `acoustic.py:_layer_thickness_m`: `return jnp.where(dz > 0.0, jnp.maximum(dz, 1.0), _flat_dz(grid))`.
(ii) Add a sanitize-adjacent diagnostic in `coupling/driver.py` that logs `jnp.min(_layer_thickness_m(state, grid))` every step; expect a monotonic-down trend to <10 m before the firing rate explodes.

**Likelihood**: **HIGH** as the proximate mechanism for *non-finite production* (vs Hypothesis #A which produces drift). The 793 M nonfinite values × 360 steps suggests an active source of Inf — and the tridiagonal Thomas solve at a degenerate `dz` is the only obvious one.

---

### Hypothesis #C — No 3D divergence damper + missing horizontal advection of `ph` (Klemp 2007 §3d omitted, WRF `rhs_ph` omitted)

**File:line**:
- `dynamics/acoustic.py:315-344` — `acoustic_once` does *not* implement Klemp 2007 §3d divergence damping (`p ← p + smdiv·(p − p_prev)`); the docstring at lines 1-9 explicitly cites §3a-c and excludes §3d.
- `dynamics/advection.py:411-420` — `compute_advection_tendencies` advects `u, v, w, theta, qv, p`. **`ph` is NOT advected.** `ph` is only updated by `ph += g·dt·w` inside the acoustic loop.
- WRF reference: `dyn_em/module_em.F:23,1292` — uses `rhs_ph` and `advect_ph_implicit` to advance geopotential **with horizontal advection and a "gw" term** every large step. The c1 dycore omits the horizontal advection of φ entirely.

**Mechanism**:
Two reinforcing accumulators.

**(C1) Acoustic-mode growth from missing §3d damper.** Forward-backward time integration of the 1-D linearized acoustic system `dp/dt = -ρc²∇·v, dv/dt = -1/ρ∇p` is *neutrally stable* in the L² sense — no growth, no decay. WRF adds `smdiv·(p − p_{n-1})` (`module_small_step_em.F:562`) to absorb the small numerical phase error and convert neutral stability to *strict* stability. Klemp 2007 §3d describes this damping for the implicit-vertical scheme. The c1 implementation has no such term. After ≈ 92 880 substep iterations per simulated hour, any 1-grid-point checkerboard mode (which IS excited by the periodic-wrap discontinuity from Hypothesis #A) grows without bound.

**(C2) Geopotential decoupled from horizontal flow.** In a real baroclinic atmosphere the geopotential `φ` has strong horizontal gradients (≈ 100 m height contour spread over ≈ 100 km). WRF advects φ horizontally (`rhs_ph`) every step. The c1 dycore evolves `ph` only by `ph += g·dt·w` accumulated across the 86 acoustic substeps. If the column at column-i has a small persistent positive `w` bias (which it does, due to (A) wrap pollution and (C1) acoustic energy buildup), `ph` rises monotonically at that column with no transport to neighbouring columns. Meanwhile boundary replay at the edges keeps `ph` pinned to the Gen2 PH+PHB. Interior ph drifts; edge ph is fixed. After ≈ 200 steps the interior-edge ph mismatch is large; the boundary apply then injects a discontinuous step at the boundary every coupled step, which the dycore reads as a strong horizontal pressure gradient and converts to spurious u, v.

**Why bug-hunts #1 and #2 missed it**: bug-hunt #1 flagged the missing `g` (a units bug) and asymmetric damping (a different kind of damping); bug-hunt #2's Hypothesis A was about *vertical buoyancy* in the w equation (a different equation). Neither audited the **complete absence of any horizontal ph transport** or the **omission of Klemp 2007 §3d**. Both are formulation gaps that bite specifically in long-time integration, exactly what the role prompt is asking about.

**WRF citation**:
- §3d damper: `dyn_em/module_small_step_em.F:557-565` (the `smdiv` loop), Klemp et al. 2007 Eq. 23.
- ph horizontal advection: `dyn_em/module_em.F:1292` (`advect_ph_implicit`) and `rhs_ph` in `module_big_step_utilities_em.F`; Skamarock 2008 ARW Tech Note Eq. 2.5.

**Minimal discriminator**:
(i) Add `+ smdiv·(p − p_prev)` per Klemp 2007 §3d with `smdiv ≈ 0.1` in `acoustic_once`. Carry `p_prev` in the scan carry.
(ii) Add `ph` to `compute_advection_tendencies` using the same mass-conservative scalar advection as `theta`/`qv`/`p_perturbation`. Note ph lives on w-faces (`nz+1`) so this needs a face-advection variant or a mass-face interpolation.

Either alone should slow the drift; both together should bring the 1 h probe to <10 % firing.

**Likelihood**: **MEDIUM-HIGH**. C1 is the simpler stability fix; C2 is the structural fix for the boundary-interior ph mismatch. Either is consistent with the "long-time only" symptom.

---

## 4. Honest uncertainty and what I could not verify

- I could not run any probes (READ-ONLY on `/tmp/wrf_gpu2_c1/`). The three hypotheses are mechanism analyses against the actual code at HEAD; each has a minimal discriminator that the next worker can run.
- The role prompt asserts "c1-A1: tridiagonal w-ph with buoyancy". Reading `acoustic.py:297-312` I see **no buoyancy term** (no theta dependence in `_vertical_implicit_w`'s RHS, no `θ'/θ_0` analogue of WRF's `t_2ave` at `module_small_step_em.F:1481-1486`). If buoyancy is actually missing, then bug-hunt #2 Hypothesis A is still unresolved and would dominate. I am **70 %** confident that buoyancy is still missing; **30 %** that I'm misreading the implicit operator. Recommend a quick test before the c1-A3 worker is dispatched: set up an Skamarock-Klemp warm-bubble test and confirm whether the bubble rises and oscillates.
- The acoustic substep uses constant reference α and ρc² (`acoustic.py:35-38`) instead of state-dependent values. This is a known c1 simplification per the sprint contract. At pressures ranging 1000-100000 Pa the actual ρc² spans ≈ 28 000 - 280 000 Pa; the constant `140 000 Pa` is wrong by up to 5× at column extremes. This is not the dominant bug but is a long-time bias that will not go away even after #A, #B, #C are fixed.
- The acoustic step lacks `mu`-coupling inside its substep (`acoustic.py:347-397`): μ is updated *after* the scan via `apply_mu_continuity_from_flux`. WRF updates μ inside the small-step loop (`module_small_step_em.F:1102-1108`). Bug-hunt #1 Hypothesis #3 flagged this; the c1 layout still has it. With 86 substeps the bias per large step is now small, but the structural decoupling remains.

I am **~75 %** confident the dominant accumulator is Hypothesis #A (periodic wrap on non-periodic d02) feeding the ignition spark in Hypothesis #B (dz floor inconsistency). I am **~50 %** confident Hypothesis #C is needed too — but #C is needed regardless to make the dycore stable to 24 h even after #A and #B are fixed.

---

## 5. Recommendation

**Do not escalate to c2 yet.** All three hypotheses are tractable inside c1, and each has a quick discriminator test. The 18-step-clean-but-1h-dirty signature *requires* a long-time accumulator that does not show up in short probes; only items in §3 here fit that requirement.

**Sequenced fix-hint to a c1-A3 worker**:

1. **First (cheapest, biggest expected leverage)** — fix Hypothesis #A. Replace `jnp.roll` in the four dycore stencils (`acoustic.py:_grad_x_to_u, _grad_y_to_v, _mass_to_u_face_2d, _mass_to_v_face_2d`; `advection.py:_periodic_flux5_faces, _mass_to_u_face, _mass_to_v_face`) with edge-mirror extrapolation (the c1-A2 pattern). Re-run the 1 h probe. **Expected**: `nonfinite_count` drops ≥ 10×, `fired_steps` drops ≥ 5×.

2. **Second** — fix Hypothesis #B. Add a 1 m `dz` floor in `acoustic.py:_layer_thickness_m`, matching `advection.py:_dz_from_state`. Cost: 1 line. Re-run.

3. **Third** — fix Hypothesis #C(C1). Add Klemp 2007 §3d `smdiv` divergence damping to `acoustic_once`: carry `p_prev` in the acoustic scan carry, add `smdiv·(p − p_prev)` correction (start with `smdiv = 0.1` per WRF default).

4. **Fourth** — fix Hypothesis #C(C2). Add `ph` to `compute_advection_tendencies` using mass-conservative flux form. Needs a face-collocation pass since ph lives on w-faces.

5. **Pre-flight check before #1–#4**: a 1 h **sanitize-disable** probe. If the dycore goes non-finite by step 50, that proves sanitize is just the symptom-catcher (clip mode is downstream of a real instability) — consistent with this report's analysis. If the dycore stays finite longer than the sanitize-on variant, that suggests sanitize itself is contributing (e.g., theta clipped to 150 K injects gravity wave at boundary). Either result is informative.

**If after #1–#4 the 1 h probe is still red**: escalate to user. The remaining candidates would be (a) buoyancy still missing in `_vertical_implicit_w` (bug-hunt #2 Hyp A re-opened), (b) reference α/ρc² instead of state-dependent — both require a non-trivial rewrite and are c2-scope territory.

**Per `AGENTS.md` operating rules**: no model code is changed in this report. The recommendation produces a c1-A3 sprint contract; the manager dispatches.

---

## 6. Files referenced (READ-ONLY)

- `/tmp/wrf_gpu2_c1/src/gpuwrf/coupling/driver.py:281-388, 692-854, 819-854`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/coupling/boundary_apply.py:31-160`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/coupling/physics_couplers.py:98-115, 250-282`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/dynamics/acoustic.py:35-91, 139-170, 218-294, 297-344, 359-397`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/dynamics/advection.py:19, 40-45, 218-229, 264-312, 406-420`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/dynamics/rk3.py:42-67`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/dynamics/step.py:19-51`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/contracts/halo.py:11-32`
- `/tmp/wrf_gpu2_c1/src/gpuwrf/contracts/grid.py:65-209`
- `/tmp/wrf_gpu2_c1/artifacts/m6x-fallback-c1/c1_a2_post_fixes_1h.json` (1 h FAIL probe)
- `/tmp/wrf_gpu2_c1/artifacts/m6x-fallback-c1/c1_a2_post_fix4_0p05h.json` (18-step clean probe)
- `/tmp/wrf_gpu2_c1/.agent/sprints/2026-05-22-m6x-c1-a2-advection-coupling-fix/worker-report.md`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:444-565, 1102-1108, 1481-1486, 1583`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F:14-29, 436, 1292`
- `/tmp/wrf_gpu2_m6x_bughunt3/.agent/sprints/2026-05-22-m6x-parallel-bughunt/bughunt-report.md` (bug-hunt #1)
- `/tmp/wrf_gpu2_m6x_bughunt3/.agent/sprints/2026-05-22-m6x-bughunt2-deeper/bughunt2-report.md` (bug-hunt #2)

No files in `/tmp/wrf_gpu2_c1/` or `/mnt/data/canairy_meteo/` were modified by this review.
