# Bug-hunt #4 META report (Opus 4.7, 2026-05-22)

Read-only audit of `/tmp/wrf_gpu2_c1/`. No operator-level fixes proposed; the
brief was to find what's outside the dycore operators that three prior
bug-hunts kept missing.

## 1. Meta-analysis: what all three bug-hunts had in common

Every previous hypothesis lived inside one file:

- #1 — `dynamics/acoustic.py:198` (PH g missing) and asymmetric damping
- #2 — buoyancy, diagnostic-p, ρ factor inside the acoustic operator
- #3 — periodic-wrap, dz floor, smdiv carry, ph advection — again all inside
  the dycore operator stack (`dynamics/acoustic.py`, `dynamics/advection.py`,
  `dynamics/rk3.py`)

All three reasoned as if the system were a single integrator and the bug had
to be in the integrator. But the M6-S2 runtime is **not a single integrator**;
it is a `dycore -> physics adapters -> lateral-boundary replay` pipeline
(`coupling/driver.py:808-816`, `coupling/driver.py:692-717`). Each leg writes
a different subset of leaves, with no explicit re-balancing step between
them. None of the prior bug-hunts audited this composition; they all assumed
the composition was consistent and only the individual operators could be
wrong.

The "step-45 nonfinite, sanitize-disabled" signature is curiously **stable
across every operator-level fix**. That is a strong tell: the cause is not in
any one operator, it is in something that runs every step and accumulates
independent of which operator stencil is used.

Three concrete suspect "outside the operator" pieces:

- `_candidate_before_boundary` (`coupling/driver.py:800-816`) feeds **zero
  base tendencies** into the dycore and then applies physics **after** RK3 as
  direct state replacement.
- `apply_lateral_boundaries` (`coupling/boundary_apply.py:31-46`) replays only
  six leaves — `u, v, theta, qv, ph, mu` — and leaves `p`, `pb`, `w` to drift.
- `required_n_acoustic` (`dynamics/acoustic.py:242-249`, `dynamics/rk3.py:49`)
  promotes `requested_n_acoustic=2` to `n_acoustic=86`, giving 92,880
  smdiv-damped substeps per coupled forecast hour.

Each of these three is a category that no prior bug-hunt opened. Below I list
them as the top three NEW hypotheses.

## 2. Top 3 NEW meta-level hypotheses

### H1 — split-physics architecture (physics-as-state-replacement)

**Where:** `coupling/driver.py:800-816`,
`coupling/driver.py:124` (`Tendencies.zeros(grid)` once, never updated),
`coupling/physics_couplers.py:229-365` (Thompson / MYNN / surface / RRTMG
adapters all `state.replace(...)` directly without touching `p`, `pb`, `ph`,
`mu`).

**Mechanism:**

1. Dycore is called with `tendencies = Tendencies.zeros(grid)` every step
   (`coupling/driver.py:124`, threaded through `run_forecast_segment`).
   Physics tendencies **never enter** the RK3 RHS at
   `dynamics/rk3.py:40-41`. WRF's split-explicit scheme depends on physics
   tendencies being baked into the RK3 stage, not applied after the fact.
2. After RK3, `thompson_adapter` modifies `theta, qv, qc, qr, qi, qs, qg,
   Ni, Nr` (`coupling/physics_couplers.py:182-196`); `mynn_adapter` modifies
   `u, v, w, theta, qv, qke` (`coupling/physics_couplers.py:274-281`);
   `rrtmg_adapter` modifies `theta` only
   (`coupling/physics_couplers.py:362-365`). **None of them touch
   `p`, `pb`, `ph`, or `mu`.**
3. The dycore stores `p = pb + p'` and treats geopotential `ph` as a
   prognostic surface (`dynamics/acoustic.py:358`,
   `dynamics/acoustic.py:104-118`). Physics jumps `theta` and `qv` without
   adjusting `p`/`ph`/`mu` → the column leaves hydrostatic balance every
   step. The next acoustic substep then sees a `p_perturbation =
   state.p - state.pb` that no longer matches the new `theta, ph` state and
   reacts by generating large `du/dt, dv/dt, dw/dt` and a runaway pressure
   correction in the vertical implicit Thomas solve
   (`dynamics/acoustic.py:307-322`).

**Why every operator fix has been silent:** the imbalance is injected *between
the operator and the next call to it*. No internal stencil change can repair
it — only the coupling order can.

**Discriminator (cheap):** in `_candidate_before_boundary` replace the
physics calls with no-ops and re-run the 1h coupled probe.

- If the run now stays finite past step 45 and the sanitize-firing rate
  collapses → H1 is the dominant accumulator.
- If step-45 still goes nonfinite with physics off → H1 is not it.

Cost: edit one `if False:` in `coupling/driver.py:811-815`, rerun one
forecast.

### H2 — lateral boundary forces only a 6-leaf subset, leaves `p`, `pb`, `w` drifting

**Where:** `coupling/boundary_apply.py:31-46`. Six leaves are written by
`apply_lateral_boundaries`; the State has 42 leaves (see
`contracts/state.py:180-224`). The dycore consumes `p` and `pb` directly in
the acoustic operator (`dynamics/acoustic.py:104-112`,
`dynamics/acoustic.py:342-358`).

**Mechanism:**

1. At t=0, IC sets `p = P + PB`, `pb = PB`, both finite and balanced
   (`coupling/driver.py:88-89`).
2. Every coupled step, the dycore writes a new `p` (acoustic substeps,
   `dynamics/acoustic.py:360`) and a new `ph` everywhere, **including
   boundary cells**.
3. `apply_lateral_boundaries` then overwrites `u, v, theta, qv, ph, mu` on
   the relax/spec zones from the replay file. `p` and `pb` on boundary cells
   are **left at whatever the dycore produced**, which has no consistency
   constraint with the externally-replayed `theta, ph, mu`.
4. The next dycore step reads `p_perturbation = p - pb` at the boundary and
   computes `_grad_x_to_u(p_star, grid)` and `_grad_y_to_v(p_star, grid)` in
   `dynamics/acoustic.py:346-351`. The boundary gradient is now inconsistent
   with the boundary `theta, qv, mu, ph` and pushes wind tendencies that the
   boundary spec then clobbers — but only after the gradient already
   propagated inward through the halo and the 5-point relaxation lap term
   (`coupling/boundary_apply.py:129-136`).

`w` is not boundary-forced at all; it cumulatively integrates whatever the
acoustic `_vertical_implicit_w` solve gives at the boundary and never gets
reset.

**Discriminator (cheap):** Pin `p[boundary] = p0[boundary]` and `w[boundary] =
0` after each `apply_lateral_boundaries` call by patching
`coupling/boundary_apply.py:45` to also overwrite `p` and `w` on the
spec-zone cells with the corresponding values from the initial state.

- If 1h probe sanitize rate drops markedly → H2 is the dominant accumulator,
  and the fix is to add `p_bdy`, `pb_bdy`, `w_bdy` to the replay schema.
- If no improvement → H2 contributes but is not dominant.

Cost: two extra lines in `boundary_apply.py` plus the initial-state freeze.

### H3 — `n_acoustic` promoted to 86 = O(10⁵) substeps/hour accumulator

**Where:** `dynamics/rk3.py:49` and `dynamics/acoustic.py:242-249`.

```
n_acoustic = max(int(n_acoustic), required_n_acoustic(grid, dt))
required_n_acoustic = ceil(c * dt / (cfl * min(dx, dy, dz_flat)))
```

With Gen2 d02 `dx=dy=3000 m`, dt=10 s, c≈322 m s⁻¹ (Klemp 2007 at
`T_ref=260 K`, `dynamics/acoustic.py:32-38`), and `_flat_dz = top_pressure /
nz`. For nz≈40 and top_pressure=5000 Pa (or similar small top), `_flat_dz` is
**tiny**, which is what gives n_acoustic=86. The role prompt records this
explicitly.

**Mechanism:**

1. `_flat_dz` is **a fallback** intended only for analytic unit states with
   zero geopotential (`dynamics/acoustic.py:86-90`). For real Gen2 states,
   `state.ph[k+1]-state.ph[k]` typically gives layer thickness of hundreds
   of metres, not metres. Yet `required_n_acoustic` uses `_flat_dz`, not the
   real `dz_min`.
2. If `_flat_dz` is a meaningless lower bound (e.g. `top_pressure / nz` =
   `5000 Pa / 40 ≈ 125 Pa`, dimensionally not a metre), the comparison
   `c*dt/(cfl*spacing)` is nonsensical and gives the **wrong** static
   substep count.
3. Two failure modes:
   - over-substepping: n=86 means dt_sub ≈ 0.116 s per RK3 stage, dt_sub
     ≈ 0.0388 s for the s2 stage `dt/2`. The vertical implicit
     `beta = rho_inv * rho_c2 * dt_sub² / dz_face` is then very small,
     forward Euler-like; each substep contributes O(noise) of the same
     magnitude regardless of total dt, so noise accumulates linearly with
     substep count.
   - The `smdiv` divergence damping at `dynamics/acoustic.py:356-357`
     applies `p_pert_next = p_pert_undamped + 0.1 * (p_pert_undamped -
     p_prev)`. Per substep this is a forward extrapolation that grows
     when `(p_pert_undamped - p_prev)` is nonzero. Over 86 × 3 = 258
     substeps per RK3 step times 360 dycore steps/hour = 92,880 substeps,
     a 1 part-per-million bias becomes a 9.3% drift per hour.

The "step-45 nonfinite is the same regardless of operator-level fixes"
signature is exactly what this mechanism produces — the noise injection is
per-substep, independent of which operator detail is patched.

**Discriminator (cheap):** Force `n_acoustic=4` (well above what physics
demands for the actual `dz_min`, but well below 86) by either patching
`rk3.py:49` to `n_acoustic = int(n_acoustic)` (no auto-promotion), or
calling `required_n_acoustic_for_state` (`dynamics/acoustic.py:252-272`,
which uses the real geopotential-derived `dz`) instead of
`required_n_acoustic`.

- If 1h probe sanitize rate drops markedly → H3 is the dominant accumulator.
- If sanitize rate is unchanged → H3 is not dominant.

Cost: one-line patch to `dynamics/rk3.py:49`, no operator code touched.
This is the **lowest-risk** discriminator and should run first.

## 3. Why these three were missed

- All three bug-hunts framed the system as "the dycore is wrong." But the
  five-iteration evidence shows the dycore is **bit-stable** for short
  trajectories (18-step probe clean) and only fails on long ones with
  physics + boundary forcing. The bug is therefore probably in the
  composition, not in any single operator. Bug-hunts kept opening
  `acoustic.py` and `advection.py` and never opened `driver.py` or
  `physics_couplers.py`.
- `Tendencies.zeros(grid)` being threaded through the entire pipeline
  unchanged is invisible if you only read the dycore — it shows up only in
  `build_initial_state` (`coupling/driver.py:124`) and in the dycore call
  signature (`coupling/driver.py:810`). Three bug-hunts cited the dycore
  call signature without noticing the input was permanently zero.
- `n_acoustic=86` was *visible* in role-prompt #4 but interpreted as "we
  picked the wrong CFL" rather than as a fundamental coupling/setup
  problem. `required_n_acoustic` uses `_flat_dz` which is documented at
  `dynamics/acoustic.py:86-90` as a fallback for analytic tests, **not** for
  Gen2 coupled runs. This is an initialization bug masquerading as an
  operator parameter.

## 4. Recommendation

Run the three discriminators in this order and report sanitize-rate +
step-of-first-nonfinite for each. They are independent and cheap.

1. **H3 first** (one-line patch, no coupling changes): force `n_acoustic=4`
   or call `required_n_acoustic_for_state`. If sanitize rate collapses, the
   c1 root cause is the `_flat_dz` fallback used in production, which is
   trivially fixable.
2. **H2 second** (two-line patch in `boundary_apply.py:45`): freeze `p,
   pb, w` at boundary cells. If sanitize rate collapses, c1 needs a
   boundary-replay schema extension to carry `p_bdy, pb_bdy, w_bdy`.
3. **H1 last** (one-line `if False:` around the four physics adapters in
   `_candidate_before_boundary`). If sanitize rate collapses with physics
   disabled, c1 needs to either (a) integrate physics as tendencies into
   RK3 RHS (proper split-explicit), or (b) add a hydrostatic-rebalance step
   after each adapter. (a) is the WRF-correct path; (b) is a c1-scoped
   stopgap.

Do **not** dispatch c1-A5 to "apply all three fixes at once" — that's the
same anti-pattern as bug-hunts #2 and #3. Run one discriminator, observe,
and only then patch.

If **none** of the three discriminators move the sanitize rate, the c1
architecture is the wrong approach and the user should be escalated to a
c2/pivot decision: at that point the bug is not localizable and the design
itself is the problem.

## 5. Honest uncertainty

I am moderately confident in H3 because the `_flat_dz` fallback is
documented as a test-only path and is being silently used in production —
that is the clearest setup/initialization error of the three. H1 and H2 are
structurally suspicious but the "step-45 every time" timing is more
consistent with a per-substep accumulator (H3) than with a per-step
coupling error (H1, H2). I cannot rule out that the dominant accumulator is
something I did not audit:

- the hybrid-eta coordinate coefficients `c1h/c2h/c3h/c4h` are absent from
  `GridSpec` (`contracts/grid.py:67-77`); Gen2 was likely run with the
  hybrid coordinate, c1 is using pure sigma. This is a model-formulation
  mismatch I have not quantified.
- RRTMG cadence: `rrtmg_adapter` applies `T_next = T + dt * heating_rate`
  (`coupling/physics_couplers.py:364`) but radiation runs every 60 steps,
  so heating is under-applied by 60× (`coupling/driver.py:34`,
  `coupling/driver.py:266-388`). This is a wrong climate, not a
  finiteness bug, and step-45 is before the first radiation call, so it
  cannot be the step-45 trigger.

The user should treat this report as "the three places nobody has looked,
prioritized by minimal-cost discriminator," not as "the bug is here." If
H3's discriminator does not help, that itself is high-value information.

## 6. Files cited

- `coupling/driver.py:34, 124, 266-388, 692-717, 800-816, 810-816`
- `coupling/boundary_apply.py:31-46, 129-136`
- `coupling/physics_couplers.py:182-196, 229-365, 274-281, 362-365`
- `contracts/state.py:180-224`
- `contracts/grid.py:67-77`
- `dynamics/rk3.py:40-41, 44-67, 49`
- `dynamics/acoustic.py:32-40, 86-90, 104-118, 242-272, 285-322, 342-360`
- `dynamics/tendencies.py:8-20`
- `dynamics/advection.py:436-451`
- `io/boundary_replay.py` (read; nothing here looks wrong)
- `io/gen2_accessor.py:213-241` (no hybrid coef export)

## 7. Output budget

Wall: ~50 minutes for audit + report. No operator code touched. Read-only
on c1.
