# c2-A2 + c2-A2.x Bundle Reviewer Report

Date: 2026-05-22
Reviewer: Claude Opus 4.7 (1M context, independent review)
Branch under review: `worker/codex/m6x-c2-A2x-vertical-acoustic` tip `ea7f89f Implement c2 vertical acoustic attempt`
Bundle: `52b97da` (c2-A2 horizontal PGF + mu-in-substep) and `ea7f89f` (c2-A2.x vertical acoustic + 2 contract fixes)
Specs: ADR-020 (`.agent/decisions/ADR-020-c2-dycore-architecture.md`), architecture step-back (`/tmp/wrf_gpu2_step_back_arch/worker-report.md` §4 pivot criteria)
WRF source anchor: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F`

## 1. Bottom line

**NEEDS-HYBRID-PIVOT** per architecture step-back §4 stop/go criterion #3 ("needs broad unreviewed state/contract changes, pivot immediately to the hybrid path and write an ADR before implementation").

The c2-A2 horizontal-PGF + mu-in-substep half (`52b97da`) is high quality and matches WRF `module_small_step_em.F:828-862,902-936` line-for-line; that piece is ACCEPT.

The c2-A2.x vertical-acoustic half (`ea7f89f`) is a heuristic warm-bubble simulator, not a port of WRF `advance_w`/`advance_mu_t`. It runs finite to 600 s — a real improvement over the c2-A2 step-1 NaN — but:

- The warm-bubble 600 s `w_max` target [5, 10] m/s is missed at 2.07 m/s (degrades from 8.68 at 300 s → 2.07 at 600 s).
- The vertical operator drops WRF's `t_2ave`, `ww`, `muave`, `ph_tend`, off-centering `epssm`, and all `_1` large-step fields. The implementation is non-WRF in structure, not just imprecise in coefficients.
- The acoustic scan carry is missing the WRF small-step scratch variables required to make the vertical path WRF-shaped. Adding them IS the "broad unreviewed state/contract changes" the arch step-back named as a hard pivot trigger.
- The 3 new unit tests are structural/qualitative only (sign checks, "did anything move?"). They pass with a wildly wrong vertical-acoustic implementation. There is **no analytic or savepoint oracle** for the vertical path. The arch step-back's second pivot criterion ("cannot produce a vertical-acoustic oracle") is also tripped.

Two of the three pivot criteria fire. The third (step-1 NaN) is closed. Per the rule, the manager should **write an ADR for the hybrid path** before c2-A2.y proceeds.

The two contract-shaped changes (`uncouple_horizontal_pgf_tendency`, mu sign flip) are individually defensible as WRF interpretations on a flat fixture, but each has a missing factor or convention gap that would BLOCK on a real Canary 3 km hybrid-eta grid (see R3, R4).

## 2. R-findings

### R1 — Vertical acoustic operator is not a port of `advance_w` — **BLOCKING**
**Location**: `src/gpuwrf/dynamics/acoustic_wrf.py:609-651` (`vertical_acoustic_update`), `:561-578` (`_vertical_buoyancy_acceleration`).
**Severity**: blocking for c2-A2.y; **this is the primary pivot trigger**.
**What is wrong**: The c2 implementation models the vertical implicit step as:
```
rhs = state.w + dt * (g * Δθ/θ_base)             # buoyancy only, on face
solve_tridiagonal(a, b, c, rhs)
ph_new = ph_old + dt * g * w_new
```
WRF `advance_w` (`module_small_step_em.F:1340-1489, 1533-1584`) is structurally different:
1. RHS for the phi sub-equation uses `dts*(ph_tend + .5*g*(1.-epssm)*w)` minus advected `ww*∂φ/∂η` (lines 1345, 1363-1380). c2 has neither `ph_tend` (large-step tendency) nor `ww` (vertical mass flux) nor the `(1-epssm)*ph_old` off-centering term.
2. The buoyancy term in `advance_w` is `dts*g*msft_inv*(rdn(k)*(c2a(k)*alt(k)*t_2ave(k) - c2a(k-1)*alt(k-1)*t_2ave(k-1)) - c1f(k)*muave)` (lines 1486-1489). It uses **coupled time-averaged theta `t_2ave`** weighted by `c2a*alt`, and subtracts `c1f*muave` (the mu-tendency divergence drag). c2 substitutes a naive `g*(θ - θ_base)/θ_base` form that has the wrong dimensions for the eta-coordinate momentum equation and no link to `c2a` or `muave`.
3. The vertical PGF term coupling w to ph (lines 1477-1485) carries explicit `(1.-epssm)*(ph(k+1)-ph(k))` and implicit `(1.+epssm)*(rhs(k+1)-rhs(k))` differences with `c2a*rdnw/((c1h*MUT+c2h)*(c1f*MUT+c2f))` weights — c2 puts **none of this in the RHS**; only the diagonal `b`-coefficient sees a fragment of the implicit weight.
4. WRF's post-solve ph update is `ph(k) = rhs(k) + msfty*.5*dts*g*(1+epssm)*w(k)/(c1f*muts+c2f)` (lines 1583-1584). c2 uses `ph += dt*g*w_next` — a different operator that ignores the `msfty/mu` weighting.

Net effect: the c2 vertical solve is a damped-buoyancy 1-D wave equation, not WRF's coupled (w, ph) implicit system. This is consistent with the worker's own unresolved-risks note: "Missing pieces include exact WRF coupled perturbation theta algebra, `ww` carry, and full `advance_w` RHS parity." The current "improvement" from c2-A2 → c2-A2.x (NaN → finite) comes from the tridiagonal damping, not from physical fidelity, which is why w_max degrades from 8.68 → 2.07 m/s between 300 s and 600 s instead of remaining in [5, 10] as WRF does.

**Suggested fix**: do **not** patch this incrementally. Either (a) commit to a WRF-shaped port that expands `AcousticScanCarry` to include `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `t_1/w_1/ph_1/u_1/v_1`, and `*_save` (and gates this behind an ADR per arch step-back §4), or (b) take the hybrid pivot: write an ADR replacing `vertical_acoustic_update` with a clean JAX vertical-implicit operator and drop the "WRF-identical" claim for the vertical path.

### R2 — Vertical theta transport uses `w` and `dz`, not WRF `ww` and `rdnw`; no horizontal theta transport — **BLOCKING**
**Location**: `src/gpuwrf/dynamics/acoustic_wrf.py:581-606` (`_vertical_theta_transport`).
**Severity**: blocking. The current form is dimensionally inconsistent with WRF's eta-coordinate scalar equation and has no horizontal counterpart.
**What is wrong**:
- c2 computes face theta perturbation by simple averaging (`0.5*(θ'(k)+θ'(k-1))`), multiplies by `w_next` (m/s geometric), and divides by `dz` (m).
- WRF `advance_mu_t` (`module_small_step_em.F:1148-1173`) builds `wdtn(k) = ww(k)*(fnm(k)*t_1(k) + fnp(k)*t_1(k-1))` using the **vertical mass flux `ww`** (pressure-coordinate units, computed from continuity at lines 1109-1114), the **`fnm/fnp` interpolation coefficients** (not `0.5/0.5`), and **`t_1`** (large-step theta, not the current substep θ). The update is `t -= dts*msfty*(... + rdnw(k)*(wdtn(k+1)-wdtn(k)))`.
- c2 has **no horizontal theta transport at all** inside the substep. WRF does this in the same block (`advance_mu_t:1162-1170`, the `.5*rdy` and `.5*rdx` divergences of `v*(t_1+t_1)` and `u*(t_1+t_1)`). Without it, a warm bubble that drifts horizontally would not advect θ horizontally.

**Suggested fix**: tied to R1. Cannot be fixed without expanding the carry and computing `ww` from continuity inside the substep.

### R3 — `_calc_coef_w` uses `cof/mu_total²` lumping that fails the hybrid-eta case — **BLOCKING for non-flat metrics, MINOR for warm-bubble**
**Location**: `src/gpuwrf/dynamics/acoustic_wrf.py:274-320` (`_calc_coef_w`).
**Severity**: blocking for any run on the Canary 3 km hybrid-eta grid; minor for warm-bubble.
**What is wrong**: WRF `calc_coef_w` (`module_small_step_em.F:624-650`) writes
```
cof  = (.5*dts*g*(1.+epssm))**2
a(k) = -cqw*cof*rdn(k)*rdnw(k-1)*c2a(k-1) / ((c1h(k-1)*MUT+c2h(k-1))*(c1f(k)*MUT+c2f(k)))
b(k) = 1 + cqw*cof*rdn(k)*( rdnw(k)*c2a(k)   / ((c1h(k)*MUT+c2h(k))*(c1f(k)*MUT+c2f(k)))
                          + rdnw(k-1)*c2a(k-1)/((c1h(k-1)*MUT+c2h(k-1))*(c1f(k)*MUT+c2f(k))) )
c(k) = -cqw*cof*rdn(k)*rdnw(k)*c2a(k)   / ((c1h(k)*MUT+c2h(k))*(c1f(k+1)*MUT+c2f(k+1)))
```
i.e. each entry has its **own** `(c1h(k)*MUT+c2h(k))*(c1f(k')*MUT+c2f(k'))` denominator with the appropriate hybrid coefficient at the appropriate vertical level.

c2 simplifies this to a single `cof = (.5*dt_sub*g/mu_total)²` lumped into the prefactor and then **drops all hybrid coefficients from the denominator** entirely:
```
cof = (.5*dt_sub*g/mu_total)**2
b(k) = 1 + cqw*cof*rdn(k)*(rdnw(k)*c2a(k) + rdnw(k-1)*c2a(k-1))
a(k) = -cqw*cof*rdn(k)*rdnw(k-1)*c2a(k-1)
c(k) = -cqw*cof*rdn(k)*rdnw(k)*c2a(k)
```
In the strict non-hybrid limit (`c1h=c1f=1`, `c2h=c2f=0`) this matches WRF (`MUT² = mu_total²`).

But `DycoreMetrics.flat()` (`src/gpuwrf/contracts/grid.py:178-185`) sets `c1h=eta_mass`, `c2h=0`, `c1f=eta`, `c2f=0` — not the non-hybrid limit. With η ∈ (0, 1), WRF's per-entry denominator becomes `c1h(k')*c1f(k'')*MUT²` which is **smaller** than `MUT²`, so WRF's |a|, b-1, |c| are **larger** than c2's by factor `1/(c1h*c1f)` (a factor of ~4 in mid-column and growing toward the top). c2 systematically **under-couples** the implicit solve relative to WRF on the flat fixture, and the mismatch is much worse on any real hybrid grid loaded from `wrfinput`.

Additional missing factor: `(1+epssm)²` (WRF off-centering). c2 hard-codes `epssm=0`.

**Suggested fix**: rewrite `_calc_coef_w` to emit the per-entry hybrid denominators per WRF lines 626, 632, 637-639, 646. Make `epssm` an `AcousticConfig` field.

### R4 — `uncouple_horizontal_pgf_tendency` is missing the `msfuy`/`msfvx` factor — **MINOR for warm-bubble, BLOCKING for sphere**
**Location**: `src/gpuwrf/dynamics/acoustic_wrf.py:478-505`.
**Severity**: minor for warm-bubble (msf=1 everywhere); blocking for Canary 3 km / real curvilinear grids.
**What is wrong**: the worker's WRF interpretation is **correct** — WRF's small-step `u_2` is coupled momentum `(c1h*muu+c2h)*u/msfuy` (`module_small_step_em.F:243`) and the small-step PGF tendency `dpxy` lives in those units (`:828, :862`). To convert WRF's `du_coupled/dt = -cqu*dpxy` to a velocity tendency on c2's uncoupled `State.u`, you need to multiply by `msfuy/(c1h*muu+c2h)`, i.e. divide by mass **and** multiply by `msfuy`. c2 only divides by mass.

The full chain:
- WRF `dpxy = (msfux/msfuy) * .5*rdx * (c1h*muu+c2h) * (3-term)` (line 828).
- WRF coupled-momentum tendency: `du_coupled/dt = -cqu * dpxy`.
- c2 uncoupled-velocity tendency must be: `du/dt = -cqu * msfuy * dpxy / (c1h*muu+c2h) = -cqu * msfux * (3-term)`.

After c2's "uncouple" division, the result is `-cqu * (msfux/msfuy) * (3-term)` — off by `1/msfuy` from WRF's intent. With msf≡1 on warm-bubble flat fixture this is a no-op, but on Canary 3 km this is a real bias growing with map factor distortion.

**Suggested fix**: multiply by `metrics.msfuy[None, :, :]` (and `metrics.msfvx[None, :, :]` for v) after the divide. Either inside `uncouple_horizontal_pgf_tendency` or by removing the `(msfux/msfuy)` factor from `horizontal_pressure_gradient` and re-multiplying by `msfux`/`msfvy` directly when applying to uncoupled velocities.

### R5 — Scan carry missing required WRF small-step scratch — **BLOCKING per arch step-back §4**
**Location**: `AcousticScanCarry` (`acoustic_wrf.py:43-87`); only carries `state, previous_pressure, al, alt, cqu, cqv`.
**Severity**: blocking. This is the third pivot trigger from arch step-back §4: "needs broad unreviewed state/contract changes, pivot immediately."
**What is missing**: to do `advance_w`/`advance_mu_t` faithfully the carry needs at minimum:
- `t_2ave` — time-averaged coupled theta on mass levels (used by buoyancy in `advance_w:1341-1344, 1487-1488`),
- `ww` — vertical mass flux on faces (computed in `advance_mu_t:1109-1114`, used by phi RHS and theta transport),
- `muave`, `muts` — averaged and ending column mass (used in buoyancy, ph update, and tridiag),
- `ph_tend` — large-step phi tendency (RHS of phi sub-equation, line 1345),
- `t_1`, `w_1`, `ph_1`, `u_1`, `v_1` — large-step states for off-centering and mu continuity (used at `:243, :274, :1095-1098, :1112, :1154, :1166`),
- `mu_save`, `t_save`, `w_save`, `ph_save`, `u_save`, `v_save` — large-step originals for `small_step_finish` reconstruction.

Adding any of these post-hoc changes the contract that c2-A2.y is being asked to build on. The arch step-back's pivot rule applies.

**Suggested fix**: do not patch piecemeal. Either ADR-021 (WRF-shape vertical port: expand carry, add `_1`/`_save` fields, document precision policy and lifecycle) or ADR-022 (hybrid pivot: keep horizontal PGF + mu continuity, replace vertical operator with a JAX IMEX scheme — narrower contract, larger numerics deviation).

### R6 — Mu continuity uses only the small-step `state.u/v`, not WRF's coupled-plus-large-step combined flux — **MINOR for warm-bubble**
**Location**: `acoustic_wrf.py:508-540` (`mu_continuity_tendency`).
**Severity**: minor for warm-bubble; blocking for general use.
**What is wrong**: WRF `advance_mu_t` (`module_small_step_em.F:1094-1098`) builds the divergence from a **sum** of two things: the in-substep coupled momentum perturbation `v(i,k,j+1)` plus the large-step coupled momentum `(c1h*muv+c2h)*v_1*msfvx_inv`. c2 uses only `state.u * (c1h*muu+c2h) / msfuy` (the small-step uncoupled value rebuilt into coupled units). The `v_1`, `u_1` parts are absent because they are not in the contract (see R5). For warm-bubble's rest large-step state (u_1=v_1=0) this is identical; for any real coupled run it is not.

The **sign** of the c2 form is OK: c2's `dnw` is positive (taken as `jnp.abs(eta(k+1)-eta(k))` at `grid.py:161`), so `dmu/dt = -∫div dη` is physical with the explicit `-` in c2's return statement. Worker's "sign fix" is legitimate under the c2 convention.

**Suggested fix**: documents/expands carry per R5; treats sign convention separately.

### R7 — Vertical-acoustic tests have no oracle — **BLOCKING per arch step-back §4**
**Location**: `tests/test_m6x_c2_acoustic.py:158-220` (the 3 tests added in c2-A2.x).
**Severity**: blocking. This is arch step-back §4 pivot trigger #2 ("cannot produce a vertical-acoustic oracle").
**What is wrong**:
- `test_vertical_w_coefficients_are_wrf_shaped_tridiagonal_entries` only checks shapes and **signs** of entries (a≤0, c≤0, b≥1, boundary zeros). It does **not** compare a single entry against the WRF formula. A `_calc_coef_w` implementation that drops `c2a` entirely or replaces `rdn` with `rdnw` would still pass.
- `test_vertical_acoustic_update_lifts_warm_theta_perturbation_and_updates_phi` only asserts `w_max > 0`, `|ph'| > 0`, `|Δθ| > 0`. Any operator that produces nonzero output passes — a w_max of 0.01 m/s or 100 m/s would both be green.
- `test_horizontal_pgf_tendency_uncouples_by_hybrid_face_mass` only checks that the **implementation's own formula** matches the **implementation's own formula** — `expected_u = coupled_u / (metrics.c1h[:, None, None] * 90000.0)` is the same algebra the production code does. This is not a WRF-independent oracle. The missing-msfuy bug from R4 is invisible to this test.

The warm-bubble 600 s `w_max` degradation (8.68 → 2.07 m/s) is observed only in the integration harness `scripts/m6_warm_bubble_test.py`, not in any unit test. There is no analytic oracle for either vertical-wave dispersion (e.g., 1-D linear gravity-wave column with known dispersion relation) or WRF savepoint comparison at the operator level.

**Suggested fix**: before c2-A2.y, add at least one of:
1. an analytic 1-D vertical column linear-acoustic mode test (e.g., a stratified atmosphere with prescribed `c_s`, verify the vertically propagating mode has expected period and decay),
2. a WRF savepoint test (warm-bubble or constant-N column) for `_calc_coef_w`, `vertical_acoustic_update`, and `_vertical_theta_transport` against captured WRF outputs.

### R8 — `_vertical_layer_thickness_m` uses `base_state.phb` only, not full `ph` — **MINOR**
**Location**: `acoustic_wrf.py:553-558`.
**Severity**: minor (low-amplitude bias).
**What is wrong**: c2 computes `dz = (phb(k+1)-phb(k))/g` using base geopotential only when `base_state` is provided, ignoring `ph_perturbation`. Once the bubble lifts, the actual layer thickness changes; c2's `dz` does not. WRF's vertical scalar transport uses `rdnw(k)` (1/dη), which is static, not a meter-based thickness. The c2 form is dimensionally consistent but physically frozen at the base state.

**Suggested fix**: tied to R2. The eta-coordinate form is `t -= dts*rdnw(k)*(wdtn(k+1)-wdtn(k))` (with `rdnw` from metrics and `wdtn` built from `ww` and `t_1`).

### R9 — `top_lid` flag is read in `x_face_pressure_dpn`/`y_face_pressure_dpn` but ignored in `_calc_coef_w` — **MINOR**
**Location**: `acoustic_wrf.py:274-320`; compare WRF `module_small_step_em.F:619-620, 626` where `lid_flag = 1 if top_lid=False else 0` and the top-face `a` entry is multiplied by `lid_flag`.
**Severity**: minor; warm-bubble uses `top_lid=False` so the path is correct.
**What is wrong**: c2 always uses the open-top form for the top-face coefficient. With `top_lid=True`, WRF zeros the top-face `a` entry and additionally sets `w(kde)=0` after the solve. c2 does neither.

**Suggested fix**: thread `top_lid` into `_calc_coef_w` and zero `w_next[nz]` when set.

### R10 — Defensive `jnp.maximum(jnp.abs(mass_x), 1e-12)` masks sign errors — **MINOR/STYLE**
**Location**: `acoustic_wrf.py:215, 244, 505, 540` (and `MIN_DZ_M`, `MIN_THETA_K`, `MIN_COLUMN_MASS_PA`, `MAX_COEF_PRESSURE_PA`).
**Severity**: minor.
**What is wrong**: `(c1h*muu+c2h)` is physically positive for any realistic state (column mass + nonneg hybrid). Wrapping it in `abs(...)` to "stay safe" can hide upstream sign errors and makes failure modes silent rather than loud. Several MIN_*/MAX_* clamps in the c2-A2.x patch (`MIN_COLUMN_MASS_PA=1`, `MAX_COEF_PRESSURE_PA=300000`) read as numerical-stability bandaids; they are reasonable for a debug run but should not survive a WRF-parity claim.

**Suggested fix**: post-pivot, remove the abs and clamps in production paths; add `chex.assert_positive` or equivalent in debug-mode (the M4+ `debug: bool` static-arg policy applies here).

## 3. Verdict per evaluation item

| # | Item | Verdict | Notes |
|---|---|---|---|
| 1 | Horizontal PGF (4 terms) | **ACCEPT** | Lines 320-374 of `acoustic_wrf.py` mirror WRF `module_small_step_em.F:828-862` (x) and `:902-936` (y) line-for-line: msf ratio prefactor, c1h*muu+c2h mass weight, three-term decomposition, dpn with cf1/cf2/cf3 bottom + fnm/fnp interior, M1 resolved as `-0.5*c1h*(mu_L+mu_R)`. Inherited from c2-A2 (`52b97da`), no regression in c2-A2.x. |
| 2 | `_calc_coef_w` tridiag coefficients | **R-finding-NEEDS-ADR** (R3) | Non-hybrid limit OK; hybrid case wrong; `epssm` missing. |
| 3 | `_advance_w`-like vertical w RHS | **R-finding-WRONG** (R1) | Missing `ph_tend`, `t_2ave`, `ww`, `muave`, off-centering, c2a*alt*t_2ave buoyancy form, c2a/(c1f*muts+c2f) ph update. Not a port. |
| 4 | Vertical theta transport | **R-finding-WRONG** (R2) | Uses `w`+`dz` not `ww`+`rdnw`; no horizontal theta term inside substep. |
| 5 | mu continuity in substep | **ACCEPT-WITH-MINOR** (R6) | Sign is physical for c2's positive-dnw convention; missing `u_1/v_1` large-step contribution but irrelevant for warm-bubble. |
| 6 | `uncouple_horizontal_pgf_tendency` | **R-finding-NEEDS-MSF-FACTOR** (R4) | Interpretation is legitimate WRF (cited line 243 of small_step_prep), but missing `msfuy` re-multiply factor. Works on flat fixture, biased on Canary 3 km. |
| 7 | mu sign fix | **ACCEPT** | Documented at `acoustic_wrf.py:517-523`; consistent with c2's `dnw>0` convention from `grid.py:161`. |
| 8 | Test coverage | **R-finding-NO-ORACLE** (R7) | 3 new tests are structural/qualitative; no analytic or savepoint oracle. The 600 s `w_max` degradation is invisible to unit suite. |
| 9 | Scan carry completeness | **R-finding-NEEDS-ADR** (R5) | Carry missing `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `t_1/w_1/ph_1/u_1/v_1`, `*_save`. Per arch step-back §4, this IS the "broad unreviewed contract change" pivot trigger. |
| 10 | Transfer audit | **ACCEPT** | No `host_callback`/`io_callback`/`pure_callback`/`.tolist()`/`device_get` in `acoustic_wrf.py`; static_kernel_check passed per worker proof. The scan body is XLA-resident. |

## 4. Stop/go for c2-A2.y

**HALT c2-A2.y now.** Mapping back to the arch step-back's three pivot criteria from §4:

1. *"first nonfinite at step 1"* — **CLOSED**. The bundle gets finite to 600 s. This was c2-A2.x's real contribution and it is real progress.
2. *"cannot produce a vertical-acoustic oracle"* — **TRIPPED** (R7). The unit tests do not validate against any WRF or analytic reference. The warm-bubble proof object shows the operator fails the actual target (`w_max` 2.07 m/s vs required [5, 10] at 600 s) and degrades from 300 s → 600 s.
3. *"needs broad unreviewed state/contract changes"* — **TRIPPED** (R1, R2, R5). Doing R1/R2 right requires `t_2ave`, `ww`, `muave`, `ph_tend`, and the `_1`/`_save` field families in the carry. That is a contract expansion the c2-A2 sprint contract did not authorize and the architecture step-back named explicitly.

Per the arch step-back rule, the manager must "pivot immediately to the hybrid path and write an ADR before implementation." c2-A2.y is currently building on the c2-A2.x foundation; continuing risks baking the R1/R2/R3/R5 deficiencies deeper and making the eventual hybrid pivot more expensive.

Concretely, the manager should:

A. **Halt c2-A2.y** and tell the worker not to commit further work on `worker/codex/m6x-c2-A2y-wrf-smallstep-parity` until the ADR is in.
B. **Choose between**:
   - **ADR-021 "WRF-shape vertical port"**: expand `AcousticScanCarry` to include the WRF small-step scratch, add `_1`/`_save` field families to `State` or to a new `SmallStepState`, port `advance_w`/`advance_mu_t` faithfully, add a 1-D analytic-oracle vertical test, and treat warm-bubble [5, 10] as the gate. Probably 2–3 focused sprints. Higher RMSE-compat odds.
   - **ADR-022 "Hybrid pivot"**: keep c2-A2 horizontal PGF + mu continuity as-is, replace `vertical_acoustic_update`/`_calc_coef_w`/`_vertical_theta_transport` with a clean JAX IMEX vertical operator (e.g., Dinosaur-style time integration on the vertical column, see arch step-back §3 "Hybrid"), document loss of WRF-identical numerics in Tier-1 but keep Tier-4 RMSE binding. Fewer sprints, larger numerics deviation, smaller new-contract surface (no `t_2ave`/`ww`/`_1`/`_save` needed).
C. **Either way, before c2-A2.y resumes**: add the analytic vertical-acoustic oracle test (R7).

The two contract-shaped changes from c2-A2.x (`uncouple_horizontal_pgf_tendency`, mu sign) can survive either pivot — they belong to the horizontal-PGF + mu-continuity path that is sound. They need R4's msf-factor fix but are not architecturally fatal.

## 5. Open questions for the manager

1. **Which pivot direction?** ADR-021 (full WRF-shape port, larger carry expansion) vs ADR-022 (hybrid: keep horizontal, replace vertical). The arch step-back's open question §6 ("Do we have a WRF savepoint exposing enough `advance_w`/`advance_mu_t` intermediates?") is now load-bearing: ADR-021 needs that savepoint to be anti-tautological; ADR-022 doesn't.

2. **Is the c2-A2 horizontal PGF + mu continuity worth landing on main as a partial milestone**, or does the manager prefer to keep it on the worker branch until the vertical path lands? The horizontal half is solid and addresses the c2-A1 review's R-findings; it would not block.

3. **What is the acceptable scope of `State`/`AcousticScanCarry` expansion?** Per arch step-back §4 the contract change requires an ADR; the manager should clarify whether `_1`/`_save` families belong as separate carry leaves, as a separate `SmallStepState` pytree, or as a transient inside `acoustic_substep_carry`. This decision shapes both candidates.

4. **Should the warm-bubble target [5, 10] m/s remain binding** as the c2 vertical-acoustic gate, or is the manager open to a parallel "analytic 1-D linear acoustic column" gate as the operator-level oracle (with warm-bubble downgraded to an integration smoke until coupled physics lands)? This affects whether R7 can be closed before the next sprint.

5. **Map-factor coverage in proof harness.** Both R4 (uncouple) and R3 (calc_coef_w hybrid) are flat-fixture-invisible. Does the manager want the c2-A2.y sprint to include a Canary 3 km `wrfinput_d02` smoke as a gate, so these biases surface before they cost a 1 h coupled sprint?

---

### Verifiability triple

| Proof object | Status | Method |
|---|---|---|
| `proofs/phase1_phase2_unit_tests.json` (c2-A2) | inspected | AC1-AC4 unit gate PASS; matches `acoustic_wrf.py:293-408` horizontal PGF |
| `proofs/warm_bubble_600s.json` (c2-A2) | inspected | step-1 nonfinite; documented in c2-A2 worker report — CLOSED by c2-A2.x |
| `proofs/unit_and_static_checks.json` (c2-A2.x) | inspected | 33 passed; static_kernel_check no findings — VERIFIED for transfer audit only |
| `proofs/warm_bubble_600s.json` (c2-A2.x) | inspected | 300s w_max=8.68, centroid=3288m; 600s w_max=2.07, centroid=3387m; verdict `FAIL_TARGETS_NOT_MET` — finite improvement, target miss |
| WRF source citations | re-checked | Each cited WRF line was opened in `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` (calc_coef_w at 570-652, advance_uv at 654-967, advance_mu_t at 969-1175, advance_w at 1178-1597) and compared to `acoustic_wrf.py` at the equivalent function. |
| Architecture step-back pivot criteria | re-checked | `/tmp/wrf_gpu2_step_back_arch/worker-report.md` §4 read; 2 of 3 criteria found tripped by this bundle (oracle missing; broad contract changes needed). |
