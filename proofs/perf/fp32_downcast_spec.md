# FP32 Downcast — Implementation-Ready Spec

Branch: `worker/opus/fp32-spec` (from `manager-2026-05-23` @ `d260e96`).
Status: **READ-ONLY design/audit. No src code changed. No GPU run.** This is the
mechanical implementation spec for the gated fp32-downcast sprint that follows.

Supersedes the prose in `proofs/perf/fp32_downcast_plan.md` (which it cites) by
making every touch-point a `file:line` with the minimal (not-applied) diff shape,
the casting hazards with their guards, and the binding validation artifacts.

Authoritative policy = `.agent/decisions/ADR-007-precision-policy.md` (authorization
matrix). The frozen storage matrix is `src/gpuwrf/contracts/precision.py::PRECISION_MATRIX`.

---

## 0. One-paragraph mechanism summary (read this first)

The model is **already a clean mixed-precision design**; the operational real-case
path is merely *pinned* to fp64 by one flag. `State.zeros` allocates every leaf at
its `PRECISION_MATRIX` dtype (`contracts/state.py:32`), so the resident real-case
state is *already* mixed (u/v/theta/qv fp32, p/ph/mu/w fp64) the moment it is built
by `d02_replay.build_replay_case` (`State.zeros` at `d02_replay.py:747`, then
`state.replace(u=…,theta=…,qv=…)` with default `_cast=True` re-pins each loaded WRF
array to its matrix dtype at `d02_replay.py:790`). `daily_pipeline._build_real_case`
then sets `force_fp64=True` (`daily_pipeline.py:204`), and
`_enforce_operational_precision(state, force_fp64=True)` upcasts **every** field to
fp64 via `state.replace(_cast=False, …)` (`operational_mode.py:319-336`) both at the
public entry (`operational_mode.py:1818,1763`) and once per step inside the scan
(`operational_mode.py:1542`). **The fp32 work is: flip that one flag** so
`_enforce_operational_precision` lands each field at `DEFAULT_DTYPES.dtype_for(field)`
(the fp32 matrix) instead. The fp64-LOCK islands (acoustic small-step coefficients,
the implicit w/φ tridiagonal solve, pressure/EOS, mass continuity) are *internally*
fp64-robust where it matters most (`calc_coef_w` force-casts `mut→fp64` at
`acoustic_wrf.py:636`; `diagnose_pressure_al_alt` builds its `dpn` pressure buffers
at `state.p_perturbation.dtype` = fp64 at `acoustic_wrf.py:304,334`), and the physics
adapters already follow the *live* state dtype via `_output_dtype` (`physics_couplers.py:352-376`).
The realistic remaining work is **boundary hardening + an HLO/dtype proof harness**,
not new precision logic.

---

## 1. The FP64-LOCK set (MUST stay fp64, regardless of the switch)

These operators/fields are catastrophic-cancellation-, conditioning-, or
mass-conservation-sensitive. The lock is enforced **inside the operator** (so an fp32
input is force-upcast at the boundary), NOT merely by the storage matrix. Per-item:
where the lock is currently robust vs. where the implementer must *add* a force-upcast.

### 1.1 Acoustic small-step coefficient construction — `calc_coef_w`
`src/gpuwrf/dynamics/acoustic_wrf.py:606-690` (`calc_coef_w`-equivalent;
`mut = jnp.asarray(mut, dtype=jnp.float64)` at `:636`, then `cqw/c2a/a/alpha/gamma`
all `dtype=mut.dtype` at `:640-652`, `cof=(0.5*dt*g*(1+epssm))**2` at `:648`).
- **Why fp32-unsafe:** the substep restoring coefficient `cof ∝ (dt·g)²` and the
  tridiagonal entries `±2·cof·rdnw²·c2a/(mass_h·mass_f)` (`:661`) span many orders of
  magnitude across the column; fp32 (~7 decimal digits) loses the small off-diagonal
  relative to the diagonal, mis-conditioning the implicit solve and amplifying the
  acoustic mode over 10 substeps × 3 RK stages × 8640 steps.
- **Status:** **already fp64-robust** (force-cast at `:636`). **No change.** Guard G-L1.

### 1.2 Implicit vertical w/φ tridiagonal solve — `advance_w_wrf`
`src/gpuwrf/dynamics/core/advance_w.py:131-440` (Thomas-style solve; `rw` buffer at
`p.dtype` `:116`; `safe_mass_*` guards `:229,233,266,313,428`; `pi=π@w_solved.dtype` `:414`).
- **Why fp32-unsafe:** a tridiagonal back-substitution accumulates the column; small
  pivots near the rigid lid / damping layer make the recurrence ill-conditioned. fp32
  rounding in the forward sweep propagates into every level below — exactly the
  historical "spurious top-face w" failure mode (`daily_pipeline.py:182-192`).
- **Status:** buffers follow `p.dtype` (= `state.p_perturbation` = fp64) — robust on the
  **production** path because `p` is in the LOCK set. **HARDEN:** add an explicit
  `p = p.astype(jnp.float64)` at the head of `advance_w_wrf` so the solve cannot be
  silently fp32 if a future caller passes an fp32 `p`. Guard G-L2.

### 1.3 Pressure / EOS diagnostics — `diagnose_pressure_al_alt`, `calc_p_rho_*`
`src/gpuwrf/dynamics/acoustic_wrf.py:232-345` (`diagnose_pressure_al_alt`: `ph_pert =
state.ph_perturbation.astype(alb.dtype)` `:263`; `dpn` buffers at
`pressure_perturbation.dtype` `:304,334`; `_safe_pressure/_safe_alt` clamps `:125,131`).
`src/gpuwrf/dynamics/core/calc_p_rho.py:79-160` (`safe_mass`/`theta_total_ref` guards
`:79,89`). EOS inverse density `_inverse_density_from_theta_pressure`
(`acoustic_wrf.py:134`).
- **Why fp32-unsafe (hydrostatic cancellation):** the interior PGF subtracts
  nearly-equal adjacent-level pressures, `Δp/p ~ 1e-3…1e-5` in the lower troposphere
  (ADR-007 §"Pressure-gradient accumulation"). fp32 leaves ~2 significant digits after
  the subtraction — operationally insufficient for momentum balance; fp64 leaves ~10.
- **Status:** pressure buffers follow `state.p_perturbation.dtype`/`state.ph_perturbation.dtype`,
  which are LOCK-set fp64 — robust on production. But `calc_p_rho.py` and the EOS
  follow the dtype of their `theta`/`p` *arguments*. On the production prep-path these
  are explicitly upcast by the caller (e.g. `_acoustic_core_state` upcasts theta/p to
  fp64 at `operational_mode.py:609,622`), but `_acoustic_core_state_from_prep` passes
  `prep.theta_work`/`prep.alt` derived from fp32 `state.theta` (see §4 H-3).
  **HARDEN:** force-cast theta and pressure to fp64 at the entry of `calc_p_rho_step`
  and `_inverse_density_from_theta_pressure`. Guard G-L3.

### 1.4 Geopotential (φ / ph) update + φ-RHS — `rhs_ph_wrf`, the φ half of `advance_w_wrf`
`src/gpuwrf/dynamics/acoustic_wrf.py` `rhs_ph_wrf` (called `operational_mode.py:769-789`
from `state.u,state.v,state.w,carry.ww,state.ph_perturbation`); φ accumulation in
`advance_w_wrf`.
- **Why fp32-unsafe:** φ = g·z is O(10⁵ m²/s²) and the perturbation φ′ that drives
  buoyancy is O(1–10²); storing/accumulating φ in fp32 quantizes φ′ below the buoyancy
  signal (same cancellation class as pressure). `ph/ph_total/ph_perturbation` are
  LOCK-set (`PRECISION_MATRIX` `precision.py:83-85`).
- **Status:** storage robust (matrix fp64). **HARDEN:** `rhs_ph_wrf` reads `state.u`,
  `state.v`, `state.w` (the first two go fp32 in perf mode). Force-upcast its u/v/w
  arguments to fp64 at the call site (`operational_mode.py:770-775`) OR inside
  `rhs_ph_wrf`. Guard G-L4.

### 1.5 Column dry-mass continuity — `mu`, `advance_mu_t`, `_limit_guarded_mass_state`
`mu/mu_total/mu_perturbation` LOCK-set (`precision.py:79,86,87`). `_limit_guarded_mass_state`
(`operational_mode.py:507-517`) and the small-step `advance_mu_t` mass average.
- **Why fp32-unsafe:** the dry-mass residual must stay ≤ 1e-10 fractional; mu is O(10⁴–10⁵)
  Pa and the per-step Δmu is O(10⁻²–10⁰), so fp32 swamps the increment (relative
  resolution ~1e-3·mu ≫ Δmu).
- **Status:** storage robust (matrix fp64); `_limit_guarded_mass_state` uses
  `jnp.asarray(origin.mu_*)` and a `1.0` floor at `mu_perturbation.dtype` (`:515`) — fp64
  because mu is fp64. **No change** provided mu stays LOCK. Guard G-L5.

### 1.6 Surface-layer stability/flux handles + precipitation accumulators
LOCK-set: `ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv, t_skin,
soil_moisture, roughness_m` and `rain_acc, snow_acc, graupel_acc, ice_acc`
(`precision.py:107-119,122-125`).
- **Why fp32-unsafe:** Monin–Obukhov is iterative and ill-conditioned near neutral
  stability; `*_acc` are monotone 24–72 h accumulators where fp32 loses the small
  per-step increment (swamping). The surface adapter already does its physics in fp64
  and the accumulators add in fp64 then store at the LOCK dtype
  (`physics_couplers.py:729-735,654-657`).
- **Status:** robust (matrix fp64 + fp64 add). **No change.** Guard G-L6.

### 1.7 FP64 boundary leaves
`w_bdy, p_bdy, pb_bdy, ph_bdy, phb_bdy, mu_bdy, mub_bdy` (`precision.py:129-137`) — feed
the LOCK fields; stay fp64 so the lateral-boundary blend (`boundary_apply.py:137`,
`.astype(boundary.dtype)`) keeps the LOCK fields fp64.

**LOCK invariant (binding):** after the switch flips to fp32, every field listed in
§1.1–1.7 must still report `dtype == float64` at the end of one step AND the compiled
HLO of the acoustic/pressure/mass region must contain **no** `convert(f64->f32)` on
these leaves inside the loop body (see §5 proof P-2).

---

## 2. The FP32-OK set (downcast when the switch is flipped)

Storage of non-acoustic prognostics + their non-acoustic arithmetic + physics outputs
+ diagnostics. These are `FP32_GATED` in `PRECISION_MATRIX` (`precision.py`):

| Field | Matrix line | What goes fp32 | What stays fp64 (boundary) |
|---|---|---|---|
| `u`, `v` | `precision.py:89,90` | resident storage between steps; non-acoustic advection/diffusion arithmetic | upcast on entry to the acoustic PGF / w-solve / φ-RHS islands (§1.2-1.4) |
| `theta` | `precision.py:92` | storage + scalar advection + physics tendency | the theta increment limiter already does its mass math in fp64 (`operational_mode.py:454-496`, `candidate64/mass64`, casts output back at `:494`) |
| `qv` | `precision.py:93` | storage + scalar advection + Thompson/MYNN tendency | fp64 inside Thompson's water budget (adapter upcasts) |
| `qc,qr,qi,qs,qg,Ni,Nr,Ns,Ng` | `precision.py:95-103` | already fp32 storage; Thompson source/sink output | fp64 inside the microphysics solver, downcast on write (`physics_couplers.py:639-648`) |
| `qke` | `precision.py:105` | MYNN TKE storage | — |
| `xland, lakemask, mavail` | `precision.py:116-118` | static land masks | — |
| `u_bdy,v_bdy,theta_bdy,qv_bdy` | `precision.py:127-131` | fp32 boundary leaves feeding fp32 fields | — |

**Non-acoustic arithmetic already fp32-safe by construction:**
- Flux-form scalar/momentum advection scratch buffers use `jnp.result_type(...)` for
  every scatter target (`flux_advection.py:159,219-220,401-402,435-436,498-499,532`) —
  so they are fp32 when the advected field is fp32 and fp64 when it is fp64 (identity on
  the force_fp64 path). **This is prior bug class #3, already fixed.** No change.
- Physics adapters write at the **live** state dtype via `_output_dtype` (returns
  `getattr(state, field).dtype`, `physics_couplers.py:352-376`): fp32 in perf mode,
  fp64 under force_fp64. Internal physics math is upcast to fp64 at the column boundary
  (`physics_couplers.py:579,729-735`) and downcast on write (`:639-648,778-779`).
  **This is prior bug class #5, already fixed.** No change.

**Diagnostics** (`_m9_snapshot`, `comprehensive_harness`) already compute deltas in fp64
(`comprehensive_harness.py:238`) regardless of storage dtype — no change.

---

## 3. The mechanism — exactly how to flip the switch (minimal diff shapes, NOT applied)

The downcast is governed by **one boolean** already plumbed end-to-end as
`OperationalNamelist.force_fp64` (declared `operational_mode.py:118`, in the pytree aux
`:230,264,295`, honored in `_enforce_operational_precision` `:319-336`, applied at the
public entries `:1542,1763,1818,1910,1958,1998,2058`). **No new dtype logic is needed.**

### 3.1 The single production touch-point — flip the real-case default
`src/gpuwrf/integration/daily_pipeline.py:204`
```
-        force_fp64=True,
+        force_fp64=False,   # fp32 operational precision policy (ADR-007 perf mode)
```
Effect: `_enforce_operational_precision(force_fp64=False)` takes the else-branch
(`operational_mode.py:332-336`), casting each field to `DEFAULT_DTYPES.dtype_for(field)`
with default `_cast` semantics — landing the FP32-OK set at fp32 and the LOCK set at
fp64. Because `State.zeros` already allocated the matrix dtypes (`state.py:32`) and
`d02_replay` re-pinned loaded arrays to the matrix via `_cast=True` (`d02_replay.py:790`),
the initial carry is already mixed; the only thing `force_fp64=True` was doing was
*overriding* that with a blanket upcast.

### 3.2 Make the switch a config knob (so gates can run both paths without editing src)
`daily_pipeline.DailyPipelineConfig` does not currently expose `force_fp64`. Add a field
(default `False` once §3.1 lands, or default `True` during the gate transition) and
thread it into the `OperationalNamelist.from_grid(... force_fp64=config.force_fp64 ...)`
call (`daily_pipeline.py:195-213`). Minimal shape:
```
# DailyPipelineConfig dataclass:
+    force_fp64: bool = False
# _build_real_case:
-        force_fp64=True,
+        force_fp64=bool(config.force_fp64),
```
This lets the §5 validation harness instantiate the **same** real case at fp64 and fp32
for a paired comparison with zero src edits between runs.

### 3.3 Idealized parallel-fp32 variant (gate 2a only)
The idealized builders hard-code `force_fp64=True` (`idealized.py:577`, carry build
`:619`). The warm-bubble/Straka gate must run a **parallel** fp32 variant. Add a
`force_fp64: bool = True` parameter to `_build_setup` (`idealized.py:560`) and
`_initial_carry` (`idealized.py:616-619`), default `True` so the existing fp64 close
gate is byte-unchanged; the fp32 gate passes `force_fp64=False`. Minimal shape:
```
-def _build_setup(case, *, require_gpu: bool = True) -> IdealizedSetup:
+def _build_setup(case, *, require_gpu: bool = True, force_fp64: bool = True) -> IdealizedSetup:
 ...
-        force_fp64=True,
+        force_fp64=force_fp64,
```

### 3.4 Mixed-dtype boundary contract (the islands)
The fp32 fields feed three fp64 islands. The boundary handling required:

| Boundary | Direction | Mechanism | Status |
|---|---|---|---|
| u/v/theta → acoustic legacy core | fp32 → fp64 | `_acoustic_core_state` already `.astype(jnp.float64)` on theta/p/ph/alt (`operational_mode.py:609-624`) but **NOT on `u=state.u`,`v=state.v`,`w=state.w`** (`:633,636,637`) | **HARDEN** — see H-2 |
| u/v/theta → acoustic prep core | fp32 → fp64 | `_acoustic_core_state_from_prep` passes `prep.u_work/v_work/theta_work` (fp32-derived) un-upcast (`operational_mode.py:793-808`); `rhs_ph_wrf`/`pg_buoy_w` read `state.u/v/w` fp32 (`:741-775`) | **HARDEN** — see H-3, G-L4 |
| `calc_coef_w` / `advance_w` | fp32 → fp64 | internal force-cast (`acoustic_wrf.py:636`) | robust (legacy + prep) |
| fp64 island result → fp32 storage | fp64 → fp32 | end-of-step `_enforce_operational_precision(force_fp64=False)` re-pins u/v/theta to matrix fp32 (`operational_mode.py:1542`); `State.replace` default `_cast=True` also down-pins | robust |

**The clean rule to implement:** make the acoustic island a *hard* fp64 boundary by
adding `.astype(jnp.float64)` to the u/v/w (and any theta-derived) leaves that enter
`_acoustic_core_state` / `_acoustic_core_state_from_prep` and the `rhs_ph_wrf`/`pg_buoy_w`
call sites — so the island is fp64-robust **independent** of the resident storage dtype,
exactly as theta/p/ph already are. This is the F7-correctness-preserving way to flip
the switch: the dynamics math is bit-for-bit the force_fp64 dynamics; only the *resident
storage between RK steps* and the *physics column arithmetic* drop to fp32.

---

## 4. Casting hazards — every place a silent cast can (a) defeat fp32 or (b) contaminate a LOCK op

Numbered H-1…H-7 with the concrete guard each implies. (a)=no speedup, (b)=instability.

- **H-1 (a) — x64-at-import promotes python-float literals.** `jax_enable_x64=True` is
  set at import (`contracts/state.py:17`, `runtime/operational_mode.py:69`). With x64 on,
  fp32 arrays are NOT auto-promoted, **but any bare `jnp.asarray(c)` / python float in an
  fp32 arithmetic path becomes fp64 and promotes the whole expression to fp64** — silent
  loss of the speedup while storage still "looks" fp32. Spot-checks of the fp32 paths show
  literals are already dtype-anchored (`flux_advection.py` uses `result_type`; physics
  uses `_output_dtype`; advection clamps use `dtype=field.dtype`). **Guard G-H1:** the §5
  HLO audit must assert the gated-field fusions are fp32 (count `convert(f32->f64)` inside
  the loop on FP32-OK leaves; expect 0 except at the named island boundaries).

- **H-2 (b) — legacy `_acoustic_core_state` u/v/w not upcast.** `operational_mode.py:633,636,637`
  pass `u=state.u, v=state.v, w=state.w` at storage dtype (fp32 in perf mode) into the
  acoustic core, while theta/p/ph ARE upcast (`:609,622`). The horizontal-momentum acoustic
  update would then run fp32 against fp64 pressure → mixed precision in the PGF. **Guard
  G-H2:** add `state.u.astype(jnp.float64)` etc. at `:633-637`. (Production currently uses
  the prep path, but the legacy path is reachable and must not be a latent trap.)

- **H-3 (b) — prep-path `u_work/v_work/theta_work` are fp32-derived.** `small_step_prep_wrf`
  computes `u_work`/`theta_work` from `state.u`/`state.theta` (`small_step_prep.py:213,215`)
  and `theta_offset` at `state.theta.dtype` (`:169`); these flow into the acoustic core
  un-upcast (`operational_mode.py:793,795,808`), and `rhs_ph_wrf`/`pg_buoy_w` read
  `state.u/v/w` directly (`:741-775`). **Guard G-H3:** either (i) force-cast `state`→fp64
  for the dynamics-relevant leaves at the top of `small_step_prep_wrf` and at the
  `rhs_ph_wrf`/`pg_buoy_w_dry` call sites, or (ii) compute `small_step_prep_wrf` against an
  fp64 view of the state. Option (i) is the smaller diff and matches G-L3/G-L4.

- **H-4 (b) — `State.replace(_cast=True)` re-pins to *current* resident dtype.**
  `state.py:586-587` casts each updated value to the field's current dtype. This is
  *desirable* for FP32-OK fields (keeps them fp32) but means: if a LOCK field's resident
  dtype were ever fp32, an fp64 island result written via `replace` would be silently
  truncated. **Guard G-H4:** the LOCK invariant (§1) + a one-step assertion that every
  LOCK field is fp64 after `_enforce_operational_precision(force_fp64=False)` — this is
  exactly prior bug class #2, and the assertion is the regression sentinel.

- **H-5 (a/b) — `_enforce_operational_precision` must actually land the matrix dtypes.**
  The else-branch (`operational_mode.py:332-336`) uses default `_cast=True` `replace`. If
  the incoming field is already at the matrix dtype this is a no-op (good); if a future
  refactor changes the matrix it follows automatically. **Guard G-H5:** assert
  `final_state.u.dtype==float32 and final_state.theta.dtype==float32 and
  final_state.p.dtype==float64 and final_state.mu.dtype==float64 and final_state.w.dtype==float64`
  at the public entry, once, after enforcement.

- **H-6 (b) — scan carry dtype stability.** `jax.lax.scan` requires the carry pytree dtype
  to be invariant across iterations. The carry's non-state scratch is already
  dtype-pinned: `t_2ave`/`ph_tend` forced fp64 (`operational_state.py:103,112`), `rthraten`
  `zeros_like(state.theta)` (`:132`, fp32 in perf mode — consistent with theta). The
  `*_save` family is `jnp.asarray(state.x)` (`:122-128`) inheriting the resident dtype, and
  `_with_save_family` rebuilds them from `state` (`operational_mode.py:566-573`). **Guard
  G-H6:** after the switch, run one `jax.eval_shape`/lower of `run_forecast_operational` and
  confirm no carry-dtype-mismatch error and that each carry leaf's dtype is stable
  (theta-derived saves fp32, w/ph/mu saves fp64). The acoustic core upcasts saves it needs
  (`ph_save.astype(fp64)`, `w_save.astype(fp64)` at `operational_mode.py:683,691`), so fp32
  saves are tolerated as long as the island upcasts.

- **H-7 (a) — `_inverse_density_from_theta_pressure` / `calc_p_rho_step` follow arg dtype.**
  `acoustic_wrf.py:134`, `calc_p_rho.py:132` derive their buffers from `theta`/`p` argument
  dtype. On the prep path `prep.alt` is built from a fp32-derived state (H-3). **Guard
  G-H7 = G-L3:** force-cast theta/p to fp64 at these entries so the EOS is unconditionally
  fp64.

**Net for the implementer:** the speedup-defeat hazards (H-1, H-5) are already mitigated
by `result_type`/`_output_dtype`; the contamination hazards (H-2, H-3, H-7) are the real
work — **make the acoustic/pressure/φ island force-upcast its inputs to fp64** so the
LOCK is intrinsic, not caller-dependent. That is ~6 one-line `.astype(jnp.float64)` edits
plus the assertions.

---

## 5. Validation plan — gates + exact proof artifacts the implementer must produce

All three gate families must pass; any failure reverts the offending field to fp64 (per
ADR-007 the unit of authorization is the field).

### Gate 2a — idealized close gates (hard floor; MUST still PASS)
Run the **parallel fp32 variant** (§3.3, `force_fp64=False`) of:
- **Warm bubble** — `gpuwrf.ic_generators.idealized.run_warm_bubble_case` →
  `proofs/f2/`. PASS = no detonation, `max|w|` saturates physically, θ′ field within the
  idealized tolerance of the fp64 reference. **This is the canonical fp32 failure to
  watch** (history: fp32 lost the perturbation and detonated, `daily_pipeline.py:168-170`).
- **Density current / Straka** — `run_density_current_case` → compare against
  `proofs/sprintU/straka_canonical_parity.json` + `straka_deformation_gate.json`
  (front ~4.25 km @ 300 s, `max|w|` ~22 m/s).
- **Artifact:** `proofs/perf/fp32_idealized_gate.json` with, per case:
  `{case, force_fp64, max_abs_w, front_position_m (straka), theta_prime_max_min,
  detonated: bool, fp64_reference_delta}` for **both** fp64 and fp32 runs side by side.
- **Decision rule:** if warm-bubble fp32 fails, keep `theta` fp64 and re-run with only
  u/v/q*/physics fp32 (matrix override), then re-gate.

### Gate 2b — operational RMSE (binding fitness test)
Re-run the +1h/+3h skill signal at fp32 (`force_fp64=False`) and compare to the **frozen
fp64 numbers** (current production):

| Field | fp64 +1h | fp64 +3h | tolerance band (each lead) |
|---|---|---|---|
| T2  | **1.351** K   | 1.83 K   | `|RMSE_fp32 − RMSE_fp64| ≤ 0.10 K` |
| U10 | **2.216** m/s | 1.91 m/s | `≤ 0.10 m/s` |
| V10 | **3.689** m/s | 2.75 m/s | `≤ 0.10 m/s` |

(+1h fp64 anchors per the manager's task brief / M19 memory: T2 1.351 / U10 2.216 /
V10 3.689. The +3h fp64 anchors are from `fp32_downcast_plan.md`; the implementer must
re-measure the fp64 baseline in the SAME harness run so the comparison is apples-to-apples,
not against a stale number.)

- **Tolerance rationale:** 0.10 K / 0.10 m/s is well below the CPU-vs-observation noise
  floor and below the fp64-vs-CPU-WRF skill gap itself (T2 already 1.35 K off CPU-WRF), so
  an fp32 perturbation under 0.10 is operationally invisible. This is a **forecast-impact**
  band, not a ULP band — per the project validation philosophy (operational RMSE > bitwise
  parity). A ULP framing is reported as *secondary* diagnostics only (mean/max per-cell
  |Δfield| fp32-vs-fp64 at +1h), not a pass/fail gate.
- **Finiteness + conservation (binding):** every emitted field finite at every lead; the
  dry-mass residual and total-water residual must be `≤` the fp64 run's value (no
  conservation regression). The 24 h coupled real-case must stay finite end-to-end.
- **Artifact:** `proofs/perf/fp32_skill_gate.json`:
  `{lead_h, force_fp64, t2_rmse, u10_rmse, v10_rmse, dt2, du10, dv10, pass_band: bool,
  mass_residual, total_water_residual, all_finite: bool}` for fp64 and fp32, at +1h/+3h
  (extend to 6/12/24 h via the M9 OOM-fixed `_advance_chunk_and_snapshot` path).

### Gate P-2 — HLO dtype/convert audit (defeat + contamination sentinel)
Extend `proofs/perf/fusion_transfer_audit.py::_scan_hlo` (`:53-74`) to count dtype
conversions in the compiled `run_forecast_operational` HLO:
```
+        "convert_f32_to_f64": count(r"convert\(f64.*f32"),   # fp32->fp64 upcasts
+        "convert_f64_to_f32": count(r"convert\(f32.*f64"),   # fp64->fp32 downcasts
```
(verify the exact HLO `convert` spelling on this XLA build when implementing; the regex
is a shape, not a guarantee). The harness already lowers/compiles the real case
(`fusion_transfer_audit.py:41-49,83-88`). **PASS criteria:**
- fp32-defeat sentinel: the count of fp64 ops on FP32-OK leaves inside the loop is ~0
  except at the named island boundaries (i.e. the physics/advection fusions are fp32).
- LOCK-contamination sentinel: **zero** `convert(f64->f32)` on LOCK leaves
  (mu/p/ph/w and acoustic coefficients) inside the loop body.
- **Artifact:** `proofs/perf/fp32_hlo_audit.json` (convert counts + the per-field dtype
  assertion from G-H5) + the saved HLO text.

### Gate 2c — w→fp32 (PHASE 2 ONLY, deferred)
`state.w` stays **fp64** in phase 1 (LOCK, `precision.py:91`). Phase 2 follows
ADR-007:70 exactly: paired 24 h fp64-w vs fp32-w, gate `|ΔRMSE| < 0.10`, inspect w-column
spectra at sea/lee/ridge/peak for spurious 2Δx noise, water-budget residual ≤ 1e-10. Out
of scope for the phase-1 sprint this spec governs.

### Test sentinels (cheap, CPU-runnable, no GPU forecast)
Add a unit test mirroring the existing `tests/test_m6b_operational_theta_fix.py` style that
asserts G-H5 (post-enforce dtypes) and the LOCK invariant for both `force_fp64=True` and
`False`, plus a `jax.eval_shape` carry-dtype-stability check (G-H6). These catch the prior
bug classes #1/#2/#4 as fast regressions without a GPU run.

---

## 6. Expected speedup + memory reduction (honest, bounded)

**Hardware:** RTX 5090 (Blackwell, consumer) FP64:FP32 throughput ≈ **1:64** (ADR-007
Context). So any work that stays fp64 sees no arithmetic benefit.

**Speedup decomposition (warmed, vs the current fp64 ~47 ms/step):**
- The acoustic/pressure/mass/φ island stays fp64 and is the per-step hot path (10 acoustic
  substeps × implicit w/φ solve × 3 RK substeps). **No fp32 arithmetic benefit there.**
- fp32 win is concentrated in: (a) **physics** column kernels (Thompson/MYNN/RRTMG/surface)
  — ADR-007 micro-bench ~3× fp64→fp32 on GPU; (b) **non-acoustic advection/diffusion**
  arithmetic — ADR-007 M4 micro-bench 713 µs→203 µs (~3.5×), **but that micro-bench
  downcast the whole state including pressure**, which this spec does NOT, so the realized
  dycore-advection gain is materially smaller; (c) **memory traffic / working set** halved
  for the FP32-OK leaves.
- **Honest end-to-end projection from mixed precision alone: ~1.3–1.8×** (≈ 26–36 ms/step
  warmed). The larger 3–4× a full-fp32 micro-bench suggests is **not reachable** while the
  fp64 acoustic island dominates; that requires the orthogonal XLA fusion / single-scan
  work, not precision. Frame any speedup against the **28-rank CPU WRF** denominator on
  this workstation (current fp64 ≈ 50–85× CPU-WRF, per the M19 preview), so post-fp32 ≈
  65–150× CPU-WRF if the projection holds — to be **measured**, not assumed.

**Memory reduction:** the FP32-OK leaves (u, v, theta, qv, and the 9 hydrometeors + qke,
masks, fp32 boundary leaves) halve from 8→4 bytes/element. The LOCK leaves (p/p_total/
p_perturbation, ph×3, mu×3, w, surface handles, acc, fp64 boundary leaves) stay 8 bytes.
The FP32-OK set is the **majority of the prognostic element count** (u/v/theta/qv +
hydrometeors), so resident state ≈ **0.55–0.65×** its fp64 size, and per-step device
traffic for those leaves halves. Task-1 measured fp64 peak ≈ 9.0 GB at 180 steps; expect
the working set to drop toward ~5–6 GB, easing the M9 RRTMG-transient OOM headroom — a
real secondary benefit beyond wall-clock. **All numbers above are projections; the
binding figures are whatever the §5 warmed-timing + memory artifacts measure post-flip.**

---

## 7. Sequencing (for the gated follow-up sprint)
1. Add `force_fp64` to `DailyPipelineConfig` + idealized `_build_setup`/`_initial_carry`
   params (§3.2, §3.3) — enables paired gating with no per-run src edits.
2. Harden the acoustic/pressure/φ/EOS island to force-upcast its inputs to fp64
   (G-H2/G-H3/G-L2/G-L3/G-L4 — ~6 `.astype(jnp.float64)` edits at the named lines).
3. Extend `fusion_transfer_audit.py` with the convert-op counters (Gate P-2) + add the
   dtype-assertion unit test (G-H5/G-H4/G-H6).
4. Flip `daily_pipeline.py:204` to `force_fp64=False` (§3.1).
5. Run Gate 2a (idealized fp32) → Gate P-2 (HLO audit) → Gate 2b (+1h/+3h skill + finite +
   conservation) → warmed-timing + memory artifact. Each emits its JSON proof.
6. Merge only if all of 2a + 2b + P-2 pass; otherwise revert the failing field to fp64 and
   re-gate.

---

## Appendix A — file:line index (audit trail)
- Switch + matrix: `daily_pipeline.py:204`; `contracts/precision.py:77-138`;
  `contracts/state.py:32` (`_zeros` uses matrix), `:566-588` (`replace` `_cast`).
- Enforcement: `runtime/operational_mode.py:319-336` (`_enforce_operational_precision`),
  applied `:1542,1763,1818` (+ `:1910,1958,1998,2058` sibling entries).
- Carry dtype: `runtime/operational_state.py:103,112,122-128,132`.
- Acoustic island: `dynamics/acoustic_wrf.py:636` (calc_coef_w fp64 force-cast),
  `:232-345` (pressure/PGF, dpn at p_perturbation.dtype `:304,334`), `:134` (EOS inv ρ);
  `dynamics/core/advance_w.py:116,131-440`; `dynamics/core/calc_p_rho.py:79-160`;
  `dynamics/core/small_step_prep.py:169,213,215,219` (u_work/theta_work fp32-derived).
- Island assembly (upcast hazards): `runtime/operational_mode.py:606-692`
  (`_acoustic_core_state`, u/v/w NOT upcast `:633-637`), `:695-849`
  (`_acoustic_core_state_from_prep`, rhs_ph/pg_buoy from `state.u/v/w` `:741-789`).
- fp32-safe arithmetic (no change): `dynamics/flux_advection.py:159,219,401,435,498,532`
  (`result_type` scatter buffers); `coupling/physics_couplers.py:352-376` (`_output_dtype`),
  `:639-648,778-779` (live-dtype writes), `:729-735` (fp64 physics boundary).
- Boundary blend: `coupling/boundary_apply.py:137` (`.astype(boundary.dtype)`).
- IC build: `integration/d02_replay.py:747` (`State.zeros`), `:790` (`replace`, `_cast`).
- Idealized gate: `ic_generators/idealized.py:560-596,616-619` (force_fp64=True hard-coded).
- HLO audit harness: `proofs/perf/fusion_transfer_audit.py:41-49,53-74,83-88`.
