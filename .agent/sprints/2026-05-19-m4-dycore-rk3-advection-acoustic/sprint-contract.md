# Sprint Contract

Sprint ID: `2026-05-19-m4-dycore-rk3-advection-acoustic`
Milestone: M4 — Minimal Dycore
Sequence: S1 (intended as the ONLY M4 implementation sprint per the "big smart steps" directive — single substantial sprint delivering reduced split-explicit dycore + debug hooks + tier-1/2/3 validation + ADR-003 draft)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (**Claude Opus 4.7 `xhigh` — explicitly tasked with HLO debug-vs-stripped diff verification + Allocation Audit + tier-1/2/3 reproduction, not just correctness**)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Approval status: **AMENDED 2026-05-19 (attempt 2)** — reviewer Decision = Reject on attempt 1 with 4 blockers + 2 majors + 1 minor; cross-AI tester (Claude Opus 4.7 xhigh) independently flagged 3 of the same plus 3 more. See `reviewer-report.md` and `tester-report.md`. Worker attempt 1 archived as commit `5335131`. Fixes below are MUST-fix for attempt 2.

### Fix-cycle amendments (attempt 2) — derived from reviewer-report.md blockers + majors

**Blocker-class (MUST fix for sprint to be reconsidered):**

- **AC #1.1.fix (RK3 mathematical correctness — reviewer Blocker #1, `src/gpuwrf/dynamics/rk3.py:45,51,57` + `src/gpuwrf/dynamics/tendencies.py:11`)**: The current `rk3_step` advances sequentially by `dt/3`, then `dt/2`, then `dt` on the already-updated state and adds the FULL passed tendency at each stage. This is incorrect: a constant RHS should integrate to `state + dt*tendency` exactly. Reviewer's spot-check: `dt=6`, `theta_tendency=1` → observed delta `11.0`, expected `6.0`. Fix: implement Wicker–Skamarock RK3 correctly: stage k computes tendency on stage state, then advances ORIGINAL state by `dt/(4-k)` (with appropriate stage-state vs origin-state distinction; SSP-RK3 references in `references/` if added). Add `tests/test_m4_rk3.py::test_rk3_constant_tendency_integrates_to_dt_times_tendency` that runs `step(state, tend, ...)` with `tend = constant` and asserts state delta = `dt * tend` to fp64 precision. The bug must NOT recur.

- **AC #3.fix (Real tier-1 oracle — reviewer Blocker #2)**: Current `run_tier1` runs `fixture_reference_update` and compares it against the fixture's own `phi_next` — a tautology. The dycore's 5H/3V upwind operators (`compute_advection_tendencies`, `advect_mass_scalar` in `src/gpuwrf/dynamics/advection.py:125,174`) are never validated against any oracle. Per `validating-physics` skill ("operator-mismatch handling: match or generate sibling fixture, never coerce tolerance"): **generate a sibling analytic fixture for the dycore's actual 5th-order upwind scheme**. Concretely: extend `scripts/generate_analytic_fixtures.py` (or add a sibling generator) to produce `fixtures/manifests/analytic-stencil-3d-upwind5-v1.yaml` + `fixtures/samples/analytic-stencil-3d-upwind5-v1.npz` containing `(phi_initial, u_face, v_face, w_face, phi_next_upwind5)` where `phi_next_upwind5` is the NumPy reference evaluation of the SAME 5th-order upwind operator the dycore implements. Then `run_tier1` calls `advect_mass_scalar` on `phi_initial` with the same dt/dx and compares against `phi_next_upwind5`. Tolerance: `tolerance_abs=1e-10`, `tolerance_rel=1e-12` at fp64 (same as existing fixture). Update `M1-DONE.md`'s fixture inventory accordingly if needed; coordinate with M1 manifest schema.

- **AC #4.fix (Real tier-2 trajectory — reviewer Blocker #3 + tester gap #2)**: Current `density_current_state` uses `u=v=w=0, p=1.0 uniform, mu=1.0` — a degenerate steady state where every prognostic stays at its initial value. Real fix: implement a **non-trivial** initial condition. Options (worker picks ONE, documents in maintainability.md):
  - **Option A (preferred)**: Real 2D density current — cold blob in a hydrostatically-balanced background. theta-perturbation = `-15 K` Gaussian centred at x=mid, z=lower-third with half-width 4 km horizontal × 1.5 km vertical. Background theta(z) hydrostatic. Background pressure(z) hydrostatic via integrating dp/dz = -rho*g. `u=v=w=0` initial. After 100 steps, the perturbation MUST have evolved (non-zero RMS velocity, non-zero theta departure from initial). Tier-2 then asserts: (i) `mass_residual_relative ≤ 1e-8` (slightly looser due to round-off across acoustic substeps), (ii) `qv_positivity_violations == 0` throughout, (iii) `nan_inf_violations == 0` throughout, (iv) **NEW**: `final_state_differs_from_initial: True` (`max(|theta_final - theta_initial|) > 0.1 K`) to prevent regression to no-op.
  - **Option B (simpler)**: 2D pure advection of a tracer bump (no acoustic, no buoyancy). Velocity field `u = const > 0` everywhere, tracer bump translates. Tier-2 asserts mass conservation + non-trivial movement after N steps (`max(tracer_t-tracer_0) > 0.1 * max(tracer_0)` at a downwind point).
  - **MUST add**: `tests/test_m4_tier2_invariants.py::test_tier2_is_not_a_noop` that asserts the trajectory's max state-delta is non-zero. The tester's `test_tier2_density_current_ic_is_a_noop_simulation` adversarial test (already in `tests/test_m4_tester_adversarial.py`) must REMAIN passing in its inverted form (the new IC must not be a no-op).

- **AC #5.fix (Real tier-3 through the integrated dycore — reviewer Blocker #4)**: Current `run_tier3` imports `ddx4_centered` directly and uses a local 1D RK helper that bypasses `step`/`run`/`compute_advection_tendencies`/the 5H/3V operator/the acoustic substep entirely. The artifact certifies a centered-reference test, not the dycore's convergence. Fix: convergence test MUST call `dynamics.step.run(state, tendencies, grid, dt, n_steps, n_acoustic=N, debug=False)` — the public dycore API — at three (dx, dt) levels with halved dx and halved dt (or quartered). Recommended setup: 1D linear advection of a smooth Gaussian bump on a 1D-flat-grid (set ny=4, nz=4 with periodic BCs and uniform velocity), translated through the actual dycore by setting tendencies appropriately OR by initialising `u = const, v=w=0, p_prime=0, theta=Gaussian bump`. Reference solution = analytic translation. Report `observed_order >= expected_order - 0.5` where `expected_order = min(advection_scheme_order, time_scheme_order) = min(5, 3) = 3`.

- **AC #1.2.fix (Velocity advection completeness — reviewer Major #6, `src/gpuwrf/dynamics/advection.py:135,141,147`)**: Current `advect_u_face` only applies x self-advection, `advect_v_face` only y self-advection, `advect_w_face` only vertical self-advection. The contract (AC #1.2 line 119–125) called for **5th-order upwind horizontal advection for `u`, `v`, `w`, `theta`** plus 3rd-order vertical. Worker MUST extend the velocity-face advection operators to advect each velocity component by the **full velocity field**: `du/dt += -u*du/dx - v*du/dy - w*du/dz` (with the cross-term interpolations needed for C-grid; reference: WRF-ARW Tech Note §3.3). If full implementation is too large for one cycle, manager pre-approves narrowing scope to "horizontal-advection-only of all four prognostics (u, v, w, theta) by the full horizontal velocity (u, v interpolated to each stagger), vertical advection by w" — but this narrower scope MUST be reflected in updated test `test_m4_advection.py::test_velocity_advection_includes_cross_terms` asserting cross-terms are non-zero.

**Major-class (MUST fix, blocking sprint accept):**

- **AC #2.4.fix (Real hand-stripped HLO sibling — reviewer Major #5, `src/gpuwrf/dynamics/step.py:54,58`)**: Current implementation exposes `step_stripped_reference` inside `step.py` that dispatches through the SAME `_step_impl(..., False)` as production. HLO identity is by code-path, not by literal stripping — weaker evidence than AC #2.4 demands. Fix: create a real `src/gpuwrf/dynamics/step_debug_stripped.py` that is a hand-edited duplicate of `step.py` with EVERY `assert_*` / `snapshot` / debug-branch call physically deleted (NOT routed through `if not enabled` — actually removed). The two files should be DIFFERENT in source but produce IDENTICAL HLO. `scripts/m4_hlo_diff.py` then compiles `step(debug=False)` from `step.py` vs `step_debug_stripped()` from `step_debug_stripped.py` and emits the diff. Diff must be 0 bytes.

- **AC #5.1.fix (Fix `scripts/m4_hlo_diff.py:54-55` typo)**: The line passes `prod` (not `stripped`) to `write_hlo` for the `dycore_step_debug_stripped.txt` artifact. The actual on-disk `dycore_step_debug_stripped.txt` IS the production HLO. Fix the typo so the named artifact contains what its filename advertises. The diff computation itself is sound; this is just artifact integrity.

**Minor-class (SHOULD fix; not blocking):**

- **AC #6.fix (Real M5 dryrun metrics — reviewer Minor #7, `scripts/m4_m5_gate_dryrun.py:33`)**: Hardcoding `local = 0` and `registers = 0` while documenting the rationale is acceptable IF the artifact's `rationale` field explicitly states "metrics unavailable from HLO — values are placeholders pending ncu/cuobjdump follow-up (M4-x or M5)". Better: emit `local_memory_bytes_per_kernel: null` and `registers_per_kernel: null` (JSON null, not `0`) so a downstream reader does not mistake unavailable-data for measured-zero. Update `check_m4_done.py` accordingly to accept null as "unknown but not failing."

**No regression in attempt 2**: the proof objects worker attempt 1 ACTUALLY got right (`host_to_device_bytes_post_init: 0`, `device_to_host_bytes_post_init: 0`, `temporary_bytes_per_step: 0`, HLO debug-branch absence verified by reviewer's `isfinite`-token check) MUST hold in attempt 2 as well. The hot-path-discipline + zero-transfer + zero-temp-bytes claims were independently verified; do not regress them while fixing the validation gaps.

**Worker attempt 2 scope**: only the files needed for these 7 fixes plus the affected tests and artifact regeneration. NO new modules. NO scope creep. The file ownership in the original contract remains binding; worker may also add `fixtures/manifests/analytic-stencil-3d-upwind5-v1.yaml` + `fixtures/samples/analytic-stencil-3d-upwind5-v1.npz` + extend `scripts/generate_analytic_fixtures.py` to produce them (manager pre-approves this scope extension for fixture generation).

## Objective

Deliver the first real model-physics code: a reduced split-explicit dycore (RK3 + 5th-order upwind horizontal advection + 3rd-order vertical advection + forward–backward acoustic sub-step) built on the M3 `State`/`Tendencies` pytrees, validated on tier-1/2/3 oracles, with **constitutionally-guarded debuggability hooks** that XLA dead-code-eliminates in production.

Per user directive of 2026-05-19 (M3 closeout + mid-M3 debuggability directive):

> "Demand elegant efficient core code. Not a single waste in spacetime complexity is acceptable. Every variable creation needs to be justified. Like WRF is for CPU."
> "Plan in now that there are debug lines used in debug mode only that would greatly improve debug complexity on kernel-related problems later. Nothing that sets off efficiency in real final run compile."

**Backend: JAX + XLA** per ADR-001 (ACCEPTED 2026-05-19). Pin: `jax[cuda13]==0.10.0`.
**State layout: SoA + C-grid + fp64** per ADR-002 (ACCEPTED 2026-05-19). Hot path is one `@jax.jit` around `jax.lax.scan`, zero allocations in the scanned body.

## Non-Goals

- **No physics schemes.** No microphysics, no PBL, no radiation, no surface. M5 territory. Dycore is dry-air dynamics + tracer advection only.
- **No terrain-following coordinates.** Flat orography is acceptable for M4 idealized cases. Real terrain (em_hill2d_x) deferred to M5/M6 when surface coupling lands.
- **No multi-GPU halo.** `apply_halo(state, halo_spec)` stays the M3 single-GPU no-op. The dycore MUST call it at the right points so a future multi-GPU halo drop-in works, but the body remains no-op.
- **No AIFS BC ingest.** M7. Use idealized periodic / rigid-wall BCs for M4.
- **No Pallas / Triton drop-down.** Pure `jax.jit` + `jax.numpy` + `jax.lax`. (If the M5 gate dry-run trips, the per-scheme Triton fallback ADR opens — but only after this sprint completes.)
- **No mixed precision yet.** All M4 hot-path code is fp64. ADR-003 *proposes* per-field downcasts with validation evidence; actual downcasting lands in M5 or as a follow-up sprint.
- **No restart / I/O.** M7.

## File Ownership

Worker may create or edit only these paths:

### Dynamics core (new module — the heart of M4)
- `src/gpuwrf/dynamics/__init__.py` (new)
- `src/gpuwrf/dynamics/rk3.py` (new — Wicker–Skamarock RK3 large step)
- `src/gpuwrf/dynamics/advection.py` (new — 5th-order upwind horizontal + 3rd-order vertical)
- `src/gpuwrf/dynamics/acoustic.py` (new — forward–backward acoustic sub-step)
- `src/gpuwrf/dynamics/step.py` (new — top-level `step(state, tend, grid, dt, *, n_acoustic=4, debug=False) -> state` + `run(state, tend, grid, dt, n_steps, *, n_acoustic=4, debug=False) -> state` using `jax.lax.scan`)
- `src/gpuwrf/dynamics/tendencies.py` (new IF NEEDED — small helpers for pressure-gradient / buoyancy / divergence accumulation; keep tight)

### Debug hooks (new module — constitutionally-guarded)
- `src/gpuwrf/debug/__init__.py` (new)
- `src/gpuwrf/debug/asserts.py` (new — `assert_finite`, `assert_physical_bounds`; both gated on `enabled: bool` Python branch)
- `src/gpuwrf/debug/snapshots.py` (new — ring-buffer stage snapshot, gated on `enabled: bool` Python branch)

### Validation (new module — the tier engines)
- `src/gpuwrf/validation/__init__.py` (new if missing)
- `src/gpuwrf/validation/tier1.py` (new — fixture parity engine)
- `src/gpuwrf/validation/tier2.py` (new — invariants engine: mass, positivity, finiteness)
- `src/gpuwrf/validation/tier3.py` (new — convergence engine: 2-norm error vs analytic at three (dx,dt))

### Scripts
- `scripts/m4_run_dycore.py` (new — single-command CLI: runs the dycore, emits dycore_profile.json + transfer_audit.json + spacetime_budget.json + HLO dumps)
- `scripts/m4_run_validation.py` (new — single-command CLI: runs tier1/2/3, emits all tier JSONs)
- `scripts/m4_m5_gate_dryrun.py` (new — runs the M5 stop/go gate logic against the dycore's compiled HLO; emits m5_gate_dryrun.json)
- `scripts/m4_hlo_diff.py` (new — produces the production-vs-stripped HLO diff)

### Artifacts (under `artifacts/m4/` — currently gitignored; commit with `git add -f`)
- `artifacts/m4/dycore_profile.json` (new)
- `artifacts/m4/transfer_audit.json` (new)
- `artifacts/m4/spacetime_budget.json` (new)
- `artifacts/m4/tier1_advection_parity.json` (new)
- `artifacts/m4/tier2_invariants.json` (new)
- `artifacts/m4/tier3_convergence.json` (new)
- `artifacts/m4/m5_gate_dryrun.json` (new)
- `artifacts/m4/hlo_dump/dycore_step_production.txt` (new)
- `artifacts/m4/hlo_dump/dycore_step_debug_stripped.txt` (new)
- `artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff` (new — MUST be empty file)
- `artifacts/m4/maintainability.md` (new ≤500 words — per-module justification)
- `artifacts/m4/agent_success.json` (new)

### ADR-003 (manager-finalized; worker drafts core technical sections)
- `.agent/decisions/ADR-003-dycore-precision.md` (new — worker drafts; manager finalizes before critical-review)

### Tests
- `tests/test_m4_advection.py` (new — operator-level tests for 5th-order upwind H + 3rd-order V; periodicity; conservation)
- `tests/test_m4_rk3.py` (new — SSP-RK3 test on a model ODE; 3rd-order time convergence)
- `tests/test_m4_acoustic.py` (new — small-amplitude sound wave test; phase speed within tolerance)
- `tests/test_m4_dycore_step.py` (new — full `step(...)` invariants: pytree shape preservation, no NaN/Inf)
- `tests/test_m4_debug_hooks.py` (new — **critical**: asserts production HLO is byte-identical to debug-stripped sibling)
- `tests/test_m4_tier1.py` (new — wraps the tier-1 fixture parity assertion)
- `tests/test_m4_tier2_invariants.py` (new — wraps the tier-2 density-current invariant assertion)
- `tests/test_m4_tier3_convergence.py` (new — wraps the tier-3 convergence assertion)

Any change outside this list requires manager approval. **Do not touch `src/gpuwrf/contracts/`, `src/gpuwrf/timestep/`, `src/gpuwrf/profiling/`, `src/gpuwrf/fixtures/`, `src/gpuwrf/backends/`, or governance files.**

## Inputs

- ADR-001 (`.agent/decisions/ADR-001-backend-selection.md`) — backend is JAX, pin `jax[cuda13]==0.10.0`; M5 stop/go gate thresholds binding for the dry-run.
- ADR-002 (`.agent/decisions/ADR-002-state-layout.md`) — `State`/`Tendencies` pytrees, C-grid, fp64.
- M3 implementation under `src/gpuwrf/contracts/` and `src/gpuwrf/timestep/dummy_loop.py` — the dycore replaces the *body* of the scanned step while keeping the M3 loop shape (one @jit, scan-based, zero alloc).
- M3 profiling tooling under `src/gpuwrf/profiling/` — reuse `transfer_audit` + `budget` modules; do NOT re-implement.
- M1 fixture `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` + `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` — tier-1 advection oracle. **Worker MUST read `scripts/generate_analytic_fixtures.py` to determine the exact analytic operator used to generate `phi_next`, then match it (or document a sibling-fixture path).**
- `PERFORMANCE_TARGETS.md` profile JSON schema + hard rules.
- `VALIDATION_STRATEGY.md` — tier definitions.
- `PRECISION_POLICY.md` — fp64 default; ADR-003 proposes validated overrides.
- `.agent/milestones/ROADMAP.md` M4 — proof-object list.
- `.agent/goals/M4-DONE.md` — binding oracle.
- Project memory `feedback_code_quality_bar.md` — MANDATORY (M3 standard, raised).
- Project memory `feedback_debuggability_hooks.md` — MANDATORY (M4 binding directive).
- Project memory `project_state_layout.md` — practices learned from M3 (dt-static, eq+hash, eliminate-cause-not-symptom).

## Acceptance Criteria

All must hold for closeout. Numbered for reviewer traceability.

### 1. Reduced split-explicit dycore (`src/gpuwrf/dynamics/`)

1.1. **`rk3.py`** — Wicker–Skamarock RK3 large step, 3 stages, pure-functional on the `State` pytree. Each stage:
- (a) compute tendencies on the current stage state
- (b) advance by `dt/k` for stage k=1,2,3
- (c) the acoustic sub-step is invoked inside stages 2 and 3 (NOT stage 1) per WRF-ARW convention

1.2. **`advection.py`** — 5th-order upwind horizontal advection for `u`, `v`, `theta`, `qv`; 3rd-order upwind vertical advection. Operators are pure-functional (`state → tendency`), respect C-grid staggering, and call `apply_halo(state, halo_spec)` at the right point. The horizontal-advection kernel is the dominant arithmetic-intensity kernel in the dycore.

1.3. **`acoustic.py`** — forward–backward acoustic sub-step for `u`, `v`, `w`, `p_prime`, `ph_prime`. `n_acoustic` is a `static_argname` (typically 4–6). Sub-step uses fp64 accumulation per ADR-003 default.

1.4. **`step.py`** — public API:
- `step(state, tendencies, grid, dt, *, n_acoustic=4, debug=False) -> state` (one large-step)
- `run(state, tendencies, grid, dt, n_steps, *, n_acoustic=4, debug=False) -> state` (uses `jax.lax.scan`, **one** `@jax.jit` for the entire loop)
- `dt` and `n_steps` and `n_acoustic` and `debug` are all `static_argnames` (the M3 `dt`-static lesson).
- Halo is applied at every RK stage boundary AND at every acoustic sub-step boundary.

1.5. **Hot-path discipline (per `feedback_code_quality_bar.md` + project memory practice #3):**
- ZERO `jnp.array(...)` / `jnp.zeros(...)` / `jnp.empty(...)` calls inside `step()` or any function it transitively calls during JIT-traced execution.
- All intermediate buffers come from preallocated `Tendencies` (the M3 contract).
- Spacetime budget MUST report `temporary_bytes_per_step == 0`.

1.6. The full integrated dycore at `(nz=40, ny=80, nx=80)` produces a stable 100-step run with NO NaN/Inf on the density-current test setup. Wall-time per step is a SOFT bound (≤ 5 ms) — performance optimization is M5 work; this sprint just needs not-pathological.

### 2. Debuggability hooks (`src/gpuwrf/debug/`) — CONSTITUTIONAL

2.1. **`asserts.py`**:
- `assert_finite(x: jnp.ndarray, name: str, *, enabled: bool) -> jnp.ndarray`. When `enabled=False`, the function body is `if not enabled: return x` (no XLA op emitted). When `enabled=True`, uses `jax.experimental.checkify.check` OR `jnp.where(jnp.isfinite(x).all(), x, jnp.nan)` with a named tag. The non-enabled path MUST be a pure Python early-return so XLA's tracer never sees the assertion ops.
- `assert_physical_bounds(x, lo, hi, name, *, enabled)`: same pattern. Used for `qv >= 0`, `T > 0`, `0 ≤ qv ≤ 1`, etc.

2.2. **`snapshots.py`**:
- `snapshot(state, stage: str, *, enabled, ring_size=8) -> state`. When `enabled=False`, returns `state` immediately. When `enabled=True`, writes to an in-JIT ring buffer (an extra pytree leaf carried through the scan, but only when `enabled=True` so it does not exist in the production trace at all).
- `dump_snapshots(state) -> dict[str, jnp.ndarray]`: post-hoc reader. Only meaningful if `enabled=True` was set at trace time.

2.3. **`debug: bool = False` is a `static_argname` on every hot-path `@jit`** (i.e. `step`, `run`, and any auxiliary `@jit` wrapping a stage). XLA sees two separate compiled programs for `debug=False` and `debug=True`. Production code calls with `debug=False` always.

2.4. **HLO byte-identity gate (the constitutional debuggability evidence):**
- Worker writes `src/gpuwrf/dynamics/step_debug_stripped.py` (internal sibling, NOT exported in `__init__.py`) which is a hand-stripped version of `step.py` with ALL `assert_*` and `snapshot` calls deleted.
- `scripts/m4_hlo_diff.py` compiles both `step(..., debug=False)` and `step_debug_stripped(...)` with identical inputs, runs `.lower(...).compile().as_text()` on each, normalises whitespace + variable-naming via a simple tokenizer, and writes the diff to `artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff`.
- **The diff file MUST be 0 bytes.** Non-empty = debug branch leaking into production = sprint fail.
- `tests/test_m4_debug_hooks.py` asserts the same.

2.5. Worker-report MUST include the SHA-256 of the empty diff file (which is `e3b0c442...855` for an empty file) as proof.

### 3. Tier-1 fixture parity (`src/gpuwrf/validation/tier1.py` + artifact)

3.1. Read `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` + the `.npz`.
3.2. Read `scripts/generate_analytic_fixtures.py` to determine the exact analytic operator that produced `phi_next`. Document the operator in `artifacts/m4/maintainability.md` (≤100 words). If the dycore's advection scheme is the same operator, parity is direct. If not, EITHER (a) implement a thin wrapper that calls the same operator OR (b) extend `generate_analytic_fixtures.py` to also produce a sibling reference for the dycore's scheme, with new manifest + new test (manager pre-approves this branch).
3.3. Run the dycore's advection kernel once on `(phi_initial, u_face, v_face, w_face)` with the fixture's dt/dx.
3.4. Emit `artifacts/m4/tier1_advection_parity.json`:
```json
{
  "fixture_id": "analytic-stencil-3d-advdiff-v1",
  "operator": "<documented operator name>",
  "max_abs_err": <float>,
  "max_rel_err": <float>,
  "tolerance_abs": 1e-10,
  "tolerance_rel": 1e-12,
  "pass": <bool>
}
```
3.5. `pass: true` required. If `false`, this is a worker-report blocker; worker investigates root cause (likely halo edge, fp64 silent downcast, or operator mismatch) and re-runs.

### 4. Tier-2 invariants (`src/gpuwrf/validation/tier2.py` + artifact)

4.1. 100-step run of a 2D density-current setup on `(nz=40, ny=80, nx=80)` with periodic x-BC, rigid lid + rigid floor, `dt=2.0 s`, n_acoustic=4. Initial condition: cold blob (theta perturbation = -15 K in a 4 km × 1.5 km region centred at x=mid, z=lower-third). This is a standard WRF-community idealized test; reference physics described in Straka et al. (1993).
4.2. Worker may use a simpler analytic setup (e.g. linear-advection-only of a Gaussian bump) if the full density current is unstable at the M4 dycore's dt/CFL combo — document in `maintainability.md`.
4.3. Compute:
- `mass_residual_relative = |total_mass(t=100*dt) − total_mass(t=0)| / total_mass(t=0)`. Bound: ≤ 1e-10 for fp64 fields.
- `qv_positivity_violations = count(qv < 0)` across the 100-step trajectory.
- `nan_inf_violations = count(non-finite cells)` across the 100-step trajectory.
4.4. Emit `artifacts/m4/tier2_invariants.json` with all four keys + `pass: bool`. `pass: true` iff `mass_residual_relative <= 1e-10 AND qv_positivity_violations == 0 AND nan_inf_violations == 0`.

### 5. Tier-3 convergence (`src/gpuwrf/validation/tier3.py` + artifact)

5.1. 1D linear-advection-of-a-smooth-bump test (simpler than density current; analytic exact solution available) at three (dx, dt) levels:
- Coarse: dx=2 km, dt=20 s, 100 steps.
- Medium: dx=1 km, dt=10 s, 200 steps.
- Fine: dx=0.5 km, dt=5 s, 400 steps.
At each level, compute the 2-norm error vs the exact analytic solution.

5.2. Worker may use a different convergence test that exercises the dycore more completely (e.g. 2D internal gravity wave or 1D advection-diffusion) IF documented in `maintainability.md` and the convergence is computable in closed form.

5.3. `observed_order = log2(err_coarse / err_medium)` (and same for medium → fine; average the two).

5.4. `expected_order = min(advection_scheme_order, time_scheme_order) = min(5, 3) = 3` for the integrated dycore (RK3 limits time order). If the worker exercises a pure-advection test (no acoustic substep), `expected_order = min(5, 3) = 3` still.

5.5. Emit `artifacts/m4/tier3_convergence.json` with `observed_order`, `expected_order`, `errors_per_level: [...]`, `pass: bool`. `pass: true` iff `observed_order >= expected_order - 0.5`.

### 6. M5 stop/go gate dry-run (`scripts/m4_m5_gate_dryrun.py` + artifact)

6.1. Compile the integrated `step(state, tend, grid, dt, n_acoustic=4, debug=False)` to HLO.
6.2. Run the M5 gate metrics extraction (re-use the M2 JAX kernel profiler tooling if applicable; otherwise reasoning from the HLO + a single `nsys` run is acceptable for the dry-run since `ncu` perfmon is blocked on this workstation).
6.3. Emit `artifacts/m4/m5_gate_dryrun.json`:
```json
{
  "kernel_launches_per_step": <int>,
  "local_memory_bytes_per_kernel": <int max across kernels>,
  "registers_per_kernel": <int max across kernels>,
  "thresholds": {
    "kernel_launches_per_step": 10,
    "local_memory_bytes_per_kernel": 256,
    "registers_per_kernel": 128
  },
  "gate_status": "pass" | "trip",
  "tripped_thresholds": [<list of tripped threshold names>],
  "rationale": "<≤200 words: what scheme is taking the budget, where pressure is>"
}
```
6.4. **A trip is NOT a sprint failure.** It is a signal to the manager that ADR-001's per-scheme Triton fallback escape hatch should be invoked. Document; do not "fix" by lowering thresholds. Per `feedback_code_quality_bar.md` lesson: eliminate the cause if possible (refactor the scheme); if not, document the trip and the manager opens the appropriate follow-up.

### 7. Profile + transfer audit + spacetime budget (`scripts/m4_run_dycore.py` + artifacts)

7.1. Reuse `src/gpuwrf/profiling/transfer_audit.py` and `src/gpuwrf/profiling/budget.py` from M3 (do NOT re-implement). The M4 invocation runs the dycore for 100 JIT-cached steps at `(nz=40, ny=80, nx=80)` and emits:
- `artifacts/m4/transfer_audit.json` — `host_to_device_bytes_post_init == 0`, `device_to_host_bytes_post_init == 0`, `iterations >= 100`.
- `artifacts/m4/spacetime_budget.json` — all six M3 keys + `temporary_bytes_per_step == 0`.
- `artifacts/m4/dycore_profile.json` — `PERFORMANCE_TARGETS.md` schema-compliant profile + kernel_launches breakdown.

7.2. HLO production dump: `artifacts/m4/hlo_dump/dycore_step_production.txt`. If >200 KB, store first 5000 lines + a `truncated: true` marker; full file in `data/scratch/`.

### 8. ADR-003 (`/.agent/decisions/ADR-003-dycore-precision.md`)

8.1. ≥1500 bytes; required tokens: `Decision:`, `Per-field precision:`, `Downcast plan:`, `Validation evidence:`.

8.2. Worker drafts the technical body. For each prognostic AND each tendency component, the table:
- field name
- current precision (fp64)
- proposed precision (fp64 retained | fp32 downcast | mixed: fp32 storage + fp64 accumulator)
- validation evidence cite (tier-1 max_abs_err at proposed precision; tier-2 mass_residual at proposed precision)
- WRF-community reference (citation or `none`)

8.3. **Hard rule per `PRECISION_POLICY.md`:** the acoustic substep accumulator stays fp64 unless ADR-003 has tier-2 mass-residual evidence ≤ 1e-10 at fp32. WRF operational experience documents fp32 acoustic accumulators introducing measurable mass drift.

8.4. Manager finalizes the draft after the sprint reviewer Accepts. Then Codex `gpt-5.5 xhigh` critical-reviews ADR-003 in a separate folder (`.agent/decisions/REVIEW-codex-ADR-003/`); manager applies findings. User explicit approval required before M5 dispatch (ADR-003 is irreversible architecture per `.agent/rules/architecture-decision-policy.md`).

### 9. Elegance & efficiency (per `feedback_code_quality_bar.md`)

9.1. Worker-report MUST include the spacetime budget table inline (same numbers as the JSON, with one-line justification per entry).
9.2. Worker-report MUST include an **Allocation Audit** section listing every `jnp.array/zeros/empty` / `flax.struct.dataclass` construction in the diff, with each tagged as: `init-only (allocated once at startup)` | `traced-init (allocated at first JIT trace, reused thereafter)` | `hot-path (FORBIDDEN — must be eliminated)`. Any `hot-path` entry = sprint fail.
9.3. Worker-report MUST include the **HLO debug-vs-stripped diff SHA-256** as proof of the constitutional gate (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` is the empty-file sha).
9.4. Every helper function has a one-line docstring justifying its existence vs inlining ("reused by N call-sites" or "encapsulates X invariant").
9.5. Reviewer MUST attest in their report: "I read every line of `src/gpuwrf/dynamics/step.py` and `src/gpuwrf/debug/asserts.py` and found {N} simplification opportunities" where N may be 0 but is explicitly claimed.

### 10. Cross-AI tester (Claude Opus 4.7 xhigh) tasking

10.1. Tester contract (auto-generated by `dispatch_role.sh`) extended for this sprint: tester MUST act as:
- correctness reviewer (re-run all validation commands from clean shell)
- aesthetics+efficiency reviewer (Allocation Audit, repeat from independent count)
- **HLO debug-vs-stripped diff verifier**: tester independently runs `scripts/m4_hlo_diff.py` from a clean shell and confirms the diff is 0 bytes. **This is the single most important cross-AI check for M4** — a different AI verifying the constitutional gate.
- tier-1/2/3 reproducer (re-run from clean shell, confirm artifact JSONs match within float tolerance).
10.2. Tester-report MUST include a section titled "HLO Identity Verification" with the byte size of the diff file produced from the tester's own run and the sha-256 thereof.

### 11. Tests

11.1. Operator unit tests (`test_m4_advection.py`, `test_m4_rk3.py`, `test_m4_acoustic.py`): each operator passes a small-case manufactured-solution test.
11.2. `test_m4_dycore_step.py`: `step(...)` preserves pytree shape + dtype; 100-step `run(...)` produces no NaN/Inf on the density-current setup.
11.3. **`test_m4_debug_hooks.py` — critical**: asserts `os.path.getsize("artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff") == 0` AND asserts the debug-branch function call with `enabled=False` adds zero ops to the lowered HLO (compare via `jax.jit(...).lower(...).compile().as_text()` against the stripped sibling).
11.4. Tier wrappers: `test_m4_tier1.py`, `test_m4_tier2_invariants.py`, `test_m4_tier3_convergence.py` each read the corresponding artifact JSON and assert `pass == true`.
11.5. `pytest -q` passes overall; suite size ≥ 320 tests after additions.

### 12. Hygiene

12.1. `validate_agentos.py` ok.
12.2. `check_m1_done.py` + `check_m2_done.py` + `check_m3_done.py` still ok (no regression).
12.3. No file >100 KB committed beyond pre-existing.
12.4. `pyproject.toml` may gain `nsys-jax` or similar if needed (justify in maintainability.md); ADR-001 venv pin held.

## Validation Commands

```bash
# Regression
python scripts/validate_agentos.py
python scripts/check_m1_done.py
python scripts/check_m2_done.py
python scripts/check_m3_done.py

# M4 deliverables
python scripts/m4_run_dycore.py
python scripts/m4_run_validation.py
python scripts/m4_m5_gate_dryrun.py
python scripts/m4_hlo_diff.py
ls -l artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff   # MUST be size 0

python -m json.tool artifacts/m4/dycore_profile.json
python -m json.tool artifacts/m4/transfer_audit.json
python -m json.tool artifacts/m4/spacetime_budget.json
python -m json.tool artifacts/m4/tier1_advection_parity.json
python -m json.tool artifacts/m4/tier2_invariants.json
python -m json.tool artifacts/m4/tier3_convergence.json
python -m json.tool artifacts/m4/m5_gate_dryrun.json

pytest -q

# Final oracle
python scripts/check_m4_done.py
```

## Performance Metrics

Captured in `artifacts/m4/spacetime_budget.json` and `artifacts/m4/dycore_profile.json`. Hard bounds:
- `host_to_device_bytes_post_init == 0` (constitutional).
- `device_to_host_bytes_post_init == 0`.
- `temporary_bytes_per_step == 0`.
- HLO debug-vs-stripped diff = 0 bytes (constitutional debuggability gate).

Soft bounds for sanity:
- `wall_time_per_step_us < 5000` (≤ 5 ms/step) on `(nz=40, ny=80, nx=80)` — M5 optimizes; this just shouldn't be pathological.
- `total_persistent_bytes < 4 GB` on the same config (well under 32 GB VRAM).

Reporting-only (the M5 gate dry-run):
- `kernel_launches_per_step` (target ≤ 10; trip → opens per-scheme Triton ADR, NOT a sprint failure).
- `local_memory_bytes_per_kernel` (target ≤ 256; same handling).
- `registers_per_kernel` (target ≤ 128; same handling).

## Proof Object

- Diff (File Ownership only).
- 11 artifacts in `artifacts/m4/` (incl. the empty-by-construction HLO diff).
- ADR-003 draft (manager finalizes; Codex critical-reviews separately; user approves).
- Lifecycle reports including Spacetime Budget table + Allocation Audit + HLO Identity Verification + reviewer's per-line attestation.

## Risks

- **HLO debug-vs-stripped diff non-empty.** Most likely cause: tracer accidentally sees `enabled` as a traced value (e.g. passed as `jnp.bool_(False)` not Python `False`). Fix: rigorously ensure `enabled` is a Python bool routed through Python branching, not a traced JAX value.
- **5th-order upwind horizontal advection register-spills on Blackwell.** Possible. If M5 gate trips on registers, manager opens the Triton-fallback ADR per ADR-001; not a sprint failure.
- **Density current at fp64 too slow to run 100 steps in sprint time.** Worker may reduce to (nz=20, ny=40, nx=40) for tier-2 if needed; document.
- **`jax.lax.scan` carry shape mismatch.** When snapshots are enabled, the carry includes the ring buffer; with `enabled=False`, the carry does NOT include it. Two separate compiled programs (already required by `static_argname`). Don't try to share a single JIT cache entry across both.
- **fp64 on RTX 5090** is 1:64 throughput vs fp32. The M4 dycore is fp64 by ADR-002; performance is NOT claimed; ADR-003 starts the conversation about validated downcasts.
- **Tier-1 operator mismatch with `analytic-stencil-3d-advdiff-v1`.** If `phi_next` was generated by a non-upwind scheme (e.g. central differences + explicit diffusion), the dycore's 5th-order upwind scheme will not bit-reproduce. Worker reads `generate_analytic_fixtures.py`, documents, and either matches the operator OR generates a sibling fixture (manager pre-approves).

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m4-dycore-rk3-advection-acoustic`.
- After reviewer Accept, manager:
  - finalizes ADR-003 body
  - dispatches Codex critical-review on ADR-003 in `.agent/decisions/REVIEW-codex-ADR-003/`
  - applies findings
  - commits + merges branch to main, pushes
  - presents user-approval status report for ADR-003
  - writes memory-patch entry summarizing the M4 dycore API + debug-hook contract for future post-compaction Claude turns
  - writes `MILESTONE-M4-CLOSEOUT.md`
- Tester is **Claude Opus 4.7 xhigh** — explicitly tasked with HLO Identity Verification per AC #10.

## Manager-during-worker hygiene reminder

No manager commits while worker is in flight. The codex worker shares the working tree; my commits land on its branch. Stay disciplined. (M3 lesson — held; do not regress.)
