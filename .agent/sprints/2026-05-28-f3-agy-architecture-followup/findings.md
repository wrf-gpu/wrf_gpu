# F3 Architecture Follow-up Findings (Opus Plan subagent)

**Worker**: Opus 4.7 Plan subagent (replaced agy after model-tier policy update)
**Date**: 2026-05-28
**Mode**: Read-only architectural review

---

## A. Is M11.3 salvageable with a 4th fix?

**No, not with a single fix.** M11.3 closed three local bugs but exposed a structural gap that single-line patches cannot fill. Three concrete defects survive in the operational path after M11.3:

### A.1 — The acoustic core has no `advance_uv` or `advance_w` momentum/W solve

The shared kernel `acoustic_substep_core` (`src/gpuwrf/dynamics/core/acoustic.py:218-272`) only runs `advance_mu_t_core` (mass + theta + ww) plus a Thomas back-substitute on `w` (`w_solve_core`) using coefficients that the operational driver builds from the OLD `muts`. The WRF small step order at `solve_em.F:3088 / 3398 / advance_w` is `advance_uv → advance_mu_t → advance_w` — three solves, all coupled, in the same iteration. The JAX core is missing `advance_uv` entirely: `u, v` are never updated inside the small-step loop. After M11.3 restored the RK-stage horizontal pressure gradient, the result is a momentum field that drifts in only the *outer* RK predictor while mass/theta/omega keep updating per substep — which is exactly the mass-momentum incoherence that detonates `mu_nonnegative` at step 11 and `theta_in_bounds` at step 12 in `proofs/m11p3/diagnostic_report_after_fix.json`.

### A.2 — `_with_save_family` clobbers the M11.3 `mu_save` fix

In `operational_mode.py:395-411`, after each RK predictor stage builds the candidate state, `_with_save_family` is called with `muts=mu_base` and `mu_save=state.mu_perturbation`. Tracing through the substep, this happens to give the correct `mu_save = mu_perturbation - 0` on substep #1, but it also throws away the previous-RK-stage `mu_save` (the column-mass perturbation needed by WRF when `rk_step > 1` per `module_small_step_em.F:213`). RK2 and RK3 must reuse the RK1 `mu_save`; the current code rebuilds it from `mu_perturbation` every stage. So the M11.3 acoustic-loop fix only holds *within* one stage; across stages, `mu_save` collapses again.

### A.3 — Pressure/geopotential refresh is one line

`_diagnose_pressure` (acoustic.py:211-215) advances perturbation `p` as `state.p + |dnw| * (mu_new - mu_old)`. WRF's actual small step (in `advance_w`) propagates `p` via the implicit vertical acoustic update with `c2a = cpovcv*(pb+p)/alt` and the divergent vertical mass flux — not a flat dnw-weighted mass increment. After M11.3 the mu_perturbation drift is amplified by this stub each substep.

**Verdict on A**: No single-line fix exists. The closest "minimum unblocking surgery" is adding both `advance_uv` and a proper `advance_w` (with acoustic `p` propagation) to the small-step loop and reading `mu_save` from the parent RK carry instead of rebuilding it. That is structurally a **200-400 LOC refactor**, not a 4th fix.

---

## B. Worst-case fallback — integrated RK3 cadence

WRF's cadence is unambiguous in the Fortran reference at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`:

- `solve_em.F:1447` — `Runge_Kutta_loop: DO rk_step = 1, rk_order`
- `solve_em.F:1472-1483` — sets `number_of_small_timesteps` per stage (1 / num/2 / num)
- `solve_em.F:1693-1830` — `first_rk_step_part1/part2` (physics tendencies, computed once on RK1, carried as constants)
- `solve_em.F:1848` — `rk_tendency` (per-stage advection: u/v/w/ph/theta) followed by `rk_addtend_dry` at line 2130
- `solve_em.F:2544 / 2605` — `small_step_prep` mass coupling
- `solve_em.F:3065` — `small_steps : DO iteration = 1, number_of_small_timesteps`
- `solve_em.F:3088, 3398, 3437-3454` — `advance_uv`, `advance_mu_t`, `advance_w` (in that order, inside each small step)
- `solve_em.F:4363` — small step loop close
- `solve_em.F:5060-6108` — scalar advection via `rk_scalar_tend`
- `solve_em.F:6765` — RK loop close

The JAX side currently has `rk_tendency = compute_advection_tendencies` (which is a *periodic-fixture* upwind operator using `jnp.roll`, `dynamics/advection.py:181-238`, not WRF flux-form mass-coupled advection with map factors), no `advance_uv` / `advance_w`, no `rk_addtend_dry`, no `rk_scalar_tend`, and physics applied *after* the RK3 loop instead of as RK1 constants.

**Scope estimate**: rewrite operational_mode.py `_rk_scan_step` (~150 LOC), add advance_uv (~120 LOC) and a proper advance_w with implicit p propagation (~200 LOC) in acoustic.py, replace advection.py's Wicker-Skamarock-on-periodic-domain with flux-form mass-coupled WS5/3 with map factors (~400 LOC), add `rk_addtend_dry` (~80 LOC). **Total: ~950-1100 LOC of new model code + tests.**

**Wall-time**: at GPT-5.5 xhigh, **4-6 calendar days for code, plus 2-3 days for *real* Fortran-savepoint validation** (the existing 100-step "parity" test is a tautology).

**Risk (next-layer-down failure)**: the moisture coupling in `horizontal_pressure_gradient` (acoustic_wrf.py:215-232) and the M11.1 `p/ph` refresh need to be re-verified; the lateral-boundary apply path is also producing `mu_perturbation` deltas of 1.6e15 (`proofs/m11p3/diagnostic_report_after_fix.json` lateral_boundary section) which suggests `apply_lateral_boundaries` itself reads pre-bounded mu — that becomes the next blocker once the dycore stops detonating.

---

## C. Restart from stencil-bakeoff?

The stencil-bakeoff artifacts under `data/scratch/m2-{jax,triton,cupy}/` and `data/scratch/{cuda_tile,kokkos}/` are leaf-stencil correctness payloads (single-kernel `.npz` outputs), not a fully composed dycore. No `gpuwrf/dynamics/**.py` imports them. The currently-broken acoustic core (`core/acoustic.py`) is shared by both the operational path AND the M6B5 `dycore_timestep_core` validation path (`core/dycore.py:71-77` calls `acoustic_scan_core`).

The validation harness `dycore_timestep_core` is *itself* just RK3 wrapped around the same acoustic-only kernel — no advection, no advance_uv, no advance_w. So "the validation passes" because validation is JAX-vs-JAX-self-compare on a kernel that omits the same physics the operational path omits. **The stencil-bakeoff is not a viable restart base** for a working dycore; it never demonstrated a closed-loop integration.

There IS one half-clean path: the legacy `dynamics/rk3.py` + `dynamics/acoustic.py:forward_backward_acoustic` (used by `step.py`) is a pre-WRF educational RK3+forward-backward acoustic that was the M4 baseline. It's simpler, runs as a closed loop, but is not WRF-faithful and would not pass TOST against real WRF output. Not recommended as a restart, but it's the only thing in the tree that currently integrates without blowing up.

---

## D. Recommendation

**(b) Multi-line refactor with named scope.**

Specifically: **abandon single-line repair on the current `_rk_scan_step` path** and execute a scoped rewrite, NOT a full architectural rewrite:

1. **Keep**: `acoustic_wrf.py` (PGF + coefficient builders), `mu_t_advance.py` (advance_mu_t numerics — the WRF-shaped one), the `OperationalCarry` + `_with_save_family` plumbing, the lateral boundary and physics adapters.
2. **Discard / rewrite**:
   - `core/acoustic.py:acoustic_substep_core` (add `advance_uv` and proper `advance_w` w/ acoustic-p)
   - `dynamics/advection.py:compute_advection_tendencies` (replace periodic-domain Wicker-Skamarock with flux-form mass-coupled WRF advection)
   - `_rk_scan_step` (mirror WRF cadence: physics tendencies fixed at RK1, advection per-stage, mu_save carried across stages)
3. **Repair non-optional**: Rewrite `scripts/m6b6_coupled_step_compare.py` to read pre-computed Fortran savepoints instead of `emit_tier`-then-read. Every claim of dycore correctness made before this fix is uncited.

**Confidence**: HIGH on A (M11.3 is not single-fix salvageable), MEDIUM-HIGH on the cause being missing advance_uv/advance_w + cross-stage mu_save loss, HIGH on C (stencil-bakeoff cannot be a restart base), MEDIUM on the LOC estimate (could be 50% higher once map-factor coupling lands).

### Critical Files for Implementation

- `src/gpuwrf/runtime/operational_mode.py` — rewrite `_rk_scan_step`, `_with_save_family`, RK-stage mu_save carry
- `src/gpuwrf/dynamics/core/acoustic.py` — add advance_uv + advance_w into `acoustic_substep_core`
- `src/gpuwrf/dynamics/advection.py` — replace periodic WS5/3 with WRF flux-form mass-coupled advection w/ map factors
- `src/gpuwrf/dynamics/mu_t_advance.py` — verify uniform sign conventions once advance_uv lands
- `scripts/m6b6_coupled_step_compare.py` — replace JAX-self-compare tautology with real Fortran-savepoint oracle

ARCH_REVIEW_COMPLETE
