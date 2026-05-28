# Sprint Contract — F7-MEGA: complete WRF-cadence dry dynamical core

**Sprint ID**: `2026-05-29-f7-mega-dry-dycore-rewrite`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7-mega-dry-dycore-rewrite`
**Wall-time**: large single coherent step (the dycore-correctness milestone)
**GPU usage**: YES — `taskset -c 0-3`, JAX must report `cuda:0`.

## Project endpoint (the bar this whole project is held to)

A **real WRF v4 GPU port** that runs **real WRF test fixtures** and produces **near-identical results / near-identical RMSE on all values** vs WRF, **with no shortcuts**, is **highly efficient for GPU architectures**, and shows **massive speedup** on this RTX 5090 workstation. Bitwise tricks, self-comparisons, clamps/limiters that mask physics, and synthetic happy-paths are all forbidden as evidence.

## Binding goal (universal — top of every contract)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24–72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on a ≥30-case seasonal ensemble, while preserving **≥10× speedup** vs 28-rank CPU WRF on this workstation.

## Where we are (honest)

- **F7.A is MERGED** (HEAD `d6824b6`): cross-RK `_1` family carry, `advance_uv_wrf`, loop-entry `calc_p_rho(step=0)`, explicit RK `dts_rk`, WRF-shaped `small_step_prep`/`finish` skeletons. The first pure-dycore critical violation moved from `step1/RK1/sub1` → `step1/RK3/sub8`.
- **F7.B was attempted but its work was lost** (uncommitted in a `/tmp` worktree wiped on restart). A prior codex run had reached `step4/RK3/sub1` with an `advance_w_wrf` flat-rest oracle passing — that is a feasibility signal, not recoverable code.
- The dycore is still **unstable**: `advance_w` / geopotential / pressure are stubs (`_advance_geopotential`, `_diagnose_pressure`, `_ph_tend_increment` in `acoustic.py`); `calc_coef_w` defaults `c2a` to ones; advection is periodic `jnp.roll` upwind, not WRF flux-form mass-coupled; `rk_addtend_dry` is missing; `calc_p_rho(step=iteration)` is missing.

This sprint completes the **dry** dynamical core to WRF cadence so it is correct and stable **physics-off, periodic/no-op boundary**. Physics RK1-bundle cadence, real lateral BC, and moisture/scalar skill are explicitly later milestones.

## Ground truth and the cardinal rule

**WRF Fortran source is ground truth, not this contract.** Before implementing each operator, read the cited WRF source and verify the equations/sign conventions yourself. If this contract and the WRF source disagree, the WRF source wins — note the discrepancy in your report.

Known contract-level WRF fact (a prior contract got this wrong and wasted a sprint):
- **`c2a` is `INTENT(IN)` in `calc_p_rho`.** It is computed **once per RK stage in `small_step_prep`** (`module_small_step_em.F:230-234`) from base+perturbation pressure and inverse density, then consumed by `calc_coef_w`. `calc_p_rho(step=iteration)` **refreshes perturbation pressure / inverse density / divergence-damping pressure memory — it does NOT recompute `c2a`.**

## Required inputs (read in order)

1. `proofs/f5/wrf_cadence_spec.md` — the binding cadence map (12 items, all with WRF `file:line`). This sprint implements items **3, 4, 6, 7, 10, 11** plus `calc_mu_uv_1` and `small_step_finish_wrf` (§2.4). Items 1, 2, 5, 8, 9 (RK descriptors, `_1` family, prep skeleton) landed in F7.A — verify and extend, don't rebuild.
2. `proofs/f7a/audit_summary.md`, `proofs/f7a/invariant_violations.json` — current failure pattern after F7.A.
3. `.agent/sprints/2026-05-28-f7-critic/critique.md` — methodology lessons (`_1` vs `*_save` lifetime; weak proof gates produce misleading partial success).
4. WRF Fortran at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`:
   - `module_small_step_em.F`: `small_step_prep` 125-285, `calc_p_rho` 492-563, `calc_coef_w` 608-649, `advance_uv` 654-942, `advance_mu_t` 969-1171, `advance_w` 1178-1584, `small_step_finish` 364-430.
   - `module_em.F`: `rk_tendency` 580-1641, `rk_addtend_dry` 1711-1782.
   - `module_big_step_utilities_em.F`: `couple`/`decouple` 555-628 / 4791-4845, `calc_mu_uv(_1)` 51-287, map-factor coupling 359-394.
   - `solve_em.F`: RK loop 1447-1483, RK1 physics block 1693-1780, per-stage `rk_tendency`/`rk_addtend_dry` 1837-2143, small-step envelope 2371-2738, acoustic loop 3065-4206, finish 4383-4462.
5. Current JAX (read fully before editing):
   - `src/gpuwrf/dynamics/core/acoustic.py` — `acoustic_substep_core` and the three stubs to replace.
   - `src/gpuwrf/dynamics/core/calc_p_rho.py` (step=0 only), `small_step_prep.py`, `small_step_finish.py`.
   - `src/gpuwrf/dynamics/acoustic_wrf.py` — `calc_coef_w_wrf_coefficients` (the `c2a`-defaults-to-ones path), PGF/coefficient builders.
   - `src/gpuwrf/dynamics/mu_t_advance.py` — `advance_mu_t_wrf` (keep, rewire inputs).
   - `src/gpuwrf/dynamics/advection.py` — periodic upwind to replace with flux-form mass-coupled WS5/3.
   - `src/gpuwrf/dynamics/tridiag_solve.py` — Thomas solver to wrap with the real `c2a` matrix.
   - `src/gpuwrf/runtime/operational_mode.py` — `_rk_scan_step`, `_acoustic_scan`, RK stage descriptors, save-family wiring.
   - `scripts/f6_transaction_audit.py` — the gate harness (how invariants are computed).
   - `src/gpuwrf/ic_generators/idealized.py` — `run_density_current_case`, `run_warm_bubble_case` (the physics-truth gate).

## Scope — implement, in this order, committing after each block

Work incrementally. Run the F6 12-step audit after each block so a regression is caught at its source, not at the end.

1. **`calc_coef_w` with real `c2a`** (item 7). Compute `c2a = cpovcv*(pb+p)/alt` once per RK stage in `small_step_prep` outputs; pass it into `calc_coef_w` instead of the ones-default. Verify the vertical implicit coefficients (`a`, `alpha`, `gamma`) against WRF `module_small_step_em.F:608-649`.
2. **`calc_p_rho` cadence** (item 6). Keep `step=0` (F7.A). Add the `step=iteration` refresh of perturbation pressure / inverse density / divergence-damping pressure memory, called after `advance_w` each substep. `c2a` is INTENT(IN) — do not recompute it here.
3. **`advance_w_wrf`** (item 10). Replace `w_solve_core` RHS + `_advance_geopotential` + `_ph_tend_increment` with WRF-faithful implicit w + geopotential: full RHS (large-step `rw_tend`, vertical PGF perturbation, buoyancy, divergence-damping via `c2a`, terrain lower BC), Thomas solve with the real coefficient matrix, geopotential advanced from the implicit-w solve (`ph_tend`), not a post-hoc 0.01·Δθ stub.
4. **`small_step_prep_wrf` / `small_step_finish_wrf` / `calc_mu_uv_1`** (§2.4 + items 5). Correct `_1` vs `*_save` lifetimes (RK1 copies `_2`→`_1`; every stage saves `_2`→`*_save`, builds coupled work arrays, `muts`/`muus`/`muvs`/`mu_save`/`ww_save`/`c2a`; finish decouples and restores `mu_2 += mu_save`, `ph`, `ww`). `calc_mu_uv_1` recomputes face masses from `muts` after substeps.
5. **WRF flux-form mass-coupled advection** (item 3, `rk_tendency` dry dynamics). Replace periodic `jnp.roll` upwind with flux-form WS5 (horizontal) / WS3 (vertical) mass-coupled advection with map factors, plus the dry stage tendencies (`rhs_ph`, buoyancy, damping). This is what actually moves the warm bubble / density current.
6. **`rk_addtend_dry`** (item 4). Per-RK-stage merge of (RK1-fixed physics tendencies = zero when physics-off) + per-stage dry dynamics tendencies into `ru/rv/rw/t/ph/mu` with field-specific map/mass coupling. Wire RK stage descriptors (`dt_rk`, `dts_rk`, `number_of_small_timesteps` = 1, n/2, n).
7. **`sumflux` accumulators** (item 11) — add `ru_m`/`rv_m`/`ww_m` to the carry and accumulate after `advance_w`. (Scalar tendency pass itself is physics/moisture → may stay a no-op stub here, but the accumulators must exist for later scalar work; document.)

Boundary cadence (item 12) stays **no-op/periodic** for these physics-off idealized + audit gates — real specified/nested BC is M14. Add the hook points where WRF applies BCs, wired to no-op for periodic.

## Acceptance gates (falsifiable — all required for F7_MEGA_COMPLETE)

- **AC1 — 12-step transaction audit clean.** `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f7mega` on combination `a` shows **no pure-dycore critical violation** through all 12 steps: `theta_in_bounds`, `pressure_bounded`, dry-mass non-negativity, finiteness all hold; `muts_mut_work_mu_consistency` clear; acoustic u/v deltas physically bounded (no exponential blow-up). If a strict algebraic invariant (e.g. theta-mass residual) is intentionally tighter than WRF, document why and show it is bounded, not detonating.
- **AC2 — Straka density current RAN_TO_COMPLETION near reference.** `run_density_current_case(require_gpu=True)` returns `status=RAN_TO_COMPLETION`, `verdict=PASS`, with `front_position_900s`, `theta_prime_min_900s`, `max_abs_w_900s` within a documented tolerance of the Straka et al. (1993) reference (front ≈ 14–16 km at 900 s, min θ′ ≈ −9 to −10 K order), and `relative_mass_drift` small (≤ 1e-3). Compare to the references cited in `proofs/f2/straka_density_current_verdict.md`.
- **AC3 — Skamarock warm bubble RAN_TO_COMPLETION, physically plausible.** `run_warm_bubble_case(require_gpu=True)` returns `RAN_TO_COMPLETION` with a symmetric, bounded rising thermal (no NaN, no runaway), matching the qualitative reference.
- **AC4 — flat-rest oracles PASS.** A hydrostatically-balanced rest state stays at rest: `advance_uv_wrf`, `advance_w_wrf`, and a full substep produce **zero** (≤ machine-epsilon) tendency on u/v/w/ph/θ. (This is the single strongest guard against sign/coupling errors and against green self-compares.)
- **AC5 — stability over hundreds of steps.** A pure-dycore integration of ≥ 300 steps (or the full Straka 900 s) stays finite and bounded with no clamp/limiter engaged.
- **AC6 — existing tests still pass.** The 3 F6 regression unit tests and the dynamics-core unit tests still pass; no test deleted, no tolerance widened, no `xfail` added (INV-6).

## Proof objects (write all)

- `proofs/f7mega/audit_combination_{a,b,c,d}.json`, `invariant_violations.json`, `audit_summary.md` (12-step audit).
- `proofs/f7mega/straka_density_current.json` + verdict, `proofs/f7mega/skamarock_warm_bubble.json` + verdict (+ the .ppm/plot artifacts the harness emits).
- `proofs/f7mega/flat_rest_oracle.json` (AC4 deltas).
- `proofs/f7mega/stability_long_run.json` (AC5).
- `proofs/f7mega/regression_diff.md` (before/after the F6 failure pattern).
- `worker-report.md` in the sprint folder using the AGENTS.md handoff format, ending with verdict `F7_MEGA_COMPLETE` or `F7_MEGA_PARTIAL` + explicit remaining gaps.

## Hard rules

1. `taskset -c 0-3` on every python/pytest invocation; confirm `cuda:0` first.
2. WRF source is ground truth; cite `file:line` in every new operator's docstring.
3. **No clamps, caps, tanh sanitizers, or positive-definite limiters** added to make a gate pass. The dycore must be stable on its own. If a limiter currently fires, the fix is the operator, not a wider clamp.
4. **No performance optimization** in this sprint (no fp32 downcast, no fusion refactor for speed) — correctness first; perf is the separate F7-perf sprint. Keep fp64 (`jax_enable_x64`).
5. Commit incrementally on branch `worker/opus/f7-mega-dry-dycore-rewrite` with clear messages; do not push to any remote.
6. Files writable: everything under `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/runtime/operational_state.py`, `src/gpuwrf/ic_generators/idealized.py` (only if a harness bug blocks the run — fix minimally and document), `scripts/f6_transaction_audit.py` (only to instrument new operators, not to weaken invariants), `tests/**` (add tests, never weaken), `proofs/f7mega/**`, the sprint folder.
7. Files NOT writable: governance (`PROJECT_CONSTITUTION.md`, `AGENTS.md`, `MILESTONES.md`, plan, ADRs), memory, skills, physics scheme code, comparator scripts under `scripts/m6b6_*`.
8. If you discover the scope cannot all land cleanly, deliver the largest correct, gated subset (e.g. through AC1 + AC4 + AC5), mark `F7_MEGA_PARTIAL`, and name precisely what remains and why — an honest partial beats a green self-compare.

## Out of scope (later milestones)

- Physics tendencies moved into the RK1 bundle (cadence item 2) — Phase B once the dry core is correct.
- Real specified/nested lateral boundary forcing (item 12 active) — M14.
- Moisture/scalar tendency skill — M17 / Phase B.
- XLA fusion + fp32 downcast + speedup recertification — F7-perf.
