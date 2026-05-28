# Sprint Contract — F7 Sprint A: WRF acoustic small-step core (cadence-faithful, stable)

**Sprint ID**: `2026-05-29-f7-acoustic-core`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7-acoustic-core` (work directly in the main tree on this branch; commit incrementally)
**GPU**: YES — every python/pytest call under `taskset -c 0-3`; confirm `cuda:0` first.
**Supersedes**: the broader `2026-05-29-f7-mega-dry-dycore-rewrite` contract, split per the GPT-5.5 critic at `.agent/sprints/2026-05-29-f7-mega-dry-dycore-rewrite/critique.md`. This is **Sprint A** (the acoustic small-step core). **Sprint B** (flux-form advection + `rk_addtend_dry` + Straka/warm-bubble) follows.

## Project endpoint (the bar)

A real WRF v4 GPU port that runs real WRF test fixtures with **near-identical results / RMSE on all values, no shortcuts** (no masking clamps, no JAX-vs-JAX self-compares, no synthetic happy-paths), GPU-efficient, with massive speedup on this RTX 5090. This sprint is a prerequisite: the dycore currently detonates by step 4, so it cannot yet be measured against WRF. Sprint A makes the acoustic recurrence WRF-cadence-faithful and stable so that per-operator WRF savepoint parity (M9) becomes possible.

## Cardinal rule

**WRF Fortran source is ground truth, not this contract.** Read the cited WRF source and verify every equation/sign/coupling yourself before implementing. If contract and WRF disagree, WRF wins — note it in your report.

## Scope — the 8 acoustic operators, in WRF order

Implement/repair the small-step acoustic core so one RK stage executes the **exact WRF cadence**:

```
small_step_prep  ->  calc_p_rho(step=0)  ->  calc_coef_w
  ->  small_steps loop { advance_uv -> advance_mu_t -> advance_w -> sumflux -> calc_p_rho(step=iteration) }
  ->  calc_mu_uv_1  ->  small_step_finish
```

Large-step dry tendencies (advection, `rhs_ph`, buoyancy source, `rk_addtend_dry`) are **Sprint B**; for Sprint A, feed the acoustic core with the current/simple large-step tendency path (or zero where a periodic idealized case has none), and prove the *acoustic recurrence itself* is correct and stable. `sumflux` accumulators (`ru_m`, `rv_m`, `ww_m`) must exist and update after `advance_w` so Sprint B's scalar pass can consume them.

### WRF-factual requirements the critic verified (fold ALL of these in)

1. **`small_step_prep` / `small_step_finish` bracket `advance_w`.** `advance_w` consumes coupled/saved `u,v,w,t,ph,ww` produced by prep (`module_small_step_em.F:238-285`); finish decouples/restores them (`:379-430`). So prep/finish lifetimes MUST be correct *before* `advance_w` can be WRF-factual. Correct `_1` vs `_2` vs `*_save` vs coupled-work lifetimes (`:125-215`). On RK1 copy `_2`→`_1`, set `mu_2`=0 work; on RK2/3 `mu_2 = mu_1 - mu_2`; finish restores `mu_2 += mu_save`.
2. **`calc_coef_w` uses full dry mass `mut`, NOT the work array `muts`** (`solve_em.F:2676-2681`, `module_small_step_em.F:570-592`). Coefficients use `mut`, `cqw`, `rdn`, `rdnw`, `c2a`, `dts`, `g`, `epssm` (`:624-649`). **`c2a` comes from `small_step_prep` (`c2a = cpovcv*(pb+p)/alt`, `:230-234`) and is NEVER defaulted to ones** in the production path (current bug: `acoustic_wrf.py:606-630`). **`cqw` must be the real WRF `cqw`** (`:584-588`), consumed by both `calc_coef_w` and `advance_w` (`:637-639`, `:1477-1489`) — not a placeholder.
3. **`calc_p_rho`**: `step=0` sets `pm1 = p` (`:522-528`). Each later substep applies divergence-damping pressure memory `p = p + smdiv*(p - pm1)` then refreshes `pm1` (`:548-567`). `c2a` is `INTENT(IN)` here — do NOT recompute it.
4. **`advance_uv`**: external-mode divergence damping is here via `mudf`/`emdiv` (`:808-810`, `:866-869`, `:879-880`, `:940-942`) — implement it. (Already partly present from F7.A; verify against WRF.)
5. **`advance_w`**: full implicit w + geopotential. RHS includes large-step `rw_tend`, vertical PGF perturbation, buoyancy via `cqw`/`pg_buoy_w`, terrain lower boundary + top-lid behavior (`:1406-1428`, `:1492-1502`), the `c2a` implicit pressure/geopotential terms (`:1477-1489`). `ph` advances from `ph_tend`/the implicit-w solve (`:1341-1502`) — **delete** the stubs `_ph_tend_increment` (0.01·Δθ), `_advance_geopotential`, and `_diagnose_pressure`. **"Divergence damping via c2a in advance_w" was a contract error — c2a is the implicit pressure term, not divergence damping.** Use the existing Thomas solver (`tridiag_solve.py`) with the real coefficient matrix.
6. **`calc_mu_uv_1`** after the loop recomputes face masses `muus`/`muvs` from `muts` (`solve_em.F:4383-4406`, `module_big_step_utilities_em.F:184-321`).
7. **Freeze the config / namelist** for the gates: divergence-damping coefficients `smdiv`/`emdiv` at WRF defaults; Rayleigh damping, `w_damp`, diffusion, Coriolis, curvature, relaxation **disabled** for the periodic idealized/audit gates — and **prove they are inactive** (zero-path), do not silently assume. `epssm` off-centering at WRF default.

## Acceptance gates (all required for `F7A_COMPLETE`)

- **AC1 — no-stub audit.** The transaction audit / a dedicated check FAILS if any legacy approximation is still on the operational acoustic call path: `_ph_tend_increment`, `_advance_geopotential`, `_diagnose_pressure`, or a `w_solve_core` path lacking the WRF `advance_w` RHS (`acoustic.py:349-402`). Prove by assertion/grep-in-test that these are gone from the path.
- **AC2 — 12-step transaction audit clean, ALL combinations.** `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f7a2` shows `first_critical_violation == null` for **every** combination a/b/c/d (pressure_bounded, muts_mut_work_mu_consistency, theta bounds, dry-mass non-negativity, finiteness all hold through 12 steps). No clamp/limiter engaged.
- **AC3 — flat-rest oracle = 0 (smoke).** A hydrostatically-balanced rest state stays at rest: each acoustic operator and a full substep produce ≤ machine-epsilon tendency on u/v/w/ph/θ/p. Kept as a smoke test, not the primary physics proof.
- **AC4 — nonzero analytic acoustic oracle (primary physics proof).** Construct at least one nonzero analytic test whose sign/response is known and that exercises the implicit vertical solve, pressure-memory refresh, and coupling — e.g. a single hydrostatic-adjustment column or a vertically-propagating acoustic/gravity mode in a periodic box — and verify the JAX acoustic core's response matches the analytic expectation in sign and order of magnitude. Document the analytic derivation. This is the gate that flat-rest cannot provide.
- **AC5 — conservation.** Over a ≥300-step pure-acoustic periodic integration: dry-mass relative drift ≤ 1e-6 and theta-mass (column-integrated) relative drift bounded; report both. State stays finite/bounded with no clamp engaged.
- **AC6 — existing tests pass, nothing weakened.** The 3 F6 regression unit tests + dynamics-core unit tests pass; no test deleted, no tolerance widened, no `xfail` added (INV-6).

## Proof objects (write all into `proofs/f7a2/`)

`audit_combination_{a,b,c,d}.json`, `invariant_violations.json`, `audit_summary.md`; `no_stub_audit.json`; `flat_rest_oracle.json`; `analytic_acoustic_oracle.json` + a short markdown deriving the analytic expectation; `conservation_long_run.json`; `regression_diff.md` (before/after the step-4 failure pattern). Plus `worker-report.md` (AGENTS.md handoff format) ending with `F7A_COMPLETE` or `F7A_PARTIAL` + precise remaining gaps.

## Hard rules

1. `taskset -c 0-3`; confirm `cuda:0`. Keep `jax_enable_x64=True`.
2. WRF source is ground truth; cite `file:line` in every new/changed operator docstring.
3. **No clamps, caps, tanh sanitizers, or positive-definite limiters** added to pass a gate. If a limiter currently fires, fix the operator, not the clamp.
4. **No performance work** (no fp32 downcast, no fusion-for-speed). Correctness first; perf is the separate F7-perf sprint.
5. Commit incrementally on `worker/opus/f7-acoustic-core`; do not push to any remote.
6. Files writable: `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/runtime/operational_state.py`, `scripts/f6_transaction_audit.py` (only to instrument/strengthen, never to weaken invariants), `tests/**` (add only), `proofs/f7a2/**`, this sprint folder.
7. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, `dynamics/advection.py` large-step rewrite (that is Sprint B — but you MAY read it and you MAY add the `ru_m/rv_m/ww_m` accumulators to the carry).
8. If the full scope cannot land cleanly, deliver the largest gated subset (at minimum AC1+AC2+AC3+AC5) and mark `F7A_PARTIAL` with precise gaps. An honest partial beats a green self-compare.

## Forward pointers (NOT this sprint)

- **Sprint B**: WRF flux-form mass-coupled advection frozen at h-order 5 / v-order 3 (a scoped restriction; WRF order is config-driven), `rhs_ph`, `pg_buoy_w` large-step source, `rk_addtend_dry` (field-specific map/mass coupling, not a generic add), Straka density-current + Skamarock warm-bubble vs published references, theta scalar advection.
- **M9**: build instrumented WRF Fortran savepoints, then per-operator WRF↔JAX parity for `small_step_prep`, `calc_p_rho(0)`, `calc_coef_w`, post-`advance_uv`, post-`advance_mu_t`, post-`advance_w`, post-`calc_p_rho(iter)`, post-`calc_mu_uv_1`, post-`small_step_finish` over fields u,v,w,theta,ph,mu,p,al,rho,pm1,mudf,ww,ru_m,rv_m,ww_m. This is the rigorous near-identical-RMSE-vs-real-WRF gate.
