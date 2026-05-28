# Gemini agy — Dycore Architecture Deep Review

You are a senior NWP/dycore architect. GPT-5.5 has now FAILED TWICE on the same defect class. Codex GPT-5.5 M11 sprint worked around it with an aggressive limiter (clip every step, 315k cells max), and Codex GPT-5.5 M11.2 sprint tried 3 candidate single-line fixes and ALL REGRESSED. Per project methodology (memory feedback-gemini-authed-use-sparingly), this is precisely the trigger for engaging you.

You have one job: read the evidence below and tell us what GPT-5.5 missed. Concrete next-fix-action recommended. Or — if the dycore is structurally beyond local fixes — say so and recommend the rewrite scope.

# The defect class

JAX dycore (`src/gpuwrf/dynamics/`) of a WRF v4 port produces nonphysical theta and wind increments every timestep. Symptoms across sprints:

- **M11 worker** (positive-definite limiter): added `_limit_guarded_dynamics_state`, clips theta to physical bounds every step. Limiter active 8640/8640 steps, max 315 351 cells clipped in one step, mass residual 0.027 K. theta 24 h RMSE dropped 77.4 → 7.5 K = -90.3 % BUT only because the limiter does all the work. T2 station RMSE regressed 10.80 → 13.11 K.

- **M11.1 worker** (p_perturbation + ph_perturbation): found dycore `p_perturbation`/`ph_perturbation` silently flatlining. Fixed in `src/gpuwrf/dynamics/core/acoustic.py` — they now advance correctly. Harness `dycore_rk3` verdict = ACTIVE. **This part works.**

- **M11.2 worker** (theta-increment root cause): three candidate fixes tested, all regressed:
  1. `_decouple_theta_after_advance` numerator `state.theta` → `state.theta_1` (per WRF `module_small_step_em.F:408-413` reconstructing `t_2` with saved `t_save`). Failed: regressed wind_in_bounds to step 72, nonfinite at step 93.
  2. `mu_tendency` sign flip `-dmdt + mu_tend` → `dmdt + mu_tend` (per WRF `module_small_step_em.F:1099-1105`). Failed: limiter clipped all 360 first-hour steps, mass residual exploded.
  3. theta horizontal flux mass-coupled vs perturbation-only (per WRF `:238-254` and `:1162-1171`). Failed: earlier wind/mass blow-up or limiter still saturated.

After all merges (M11 + M11.1 + M14 + harness shim), the harness 1h smoke shows:
- first invariant break: `wind_in_bounds` at step 72, operator `dycore_rk3`
- `theta_in_bounds` first violates at step 85, operator `dycore_rk3`
- first nonfinite: step 93, operator `dycore_rk3`
- 24h limiter diagnostic mass residual = **Infinity**

100-step parity test against the WRF B6 savepoint still PASSES (439 s). So at the dycore-in-isolation level the operators are bitwise-correct vs WRF; only on the operational coupled path (with physics, BC, lateral relaxation) do they explode.

# Files to read (in this order)

1. `.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/root_cause_analysis.md` — M11.2 worker's 3 candidates + WRF reference lines they checked
2. `.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/worker-report.md` — full diagnostic table
3. `proofs/m11p2/diagnostic_report_after_fix.json` — operator verdicts + first-invariant-break per scheme
4. `proofs/m11p2/limiter_diagnostics_24h.json` — 8640/8640 + Infinity residual
5. `src/gpuwrf/dynamics/core/acoustic.py` — full file; this is where M11.1 added the p/ph refresh; this is also where m11p2 tried `_decouple_theta_after_advance` change
6. `src/gpuwrf/dynamics/mu_t_advance.py` — `mu_tendency` + theta horizontal flux divergence
7. `src/gpuwrf/dynamics/core/small_step.py` (or equivalent) — the small-step driver
8. `src/gpuwrf/runtime/operational_mode.py` — `_physics_boundary_step`, `_limit_guarded_dynamics_state`, `_positive_definite_theta_increment_limiter`
9. `tests/savepoint/test_dycore_100_steps.py` — the validation-mode comparator that DOES pass

# What I need from you

A `.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md` file in this repo (write it, then commit). Structure:

## 1. Root-cause hypothesis

Identify the architectural defect that explains:
- Why the dycore is bitwise-correct on 100 isolated steps (validation mode) but explodes on coupled operational mode
- Why every single-line fix attempted regresses
- Why limiter activity is 8640/8640 with Infinity residual

Hypothesize a single concrete root cause that explains all four observations.

## 2. Why GPT-5.5 missed it

Pinpoint the specific reasoning step where the two GPT workers went wrong.

## 3. Recommended next sprint

ONE OF:
- (a) Specific file:line edit with WRF reference, with high confidence it will pass AC3+AC4
- (b) Multi-line refactor scope, naming the function or module
- (c) Architectural rewrite scope for `src/gpuwrf/dynamics/`, naming what to keep and what to discard

Give concrete acceptance criteria for the recommended sprint.

## 4. Worst-case fallback

If your recommendation in (3) fails, what's the next move? Be honest about whether the project can still hit Canary L2/L3 TOST equivalence with the current dycore architecture, or whether a deeper rewrite is needed.

## End with: `AGY_REVIEW_COMPLETE` or `AGY_REVIEW_PARTIAL`.

# Hard constraints

- CPU pinning: prefix every shell command with `taskset -c 0-3`.
- GPU usage: not required for this review; pure reading + reasoning.
- No model code changes — analysis only.
- No remote push.
- Manager repo only.
- Auto-notify on exit: end with `tmux send-keys -t 0 "AGENT REPORT: agy-dycore-review DONE exit=$?" Enter`.
