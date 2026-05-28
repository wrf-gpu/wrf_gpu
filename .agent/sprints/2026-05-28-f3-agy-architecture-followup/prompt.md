# Gemini agy — Post-M11.3 Architecture Follow-up

**Worker**: gemini agy
**Wall-time**: 2-4 hours (analysis only)
**No code changes.**

## Context update since your last review

Your previous review (`.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md`)
identified 3 coordinated dycore bugs (advection deleted, mu=mu_delta wrong,
theta decouple wrong reference) and the JAX-vs-JAX self-compare tautology.
Manager source-inspected and verified all 4 findings.

The M11.3 worker applied ALL THREE coordinated fixes (your section "3.A.1-3"):
- Restored advection in `_rk_scan_step` of operational_mode.py
- Fixed `mu=advanced["mu"]` in acoustic.py:261
- Fixed `state.theta_1` in `_decouple_theta_after_advance` at acoustic.py:188

**Result: REGRESSION, not improvement.**
- Baseline (pre-fix): first nonfinite at step 93 (24h pipeline reached step 93 before blowing up)
- After coordinated fix: first nonfinite at step 12 (much earlier)
- Limiter mass residual: Infinity (same as before, no improvement)
- Diagnostic harness now shows NaN propagating from dycore_rk3 into surface fluxes (qke, fltv, qv_flux, theta_flux, tau_u, tau_v, ustar) within first hour

So your hypothesis was directionally right (the 3 bugs ARE bugs), but applying them together exposes a deeper structural inconsistency that single-line fixes can't resolve.

## What I need from you (in this priority order)

### A. Is M11.3 salvageable with a 4th fix?

Read these files NOW with care, focusing on the interactions between the 3 fixes:

1. `src/gpuwrf/runtime/operational_mode.py` — `_rk_scan_step` (~lines 588-617 if unchanged) — see how advection now gets called and how it interacts with the pressure gradient
2. `src/gpuwrf/dynamics/core/acoustic.py` — `acoustic_substep_core` (~lines 200-290), `_decouple_theta_after_advance` (~lines 180-200)
3. `proofs/m11p3/diagnostic_report_after_fix.json` — where the new failure starts
4. `proofs/m11p3/limiter_diagnostics_24h.json` — what the limiter is now seeing
5. `src/gpuwrf/dynamics/mu_t_advance.py` — mu_tendency, theta_tendency interactions

Hypothesis to verify or refute:
- Is the advection tendency being applied to the WRONG state copy? (e.g., applied to `state.theta` but `_decouple_theta_after_advance` reads `state.theta_1`)
- Is `mu` perturbation now growing without bound because the M11.1 p/ph refresh path doesn't account for the new advection contribution?
- Is the RK3 stage cadence still wrong (we apply physics+advection together as a constant tendency but WRF re-evaluates advection per RK stage)?

### B. Worst-case fallback evaluation

From your previous review section 4: "restructure operational driver to match WRF's integrated RK3 cadence" — name the WRF source file ranges (paths under hypothetical `/home/enric/src/wrf_gpu/Registry/` or `dyn_em/` references that GPT can hunt for in our reference docs) that show the integrated RK3 cadence we'd need to mirror.

Estimate:
- Scope (lines of code in operational_mode.py + acoustic.py + new mu_t_advance.py refactor)
- Wall-time at GPT-5.5 xhigh competence with M11/M11.2/M11.3 lessons learned
- Risk (will this also expose deeper bugs? what's the next-layer-down failure mode?)

### C. Cheaper alternative: restart from stencil-bakeoff?

The original M2-S1 stencil-bakeoff produced clean WRF-faithful kernels. Was the operational mode's `_rk_scan_step` always a parallel reimplementation that drifted from the stencil-bakeoff kernels? Or is the stencil-bakeoff already calling into the same broken acoustic.py?

If the stencil-bakeoff kernels are clean, propose a restart sprint that builds a NEW `_rk_scan_step` from the stencil-bakeoff kernels and rewires operational mode to use it. This may be cheaper than fixing the current path.

## Deliverable

Write `.agent/sprints/2026-05-28-f3-agy-architecture-followup/findings.md`
with sections A, B, C answered. End with `AGY_REVIEW_COMPLETE`.

## Hard rules

- CPU pinning: `taskset -c 0-3`.
- No model code changes — analysis only.
- No remote push.
- Manager repo only.
- Auto-notify on exit: `tmux send-keys -t 0 "AGENT REPORT: f3-agy DONE exit=$?" Enter`.
