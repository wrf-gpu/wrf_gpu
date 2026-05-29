You are GPT-5.5 xhigh running an INDEPENDENT, from-scratch BUG HUNT on the wrf_gpu2 dynamical core (JAX GPU port of WRF v4, mass/hydrostatic-pressure vertical coordinate, C-grid, RK3 + acoustic small-step split-explicit). An Opus frontrunner is fixing it in parallel on a different angle; you are the second, independent angle. Your job is to find the TRUE cause of a persistent linear `w` runaway in idealized cases — do NOT anchor on any single prior hypothesis; several have already failed.

This is a READ + ANALYZE + RUN-DIAGNOSTICS task on a STABLE snapshot. You are in a detached worktree at `/tmp/wrf_gpu2_bughunt` (commit 4ce8f07). You MAY read any file and RUN read-only diagnostics/probes here (`PYTHONPATH=src taskset -c 0-3 python ...`, cuda:0, fp64). Do NOT edit the dynamical-core source to "fix" it (the Opus frontrunner owns the fix). Your ONLY persistent write is the findings file named at the end (an ABSOLUTE path in the main repo, so it survives).

## The symptom (warm bubble = Skamarock; Straka = density current)
- Warm bubble: `max|w|` grows ~LINEARLY (≈ +6 m/s per 10 s), `max|u|` ~linearly, θ′max pinned ~2.0 K, mass conserved to ~0, `ph_perturbation` reportedly FROZEN at 131.83, → NaN ~190 s. Straka → NaN ~40 s.
- Controls already established (trust these): a θ′=0 hydrostatically-balanced base column is EXACTLY stable (machine-zero tendencies, mass conserved 300+ steps) — so the integrator + base-state balance are sound. epssm 0.1 vs 0.5 → identical runaway (constant forcing, NOT an eigenmode). Explicit diffusion up to ν=200 does not damp it. The IC perturbation geopotential `ph'` round-trips through the dycore's own `calc_p_rho_phi` to 1e-12 under BOTH sign conventions (so it is NOT a sign or IC-hydrostatic-balance bug).

## What has ALREADY been fixed (verified — do not re-litigate, but they did NOT stop the runaway)
WRF acoustic cadence, w_damping/Rayleigh/divergence damping, flux-form WS5/3 advection (5th-order verified), MUT/MUTS total-mass semantics, large-step PGF + rk_addtend_dry (flow circulates), calc_p_rho_phi geopotential term restored, signed-metric view, once-per-stage pg_buoy_w + work-variable t_2ave staging.

## Your charter — find the cause, independently
A linear (constant-rate) runaway with mass conserved and θ′ pinned means a CONSTANT net force with NO restoring feedback. Investigate the full vertical w/ph/p restoring loop and anything else, from scratch:
1. Is the vertical implicit `advance_w` solve actually closing the w↔ph↔p restoring loop each substep? Read `src/gpuwrf/dynamics/core/advance_w.py`, `core/acoustic.py` (`acoustic_substep_core`, the lax.scan carry), `acoustic_wrf.py` (`calc_coef_w`, `diagnose_pressure_al_alt`), `core/calc_p_rho.py`, `tridiag_solve.py`. Does the solved `w` actually update `ph` (`module_small_step_em.F:1581-1586`), and does that updated `ph` feed back into the next substep's pressure (`calc_p_rho_step`) and next stage's PGF? RUN a probe that traces `max|w|`, `max|ph_work|`, `max|ph_perturbation|`, `max|p_perturbation|` per substep for a warm-bubble step — confirm or REFUTE that ph is frozen.
2. If ph IS frozen: where exactly is the carry/update dropped (scan carry not returning ph_next? prep seeding ph_work wrong? finish not adding the evolved work back?)? Pinpoint file:line.
3. If ph is NOT frozen: then the restoring loop runs but still doesn't balance — look elsewhere: is `calc_coef_w`'s implicit coefficient matrix (a/alpha/gamma, the c2a/cqw/dts/g/epssm assembly) correct so the Thomas solve gives the right implicit w (compare to `module_small_step_em.F:608-649`, `:1477-1502`)? Is the large-step `rw_tend` (pg_buoy_w) being ADDED every substep instead of once (re-injection → constant forcing)? Is there a sign error making the pressure response REINFORCE rather than oppose w? Is `ww` (omega) or the vertical mass flux missing, so there's no vertical advective adjustment?
4. Whatever you conclude, give the SINGLE most likely root cause with file:line evidence, a concrete fix, and a falsifiable check that would confirm it.

## WRF ground truth
`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`: `module_small_step_em.F` (advance_w 1178-1586, advance_mu_t 969-1171, calc_coef_w 608-649, calc_p_rho 492-567), `module_big_step_utilities_em.F` (pg_buoy_w 2539-2572, calc_p_rho_phi 1023-1088).

## Output
Write to EXACTLY: `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7h-geopotential-carry-and-wrf-truth/gpt-bughunt-findings.md`
Structure: (1) what your probe showed (ph frozen? the per-substep trace numbers); (2) THE single most likely root cause with file:line; (3) the concrete fix; (4) a falsifiable check. Be decisive and evidence-first (run probes, don't just reason). End with `F7H_BUGHUNT_COMPLETE`. Do not modify any source file.
