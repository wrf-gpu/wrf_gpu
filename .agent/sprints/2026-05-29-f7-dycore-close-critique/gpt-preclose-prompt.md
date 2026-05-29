You are GPT-5.5 xhigh performing the PRE-CLOSE critique of the WRF v4 dry dynamical core for the wrf_gpu2 project, before it is declared DONE and Phase B (physics) builds on it. This is the principal's firm rule: an independent WRF-domain review before any major milestone close, specifically to catch HIDDEN bugs, shortcuts, or false-green proofs. Be adversarial and specific.

## What is being closed
After the F7.A-N rewrite, BOTH published-reference idealized cases now PASS (the honest gates):
- Skamarock warm bubble: 6/6 (thermal_rise 1924 m, max|w| 11.68, θ′max 1.92, drift 0, mass drift 0).
- Straka density current: all 6 (finite to 900 s; min θ′ −9.97 K; max|w| 14.57; front 14,150 m vs ~15 km ref; 4 KH rotors; mass drift 2.3e-9).
The final fix (F7N) was a sign correction in `flux_advection._vertical_flux_div_3` (vertical momentum upwind: JAX had +|vel|·corr anti-dissipative; WRF advect_u `module_advect_em.F:202-204,1474-1480` uses −|vel|·corr dissipative).

## Read
- `proofs/f7/DYCORE_STATUS.md` (verified-correct components list).
- The dycore code: `src/gpuwrf/dynamics/core/{acoustic.py,advance_w.py,calc_p_rho.py,small_step_prep.py,small_step_finish.py,rk_addtend_dry.py,rhs_ph.py}`, `src/gpuwrf/dynamics/{acoustic_wrf.py,flux_advection.py,advection.py,mu_t_advance.py,explicit_diffusion.py}`, `src/gpuwrf/runtime/operational_mode.py`.
- Proofs: `proofs/f7n/` (Straka + warm-bubble verdicts + touchdown diff), `proofs/f7m/wrf_vs_jax_straka_front.json`, `proofs/f7a2/`, the WRF ground truth `proofs/m9/` + `/mnt/data/wrf_gpu2/wrf_truth/`.
- WRF source (ground truth): `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/` and the pristine build `/home/enric/src/wrf_pristine/WRF`.

## Questions (the hidden-bug hunt)
1. **Are the idealized PASSES honest?** Any masking clamp/cap/tanh-sanitizer/positive-definite-limiter, tolerance widening, deleted/xfailed test, or tuned-to-pass coefficient (beyond the benchmark-defined ν=75 + WRF-default damping) on the path that produces the warm-bubble/Straka verdicts? Check the idealized harness + the operational path. Flag ANY shortcut.
2. **Are the verdicts real (not self-compares)?** The warm-bubble/Straka PASSES are vs published references; the WRF touchdown diff is vs a pristine-WRF binary. Confirm no JAX-vs-JAX tautology remains in the accepted gates (the original sin).
3. **WRF-faithfulness of the key operators** (spot-check vs WRF source, cite file:line): the F7N vertical-momentum-advection sign fix; rhs_ph/ph_tend (F7J); advance_w implicit solve + calc_coef_w/epssm; calc_p_rho (smdiv/pm1); MUT/MUTS mass semantics; flux-form advection orders; the const-K diffusion (u,v,w,θ). Any operator that passes the idealized gate but is WRF-WRONG in a way the smooth/idealized cases don't exercise (esp. things that will bite in the real Canary regime: moisture coupling, map factors away from the idealized f-plane, real terrain, boundaries)?
4. **What does the idealized suite NOT cover that the operational forecast needs?** (moisture/scalar coupling, real map factors, full terrain, lateral BC, physics tendencies into the RK1 bundle). I.e., what's the gap between "idealized PASS" and "ready for Phase B physics coupling"?
5. **Verdict**: is the dry dycore SOUND to close and build Phase B physics on? Score /10. List any must-fix-before-Phase-B items vs nice-to-haves.

## Output
Write to EXACTLY `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7-dycore-close-critique/gpt-preclose-findings.md`. Read-only on code; only write that file. Be specific (file:line). End with `F7_PRECLOSE_COMPLETE` and an explicit CLOSE-APPROVED or CLOSE-BLOCKED-pending-<items> verdict.
