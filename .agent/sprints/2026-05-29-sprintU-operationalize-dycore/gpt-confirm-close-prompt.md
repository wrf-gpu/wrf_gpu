You are GPT-5.5 xhigh performing the FINAL CONFIRM-CLOSE of the WRF v4 dry dynamical core for wrf_gpu2, before it is declared operational-ready and Phase B physics is built on it. This is the principal's firm rule: an independent WRF-domain review before any major milestone close, to catch HIDDEN bugs, shortcuts, or false-green proofs. Be adversarial and specific (cite file:line).

## Context
You earlier ran the PRE-CLOSE critique (`.agent/sprints/2026-05-29-f7-dycore-close-critique/gpt-preclose-findings.md`) and CLOSE-BLOCKED with 4 P0 + 3 P1 findings. Sprint U claims to have closed ALL of them on branch `worker/opus/f7d-pressure-mass-fix` (8 commits, tip `1b7836c`). Your job NOW is narrow: verify the remediation is GENUINE, not false-green, and that no NEW shortcut was introduced. Do NOT re-litigate already-verified F7 operators (the sign fix, implicit w/ph solve, calc_p_rho, MUT/MUTS, flux orders) — they were confirmed real in your pre-close pass; focus on the REMEDIATION.

## What to verify (per item: GENUINELY CLOSED or NOT — with evidence)

**P0-1 (the critical one) — operational/real-case path unified with the F7 dycore.**
The claim: `daily_pipeline._build_real_case` (`src/gpuwrf/integration/daily_pipeline.py:156`) now builds the namelist with `use_flux_advection=True, force_fp64=True, diff_6th_opt=2, w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2` (was pre-F7 primitive/fp32/no-damping), and `run_forecast_operational` matches the idealized harness BITWISE over 50 warm-bubble steps (theta/w linf = 0.0). Proof: `proofs/sprintU/operational_path_unification.md` + `real_case_smoke.json`.
- VERIFY: is the bitwise claim real (read the proof + the code path `operational_mode.py` / `daily_pipeline.py`)? Does the operational scan actually call the SAME `_physics_boundary_step`/dycore the idealized harness calls, or is there a divergent branch? Is the flux-form branch provably TAKEN on the real Canary case (not silently falling back to primitive)? Any config that LOOKS unified but isn't (e.g. a guard/precision/advection toggle that flips back inside the operational entry)?

**P0-2 — WRF deformation-tensor momentum diffusion wired.** New `wrf_deformation_momentum_tendency` for u/v/w; theta keeps conservative scalar flux-divergence. Claim: analytic FD oracle matches to round-off (du) / ~1% 2nd-order (dw); Straka PASSES 6/6 WITH it (mass drift 1.4e-16). Proofs: `momentum_diffusion_deformation.md`, `straka_deformation_gate.{md,json}`. VERIFY vs WRF `module_diffusion_em.F` (defor11/22/33/12/13/23, factor-2 diagonal, horizontal_diffusion_u/v/w_2). Is the operator WRF-faithful or just FD-self-consistent?

**P0-3 — canonical-WRF Straka array-level parity.** Claim: reran under canonical em_grav2d_x controls (dt=1, 6 acoustic substeps, damp_opt=0, nz=64, nx=512) and compared ARRAY-LEVEL to pristine WRF v4.7.1 through touchdown: worst max|w| rel diff 0.119 (5% at 240s peak, 0% at 300s), front within 400 m, finite to 360s. Proof: `straka_canonical_parity.{json,md}`. VERIFY: is this a real WRF-binary comparison (not JAX-vs-JAX)? Is 0.119 at the touchdown peak acceptable or does it hide a real operator error? Is the WRF reference run the canonical em_grav2d_x?

**P0-4/P0-5 — CI close-gate.** `tests/idealized/test_dycore_close_gate.py` + the two idealized tests now assert `verdict == "PASS"` (not `in {PASS,FAIL}`). VERIFY they actually assert PASS and archive the proof; confirm not skipped/xfailed.

**P1-5 — advect_w open-top face.** WRF top-face flux + lid pickup behind `top_lid`; rigid-lid path byte-unchanged. VERIFY vs `module_advect_em.F:6014-6028`.

**P1-6 — guards-off operational proof.** Claim: warm bubble PASSES 6/6 fully guards-off; real Canary dycore finite guards-off; theta limiter engages under guards-ON but is NOT load-bearing. Proof: `guards_off_operational_proof.json`. VERIFY: is the guards-off PASS real (no hidden clamp/cap/sanitizer still active when disable_guards=True)? Grep the dycore for any tanh/clip/clamp/maximum/where-sanitize/nan_to_num on the state path that survives disable_guards.

## Also
- Any NEW shortcut, masking, tolerance-widening, or tuned-to-pass coefficient introduced by Sprint U's 8 commits (diff `7a17f3e..1b7836c`)? 
- The report admits these remain for Phase B (NOT closed, documented): 3D terrain slope (zx/zy) diffusion cross terms, map factors, lateral/nested boundaries, moist/scalar RK-bundle coupling, per-cell (vs time-series) WRF parity. Is that scope-boundary HONEST, or is something that SHOULD be a dycore-close blocker hiding in there?

## Output
Write to EXACTLY `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-confirm-close-findings.md`. Read-only on code; only write that file. Per-item verdict (CLOSED / NOT-CLOSED + evidence). End with `SPRINTU_CONFIRM_COMPLETE` and an explicit overall **CLOSE-CONFIRMED** (operational-ready, build Phase B) or **CLOSE-REJECTED-pending-<items>** verdict with a /10 confidence.
