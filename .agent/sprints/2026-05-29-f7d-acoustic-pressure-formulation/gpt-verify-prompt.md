You are GPT-5.5 xhigh acting as the WRF-domain verifier/debugger for the wrf_gpu2 project (a JAX-native GPU port of the WRF v4 dynamical core). Three Opus sprints made the dry dycore acoustic core WRF-cadence-faithful, damped, conservative, with flux-form advection and a correct large-step PGF cadence (the flow now circulates). But the idealized cases (Straka density current, Skamarock warm bubble) still go non-finite (~80–100 s), and the Opus frontrunner has localized this to a **fundamental acoustic-core pressure-formulation mismatch**. Your job: **verify or refute that diagnosis against WRF Fortran source, and if confirmed, specify exactly what the JAX fix must be.** This decides a significant acoustic-core rework, so be rigorous and cite WRF `file:line`.

This is a VERIFY + DESIGN-SPEC task. Do NOT edit any source code. Your ONLY write is the findings file named at the end.

## The diagnosis to verify (from the Opus frontrunner, `proofs/f7c/rk_addtend_dry_proof.md` §3)

Claim: The JAX dycore uses a **delta-from-RK-reference** acoustic formulation — the small-step perturbation-pressure work array `p` produced by `calc_p_rho` is relative to the RK-stage reference state, which for a slowly-evolving parcel is ≈ the current state, so the work-`p` carries **≈ no buoyancy**. To make a statically-imbalanced parcel rise, Sprint B had to feed the vertical buoyancy term `pg_buoy_w` the **absolute** perturbation pressure p′ (from `rk_step_prep`), confirmed required (using the substep work-`p` gives max|w|→0.01, i.e. no rise). BUT that absolute p′ is a **once-per-RK-stage constant** forcing that never receives the acoustic pressure-adjustment feedback during the small steps (the work-`p` used by `advance_uv`/`advance_w` stays ≈ 0), so buoyancy + PGF pump momentum into the circulation with **no acoustic restoring loop** → coherent linear u/w runaway (independent of acoustic substep count, epssm 0.1–0.5, and explicit diffusion up to ν=200) → NaN. Proposed fix: diagnose the small-step pressure from the **absolute small-step total state** (WRF's `mu_save` + `mu_work` totals through `calc_p_rho`) so buoyancy and pressure-restoring are consistent.

## What to determine against WRF source (ground truth)

WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`. Read at minimum:
- `module_small_step_em.F`: `small_step_prep` (125-285), `calc_p_rho` (492-563), `advance_uv` (654-942), `advance_mu_t` (969-1171), `advance_w` (1178-1584), `small_step_finish` (364-430).
- `module_em.F`: `rk_step_prep`/`calc_p_rho` callers, `pg_buoy_w`/buoyancy assembly.
- `solve_em.F`: the calc_p_rho call sites (step=0 at 2628-2670; step=iteration at 4164-4206) and what state arrays they pass.

Answer precisely:
1. **What state does WRF's `calc_p_rho` operate on each substep** — the absolute small-step total (`mut`/`muts` = base + work dry mass, total theta = reference + coupled work), or a perturbation/delta relative to the RK-stage entry? Quote the `al`/`p` update lines.
2. **Does the WRF small-step `p` (perturbation pressure) that `advance_uv`/`advance_w` use for the PGF accumulate the buoyancy/pressure response of a statically-imbalanced column during the substeps** (i.e. is there a genuine acoustic restoring feedback within one RK stage), or is buoyancy carried only by a frozen once-per-stage term? Where exactly does the vertical buoyancy enter `advance_w` in WRF, and is it the absolute p′ or the work-`p`?
3. **Is the frontrunner's "delta-from-reference vs absolute-total" framing correct?** Confirm or refute that WRF's formulation provides the restoring loop the JAX delta formulation lacks. If the real WRF mechanism is different (e.g. the restoring comes from `advance_mu_t` updating `mu`→`calc_p_rho`→p′ each substep, or from `t_2ave`, or from the `c2a`/`cqw` coupling), say so precisely.

## Then read the JAX side and pinpoint the divergence
- `src/gpuwrf/dynamics/core/calc_p_rho.py`, `src/gpuwrf/dynamics/core/acoustic.py` (`acoustic_substep_core`, `calc_p_rho_step`, the work-array `p` lifetime), `src/gpuwrf/dynamics/core/small_step_prep.py`, `src/gpuwrf/dynamics/core/advance_w.py` (`pg_buoy_w`), `src/gpuwrf/dynamics/mu_t_advance.py`.
- State exactly which JAX line(s) make `p` a delta-from-reference instead of the WRF absolute-total path, and whether `calc_p_rho_step` updates `p` from the per-substep advanced `mu`/`theta`/`al` (the restoring feedback) or not.

## Output
Write to EXACTLY this absolute path (so it survives — never inside a /tmp worktree):
`/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7d-acoustic-pressure-formulation/gpt-findings.md`

Structure: (1) WRF ground-truth answers to Q1-Q3 with file:line; (2) verdict CONFIRM / REFUTE / PARTIAL on the frontrunner's diagnosis; (3) the exact JAX fix spec for the next Opus sprint — which functions/arrays change, what the new `calc_p_rho`/work-array formulation must be, what the acoustic substep restoring loop must look like, and a falsifiable check that would prove the fix works (e.g. "warm bubble max|w| saturates rather than growing linearly"). End the file with `F7D_VERIFY_COMPLETE`. Do not modify any other file.
