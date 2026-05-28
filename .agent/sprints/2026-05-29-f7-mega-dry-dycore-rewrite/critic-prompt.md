You are GPT-5.5 xhigh acting as an independent WRF-domain critic for the wrf_gpu2 project. You are reviewing a sprint CONTRACT before a large (~1000-1500 LOC) JAX rewrite of the WRF dry dynamical core executes. Your job is to catch WRF-factual errors, missing cadence, and weak/misleading acceptance gates BEFORE the rewrite runs — a prior contract had a real WRF error (it claimed `calc_p_rho(step=iteration)` recomputes `c2a`, but `c2a` is INTENT(IN), set once in `small_step_prep`) that wasted a sprint. Be specific and adversarial.

This is a REVIEW-ONLY task. Do NOT edit any source code. Your ONLY write is the critique file named below.

## Read these (WRF Fortran source is ground truth)

1. The contract under review: `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7-mega-dry-dycore-rewrite/sprint-contract.md`
2. The binding cadence spec: `/home/enric/src/wrf_gpu2/proofs/f5/wrf_cadence_spec.md`
3. Current JAX dycore core: `/home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py`, `.../calc_p_rho.py`, `.../small_step_prep.py`, `.../small_step_finish.py`, `/home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/acoustic_wrf.py`, `.../mu_t_advance.py`, `.../advection.py`, `.../tridiag_solve.py`, and `/home/enric/src/wrf_gpu2/src/gpuwrf/runtime/operational_mode.py`.
4. WRF Fortran at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`: `module_small_step_em.F`, `module_em.F`, `module_big_step_utilities_em.F`, `solve_em.F`. VERIFY the contract's WRF claims against these line ranges yourself.

## Questions to answer

1. **WRF-factual correctness.** Are there any WRF-factual errors in the contract like the prior `c2a` mistake? Check specifically: the `advance_w` RHS term list and sign conventions; the `calc_p_rho(step=0)` vs `step=iteration` responsibilities; the `_1` vs `*_save` lifetimes in `small_step_prep`/`small_step_finish`; `calc_mu_uv_1` from `muts`; the `rk_addtend_dry` map/mass-coupling per field; flux-form mass-coupled advection coupling. Cite WRF `file:line` for each correction.
2. **Cadence/order.** Is the implementation order (calc_coef_w c2a → calc_p_rho cadence → advance_w → prep/finish/calc_mu_uv_1 → flux-form advection → rk_addtend_dry → sumflux) sound, or will an earlier block be numerically meaningless without a later one (e.g. advance_w meaningless without real c2a)?
3. **Missing operators.** Does anything required for a *physics-off, periodic-BC* dry core to be correct and stable get omitted? (e.g. divergence damping, off-centering epssm, vertical/horizontal mom. coupling, geopotential lower BC.) Is anything in scope that is actually unnecessary for the physics-off gate and should be deferred?
4. **Gate strength.** Are AC1–AC6 falsifiable enough to prevent a green self-compare from masking a wrong dycore? Is the Straka/Skamarock tolerance framing right (front position, min θ′, max w, mass drift)? Is the flat-rest oracle (AC4) constructed correctly to catch sign/coupling errors? What gate is missing?
5. **Scope realism.** Is this honestly one coherent sprint for a strong implementer, or should one block be split out? If split, name the cut line.
6. **Honest verdict.** Score /10 and the 2-3 changes you'd make to the contract before it executes.

## Output

Write your critique to EXACTLY this absolute path (so it survives — do not write it inside any /tmp worktree):
`/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7-mega-dry-dycore-rewrite/critique.md`

End the file with a line `F7_MEGA_CRITIQUE_COMPLETE`. Do not modify any other file.
