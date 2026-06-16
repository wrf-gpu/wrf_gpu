# GPT Stage3 Wrapper Verifier

Date: 2026-06-11
Verifier: GPT-5.5 xhigh
Reviewed Fable commit: `a5f282521090c4b1e3d1d4618295db09d49cdc17`
Fable worktree: `/home/user/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

## Verdict

`LOCAL_FIX_PROPOSED`: the Fable patch is substantially useful boundary-faithfulness work and the h36->h37 blocker remains open, but it should not be merged as fully WRF-faithful until one small wrapper defect is fixed: `apply_lateral_boundaries(dry_spec_only=True)` must not overwrite specified-domain `w` from `w_bdy` at end-of-step; WRF uses `zero_grad_bdy` for specified `w` inside the acoustic loop and has no end-of-step dry overwrite.

This defect does not overturn Fable's main residual conclusion. The interior `ph` sink / `p` rise appears by `step1_stage1_vs_21602`, before wrapper end-of-step writes, and is numerically unchanged across `rhsph2`, `speccad`, and `advdeg` variants.

## Fable Claims vs Evidence

| Claim | Evidence checked | GPT judgment |
|---|---|---|
| Per-stage specified LBC cadence is directionally WRF-shaped | WRF `solve_em.F:938-965`, `:1346-1607`; WRF `module_bc.F:1221-1427`; Fable code in `boundary_apply.py`, `acoustic.py`, `operational_mode.py`; tests `5 passed, 1 skipped` | Mostly yes. Relax-zone dry tendencies, in-loop `ph`/mass/theta pins, tangential wind targets, and dry end-of-step relax removal match the WRF cadence shape. |
| Boundary band improves against WRF-native dumps | `switzerland_stage3_wrapper_cadence.json`: step1_stage1 `p` band `72.900 -> 3.936 -> 2.084`; step1 final `mu` ring1 `39.049 -> 37.155 -> 1.877`; replica-vs-jit max diffs <= `5.97e-08` | Yes for the measured `p/mu/ph` bands. Proof is WRF-native stage-dump anchored, not JAX-vs-JAX. |
| Specified advection degradation is WRF-faithful | WRF `module_advect_em.F` order-5 degrade blocks for scalar/u/v/w; helper test mirrors tier map including upstream normal rule | Yes for the h=5 path under this proof. Risk remains: the implementation is hard-coded to order-5 behavior and should stay opt-in unless broader order handling is added. |
| h36->h37 blocker is not closed | Gate table below; advdeg residual `-21.064` vs CPU `+5.178`; advdeg excess outflux `-32.870`, worse than `rhsph` `-27.204` | Correct. Do not run the Switzerland 72h gate from this result. |
| Residual points to interior acoustic `advance_w`/`phi` lane | `phi_p_hydrostatic_pair`: stage1 `ph_err_mean=-0.35948`, `p_err_mean=+0.53331`; final `ph_err_mean=-2.16384`, `p_err_mean=+2.63214`, with `mu_err_mean ~ -0.001` | Survives review. The first direct discriminator should split `advance_w_wrf()` terms, not send another boundary sprint. |

## Source Risk Table

| File/function | Classification | Review notes |
|---|---|---|
| `src/gpuwrf/coupling/boundary_apply.py::_apply_3d_spec_only` and `apply_lateral_boundaries(... dry_spec_only=True)` | WRF-faithful required change with one local defect | Drops relax-zone dry overwrite and `p/pb` forcing as intended. Defect: `w = _spec3(state.w, state.w_bdy)` is not WRF specified cadence; replace with `w = state.w` so in-loop `zero_grad_bdy` remains the owner. Update the unit test to assert `w` is unchanged in `dry_spec_only`. |
| `boundary_apply.py::specified_relax_dry_tendencies` | WRF-faithful required change | Matches WRF relax-zone exclusion of ring 0 and corner trims; independent theta/mu mirror passed. Known approximation: decoupled leaves couple both sides with reference mass. Acceptable while default-off and documented. |
| `boundary_apply.py::tangential_bdy_work_target_u/v` | WRF-faithful required change | Covers tangential u S/N and v W/E work targets that normal-face wind handling did not cover. Unit reconstruction checks passed. |
| `src/gpuwrf/dynamics/core/acoustic.py` specified target fields, pins, and `spec_w_zero_grad` | WRF-faithful required change | Aligns with WRF in-loop `spec_bdyupdate`, `spec_bdyupdate_ph`, and `zero_grad_bdy` ordering. Default-off. |
| `src/gpuwrf/dynamics/flux_advection.py::specified_flux_faces`, `_specified_div`, `couple_uv_specified`, advection branches | WRF-faithful for current h=5 proof; risky for broader default | Stage proof strongly supports the h=5 Switzerland path. Do not generalize to h=2/3/4/6 cases or default-on without order guards/tests. PD limiter edge behavior is documented as not fully line-ported. |
| `src/gpuwrf/runtime/operational_mode.py` flags, `_specified_*_active`, `_specified_bdy_relax`, stage target plumbing | WRF-faithful required change | Active only for `run_boundary`, non-ideal specified domains with `force_geopotential=True` and explicit flags. Idealized and nested `force_geopotential=False` paths stay off. |
| `src/gpuwrf/integration/daily_pipeline.py` env flags | Proof/operational support | Keeps both features default-off via `GPUWRF_SPECIFIED_BDY_CADENCE=1` and `GPUWRF_SPECIFIED_ADV_DEGRADE=1`. Good merge discipline while blocker is open. |
| `proofs/v014/switzerland_acoustic_substep_blocker.py` updates | Proof-only support | Mirrors production relax/boundary behavior in the replica. No runtime effect. |
| `proofs/v014/switzerland_stage3_wrapper_cadence.py/json` and `tests/test_v014_specified_bdy_cadence.py` | Proof-only support | Useful and should be kept with the patch; JSON validates and focused tests pass. |

Recommended local patch before merge:

```diff
diff --git a/src/gpuwrf/coupling/boundary_apply.py b/src/gpuwrf/coupling/boundary_apply.py
@@
-        w = _spec3(state.w, state.w_bdy)
+        # WRF specified domains do not leaf-pin w at end-of-step; zero_grad_bdy
+        # inside the acoustic loop owns the specified w ring.
+        w = state.w
```

## Gate Result Comparison

Depth-8 h36->h37 dry-mass budget vs CPU truth:

| Run | Excess outflux Pa/cell/h | Residual Pa/cell/h |
|---|---:|---:|
| CPU truth | n/a | `+5.178443877551032` |
| old `ec4d6769` | `-28.614795918367335` | `-32.686352040816345` |
| hypso `3d0b439c` | `-28.3281887755102` | `-27.697448979591826` |
| rhs_ph `79b0c22e` | `-27.203954081632645` | `-21.882908163265313` |
| + specified LBC cadence | `-30.68188775510204` | `-20.302933673469383` |
| + cadence + adv degrade | `-32.86951530612245` | `-21.064285714285717` |

h36->h38 remains red: CPU residual `-64.34387755102041`; rhs_ph residual `-149.5767857142857`; specified-cadence residual `-154.42232142857142`, excess `-23.278698979591837`.

## Runtime Risk

Default-off cost: expected zero hot-path cost except extra static namelist fields and code size; the env flags default to false and active checks exclude idealized and nested `force_geopotential=False` paths.

Enabled cost: extra per-step relax-bundle construction, per-substep ring pins, `w` zero-grad, and specified advection zero-fill/tier masks plus full-face `ru/rv`. This adds JIT/code complexity in flux advection and should be profiled only after correctness closure.

Hot-path concerns: no host/device transfers were introduced in the timestep loop in the reviewed code. The advection branch is order-5-specific and should remain opt-in. The `dry_spec_only` `w_bdy` overwrite is the only merge-blocking local correctness defect I found.

## Next Interior Discriminator

Build `proofs/v014/switzerland_advance_w_phi_discriminator.py` or extend `switzerland_acoustic_substep_blocker.py --capture-intra` to split stage-1 substep-1 `advance_w_wrf()` from the bit-identical h36/call-21601 input.

Exact JAX dump points:

1. In `_advance_stage_replica()` after `small_step_prep_wrf()` and `calc_p_rho_wrf()`, keep current `intra["prep"]` plus add `ww_stage`, `pressure.p`, `pressure.al`, and `pressure.alt`.
2. In `src/gpuwrf/dynamics/core/acoustic.py::acoustic_substep_core`, immediately before line calling `advance_w_wrf()`, dump `state_for_w.ph`, `state_for_w.w`, `state_for_w.ph_tend`, staged `rw_tend`, `ww_new`, `muave_new`, `muts_new`, `theta_coupled`, `state_for_w.t_2ave`, `c2a`, `cqw`, and `alt`.
3. In a proof-only copy/wrapper of `src/gpuwrf/dynamics/core/advance_w.py::advance_w_wrf` (`:229-532`), emit term arrays matching WRF `module_small_step_em.F:1314-1463`: `t_2ave_next`, `rhs_seed`, `wdwn`, `rhs_after_phi_adv`, `rhs_predictor`, `w_pre_solve` split into `rw_tend`, `termA`, `termB`, `w_fwd`, `w_solved`, and `ph_next`.
4. Compare those arrays against a WRF-native/line-ported oracle for call 21602. If WRF lacks these term dumps, add a minimal WRF patch/dump run only for calls 21601-21602, not a forecast gate.

Acceptance thresholds for the discriminator:

| Surface | Threshold | Decision |
|---|---:|---|
| JAX pre-`advance_w` inputs vs WRF | `max_abs <= 1e-8` for `ph/w/ph_tend/rw_tend/ww`, relative `<=1e-10` for mass weights where scale is large | If fail, fix staging (`rw_tend`/`ph_tend`/`ww`) before editing solver math. |
| `rhs_seed`, `wdwn`, `rhs_after_phi_adv` | `max_abs <= 1e-8` or explainable fp64 roundoff | If fail, root is phi RHS/advection branch. |
| `w_pre_solve`, `w_fwd`, `w_solved` | `max_abs <= 1e-8` | If fail, root is implicit coefficients or Thomas solve consumption. |
| `ph_next` with matching `w_solved` | `max_abs <= 1e-8` | If fail, root is final `ph` update/mass denominator/off-centering. |
| If all `advance_w` terms pass but stage `p/ph` still fails | n/a | Move to post-stage `small_step_finish_wrf`, `calc_p_rho_step`, `_refresh_grid_p_from_finished`. |

Run command target after implementing the proof script:

```bash
python proofs/v014/switzerland_advance_w_phi_discriminator.py --stage 1 --substep 1 --dt 18 --substeps 4 --tag advw_phi_h36_call21602
```

Do not run a long GPU gate. This is CPU/static/oracle work until a source fix exists.

## Commands Run

```bash
sed -n '1,240p' PROJECT_CONSTITUTION.md
sed -n '1,260p' AGENTS.md
sed -n '1,260p' .agent/sprints/2026-06-11-v014-gpt-stage3-wrapper-verifier/sprint-contract.md
sed -n '1,260p' .agent/skills/validating-physics/SKILL.md
sed -n '1,260p' .agent/skills/managing-sprints/SKILL.md
sed -n '1,260p' .agent/decisions/V0140-RELEASE-CHECKLIST.md
sed -n '1,320p' .agent/sprints/2026-06-11-v014-fable-stage3-wrapper-cadence/sprint-contract.md
sed -n '1,260p' .claude/worktrees/v014-hpg-native-face-fix/.agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md
python -m json.tool .claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_stage3_wrapper_cadence.json
python -m json.tool proofs/v014/switzerland_acoustic_continuation.json
python -m json.tool .claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_acoustic_substep_blocker.json
git -C .claude/worktrees/v014-hpg-native-face-fix status --short
git -C .claude/worktrees/v014-hpg-native-face-fix log --oneline -5
git -C .claude/worktrees/v014-hpg-native-face-fix show --stat --oneline --decorate --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17
git -C .claude/worktrees/v014-hpg-native-face-fix diff --name-only a5f282521090c4b1e3d1d4618295db09d49cdc17^ a5f282521090c4b1e3d1d4618295db09d49cdc17
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- src/gpuwrf/coupling/boundary_apply.py
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- src/gpuwrf/dynamics/core/acoustic.py
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- src/gpuwrf/dynamics/flux_advection.py
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- src/gpuwrf/runtime/operational_mode.py
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- src/gpuwrf/integration/daily_pipeline.py
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- tests/test_v014_specified_bdy_cadence.py
git -C .claude/worktrees/v014-hpg-native-face-fix show --format= --no-renames a5f282521090c4b1e3d1d4618295db09d49cdc17 -- proofs/v014/switzerland_stage3_wrapper_cadence.py
nl -ba /home/user/src/wrf_pristine/WRF/dyn_em/solve_em.F
nl -ba /home/user/src/wrf_pristine/WRF/share/module_bc.F
nl -ba /home/user/src/wrf_pristine/WRF/dyn_em/module_bc_em.F
nl -ba /home/user/src/wrf_pristine/WRF/dyn_em/module_advect_em.F
nl -ba /home/user/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F
pytest -q tests/test_v014_specified_bdy_cadence.py
git diff --check a5f282521090c4b1e3d1d4618295db09d49cdc17^ a5f282521090c4b1e3d1d4618295db09d49cdc17
python -m json.tool /home/user/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_stage3_wrapper_cadence.json >/tmp/gpt_stage3_wrapper_verifier_stage3_json.validated
python -m json.tool /home/user/src/wrf_gpu2/proofs/v014/switzerland_acoustic_continuation.json >/tmp/gpt_stage3_wrapper_verifier_continuation_json.validated
python -m json.tool /home/user/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_acoustic_substep_blocker.json >/tmp/gpt_stage3_wrapper_verifier_blocker_json.validated
```

Additional `rg`, `nl`, and Python read-only summary commands were used to inspect flag reachability, source line references, gate values, and `phi_p_hydrostatic_pair` values. No GPU command was run.

## Artifacts Checked

- `.claude/worktrees/v014-hpg-native-face-fix/.agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md`
- `.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_stage3_wrapper_cadence.json`
- `.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_stage3_wrapper_cadence.py`
- `proofs/v014/switzerland_acoustic_continuation.json`
- `.claude/worktrees/v014-hpg-native-face-fix/proofs/v014/switzerland_acoustic_substep_blocker.json`
- `.claude/worktrees/v014-hpg-native-face-fix/tests/test_v014_specified_bdy_cadence.py`
- WRF pristine source under `/home/user/src/wrf_pristine/WRF/dyn_em` and `/home/user/src/wrf_pristine/WRF/share`

## Manager Next Actions

- Apply the one-line `w` wrapper fix or ask the Fable branch owner to apply it; rerun `tests/test_v014_specified_bdy_cadence.py` with the updated expectation.
- Keep `specified_bdy_cadence` and `specified_adv_degrade` default-off.
- Accept the boundary/advection proof as useful after the `w` fix, but do not claim the h36->h37 blocker closed.
- Do not start Switzerland 72h.
- Start the `advance_w`/`phi` discriminator proof loop above.
- If `advance_w` term parity fails, fix the first failing term and rerun the h36->h37 short gate only.
- If `advance_w` term parity passes, move to `small_step_finish_wrf`/`calc_p_rho_step`/grid-p refresh before another model sprint.

## Handoff

- objective: independently verify Fable stage-3/wrapper-cadence result and define the next interior discriminator.
- files changed: `proofs/v014/gpt_stage3_wrapper_verifier.md`.
- proof objects produced: this verifier report.
- unresolved risks: `dry_spec_only` specified `w` overwrite; order-5-only advection degradation; no new WRF term dumps for `advance_w` yet.
- next decision needed: whether manager applies the one-line `w` fix before accepting the Fable boundary patch.
