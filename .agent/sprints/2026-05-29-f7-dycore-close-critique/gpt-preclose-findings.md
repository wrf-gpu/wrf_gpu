# GPT Pre-Close Findings - F7 Dry Dycore

Date: 2026-05-29
Reviewer: GPT-5.5 xhigh / Codex
Scope: read-only pre-close critique of the F7 dry dynamical core, with one write to this report file.

Verdict: CLOSE-BLOCKED-pending-operational-path-unification+canonical-WRF-Straka-rerun+WRF-momentum-diffusion+CI-pass-gates

Score: 6/10.

The F7N work is a real improvement. The two idealized verdicts pass, the vertical momentum advection sign fix matches WRF's flux3 sign convention for the checked path, and I did not find evidence that the accepted F7N idealized verdicts are JAX-vs-JAX tautologies or are passing because of the production theta limiter. I would not close this as "WRF dry dycore DONE" yet. The proof stack still has several places where the published green path is narrower than the operational path Phase B is likely to build on, and at least one WRF operator is knowingly approximated while the status text reads as closure.

## Blocking Findings

### P0 - The passing dycore path is not the operational/default path

The F7N passes are produced through the idealized harness with special controls:

- `src/gpuwrf/ic_generators/idealized.py` builds `OperationalNamelist(... disable_guards=True, force_fp64=True, use_flux_advection=True, const_nu_m2_s=...)`.
- The same setup forces `run_physics=False`, `run_boundary=False`, `top_lid=True`, `w_damping=1`, `damp_opt=3`, `dampcoef=0.2`, `zdamp=3000`.

The default operational configuration is materially different:

- `src/gpuwrf/runtime/operational_mode.py` defaults to `disable_guards=False`, `w_damping=0`, `damp_opt=0`, `const_nu_m2_s=0`, `use_flux_advection=False`, and `force_fp64=False`.
- `src/gpuwrf/integration/daily_pipeline.py::_build_real_case` constructs an `OperationalNamelist.from_grid(...)` for real cases without setting `use_flux_advection=True`, `disable_guards=True`, `force_fp64=True`, or the idealized diffusion/damping controls.
- `operational_mode._augment_large_step_tendencies` only uses the F7 flux-form advection path when `namelist.use_flux_advection` is true; otherwise it uses the older primitive `compute_advection_tendencies` path.

Impact: Phase B can build on a real-case/physics path that bypasses the just-closed advection fixes and re-enables guards. That makes "F7 dry dycore closed" ambiguous: the proven path is not the path the operational pipeline will run by default.

Required close action: unify the operational Phase-B path with the closed F7 dycore path, or explicitly version the namelist/mode that is declared closed and add a real-case smoke/proof showing that Phase B uses that mode.

### P0 - WRF momentum diffusion is not wired; scalar-like constant-K diffusion is used for momentum

WRF `diff_opt=2/km_opt=1` does not diffuse momentum by applying the scalar Laplacian independently to `u`, `v`, and `w`. In pristine WRF:

- `dyn_em/module_diffusion_em.F` calls `horizontal_diffusion_u_2`, `horizontal_diffusion_v_2`, and `horizontal_diffusion_w_2` for momentum, and scalar routines separately.
- The momentum routines use deformation/stress terms such as `defor13`, `defor23`, and `defor33`; the vertical momentum diffusion path is not the same scalar flux divergence used for theta.

In the JAX code:

- `src/gpuwrf/dynamics/explicit_diffusion.py::conservative_constant_k_diffusion_tendency` implements a conservative scalar-style flux divergence.
- `src/gpuwrf/runtime/operational_mode.py` adds that same scalar-style tendency to `u`, `v`, `w`, and `theta` when `const_nu_m2_s > 0`.
- `src/gpuwrf/dynamics/explicit_diffusion.py::constant_k_deformation_momentum_tendency` exists, but `rg constant_k_deformation` shows it is not wired into the runtime path.

Impact: the accepted Straka proof includes constant-K diffusion, but the wired operator is not WRF's momentum diffusion operator. This is exactly the kind of "close but not WRF" shortcut that can pass broad front-location metrics while diverging in shear, vertical velocity, terrain, or real-case coupling.

Required close action: wire the deformation-tensor momentum diffusion path, prove it against WRF/source-derived or analytic operator checks, and rerun the F7 idealized gates; or explicitly demote the claim from WRF-faithful dry dycore closure to "published-metric idealized prototype with scalar momentum diffusion approximation."

### P0 - The WRF-vs-JAX Straka evidence is not a canonical parity artifact

The F7M/F7N touchdown proof usefully localized the vertical momentum advection sign, but it is not yet a strict WRF-vs-JAX equivalence proof:

- WRF canonical `test/em_grav2d_x/namelist.input.100m` uses `time_step=1`, `time_step_sound=6`, `diff_opt=2`, `km_opt=1`, `khdif=kvdif=75`, `damp_opt=0`, horizontal/vertical advection order 5/3, and the WRF grid/top configuration.
- The JAX Straka setup in `proofs/f7m/wrf_vs_jax_straka_front.json` records `dt=0.1`, 10 acoustic substeps, `nu=75`, `damp_opt=3`, `zdamp=3000`, `dampcoef=0.2`, `nx=500`, `nz=60`, and a theta perturbation minimum of about `-15`.
- The F7N scripts themselves acknowledge mismatches: WRF `time_step_sound=6` while JAX uses 10 acoustic substeps; WRF has 65 vertical levels while the JAX snapshot path uses 60.
- The accepted F7N verdicts compare to broad published-reference ranges, not to near-identical WRF state arrays.

Impact: this is good diagnostic evidence, but it is not enough to declare the WRF v4 dry dycore closed before Phase B. The proof can pass while damping/substep/grid/IC differences hide operator errors.

Required close action: rerun the Straka close gate under canonical WRF controls or a documented exact transform of them: `damp_opt=0`, WRF-compatible `time_step_sound=6` semantics, matching grid/vertical coordinates/IC, and array-level WRF-vs-JAX diagnostics through at least the previously failing touchdown window.

### P0 - Idealized close tests can be false-green in pytest

The committed idealized test files do not enforce the verdict:

- `tests/idealized/test_warm_bubble.py` and `tests/idealized/test_density_current.py` assert only that `result.verdict in {"PASS", "FAIL"}`.
- A regression to `FAIL` would still pass pytest.
- The tests also skip when no GPU is available, which is reasonable for local CPU-only runs but not sufficient as a close gate.

Impact: the manually committed proof objects are green, but the repository test suite does not protect this milestone from regressing. A future "all tests passed" can be false-green for F7.

Required close action: add a close-gate test or script that asserts `PASS` for the idealized cases and archives the JSON/markdown proof object, or make the existing tests assert `PASS` under the close-gate marker/environment.

## Major Findings

### P1 - WRF `advect_w` top-face contribution appears missing for non-lid/open-top cases

For `vert_order == 3`, pristine WRF `dyn_em/module_advect_em.F::advect_w` computes the top vertical flux and adds a top-face tendency contribution:

- WRF computes `vflux(i,kde)` from the top velocity/stencil.
- WRF then applies `tend(i,k) = tend(i,k) + 2*rdzu(k-1)*vflux(i,k)` for the top face.

The JAX `src/gpuwrf/dynamics/flux_advection.py::_vertical_flux_div_w` fills interior `tend[1:nz]` from face differences and leaves the top-face tendency zero. The idealized cases use `top_lid=True`, which can mask this. The real operational path is not proven to be lid-only.

Impact: the F7 vertical momentum fix may be correct for the closed idealized top-lid cases while still being WRF-wrong for open/top-damped configurations needed by real weather runs.

Required action: either implement and test the WRF top-face branch for `w`, or state that the closed F7 operator is top-lid-only and block Phase B real-path use until the open/top branch is proven.

### P1 - Runtime guards and theta limiter can mask operational failures

The idealized harness disables guards, so the accepted F7N idealized verdicts are not passing because of the production theta limiter. However, the operational mode defaults keep guards enabled:

- `_positive_definite_theta_increment_limiter` can scale theta increments.
- `_limit_guarded_dynamics_state_with_diagnostics` applies finite bounds.
- `_physics_boundary_step_with_limiter_diagnostics` applies limiter/guard logic when `disable_guards` is false and uses fallback states on non-finite failures.

Impact: Phase B/real-case runs may appear stable while relying on guard intervention. The proof objects do not yet record "zero limiter/guard engagement" for the operational path.

Required action: add validation-mode proof objects that either run with guards disabled or record diagnostics proving zero limiter/guard/fallback engagement over the accepted gate.

### P1 - The F7 scope is narrower than the status language

Several core files explicitly narrow the implementation:

- `src/gpuwrf/dynamics/flux_advection.py` says the flux-form operators are currently periodic/unit-map-factor and defer map factors and specified/nested boundaries.
- `src/gpuwrf/dynamics/core/rhs_ph.py` documents an idealized/periodic scope and defers map factors and higher-order horizontal branches.
- The accepted F7 gates are dry, flat/idealized cases. They do not prove terrain, map factors, lateral boundary handling, moisture/scalar coupling, or physics tendencies through the RK bundle.

Impact: this is acceptable only if the milestone is explicitly closed as a flat/periodic dry-dycore subset. It is not sufficient evidence for a general WRF v4 dycore foundation for real Canary Islands cases without follow-on gates.

Required action: update the close statement to include the precise closed subset and create Phase B gates for terrain/map factors/boundaries/moist coupling before using the status as a general dycore foundation.

## Positive Checks

- The F7N idealized verdicts are real proof objects, not synthetic happy-path-only text. `proofs/f7n/skamarock_bubble_verdict.md` passes 6/6 with thermal rise, finite velocity, symmetry, mass drift, and theta anomaly checks. `proofs/f7n/straka_density_current_verdict.md` passes 6/6 with front location, rotor proxy, mass drift, and finite-state checks.
- I found no evidence that the accepted F7N idealized verdicts are JAX-vs-JAX self-compares. They are metric comparisons to published/reference ranges. The WRF touchdown diagnostic uses pristine WRF output, although with the configuration caveats above.
- The vertical momentum advection sign fix itself matches WRF for the checked third-order path. WRF's `flux3(..., -vel)` convention is mirrored by the JAX `sign(-rom_k)` correction in `_vertical_flux_div_3`.
- `calc_p_rho`, `advance_w`/implicit vertical solve, `rhs_ph`, and signed `mu_t` update look broadly WRF-shaped for the flat/idealized subset inspected. I did not find an obvious sign inversion in the `smdiv/pm1` pressure-density memory update.
- The idealized harness sets `disable_guards=True`, so the accepted F7N idealized passes are not obviously due to the production theta limiter or finite-state fallback.

## Answers To The Pre-Close Questions

1. Is the dycore truly honest enough to build Phase B on?

Not yet as a general WRF dry dycore. It is honest as a much-improved flat/periodic idealized branch, but Phase B must not assume the operational real-case path is covered until `use_flux_advection`, guard mode, precision, damping, and diffusion semantics are unified and proven.

2. Are the Straka / bubble tests real, or synthetic green tests?

They are real idealized metrics, but not sufficient WRF parity tests. The pytest wrappers are currently false-green because they accept both `PASS` and `FAIL`. The proof markdown files are meaningful, but the test gate needs to enforce them.

3. Is vertical momentum advection actually fixed?

For the inspected third-order vertical momentum flux path, yes: the sign convention now matches WRF's `flux3(..., -vel)` pattern and the F7N Straka result no longer blows up. The open/top `advect_w` branch is still suspect because the WRF top-face contribution is not represented in the JAX `_vertical_flux_div_w` path.

4. Are pressure, density, phi, MUT, RK, and W coupling WRF-faithful enough?

For the flat/idealized subset, I did not find a decisive blocker in `calc_p_rho`, `advance_w`, `rhs_ph`, or `mu_t_advance`. I did not prove full WRF fidelity for terrain, map factors, boundaries, moisture, or all RK/physics bundle interactions. Those remain out of scope of the accepted evidence.

5. Are there clamps, masks, self-comparisons, disabled terms, or tolerances hiding issues?

No production theta clamp appears to be used in the accepted idealized proofs because guards are disabled. However, the idealized setup uses noncanonical damping, a top lid, special precision, special flux-form opt-in, and broad pass thresholds. Those are not "clamps", but they are evidence-shaping controls that must be recorded and either matched to WRF or declared out of scope.

6. Would I trust this as the foundation for real Canary Islands surface forecasting?

Not yet. I would trust it as a candidate flat/idealized dycore branch after fixing the close gates. I would not let Phase B build real terrain/physics workflows on the default operational path until the F7-proven operators are the same operators used there and the WRF diffusion/top/boundary gaps are either closed or explicitly gated.

## Commands Run

Representative commands used during the read-only critique:

- `pwd`, `git status --short --branch`, `git rev-parse HEAD`, `git log -1 --oneline`
- `sed -n` / `nl -ba` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, sprint prompt, local skill files, F7/F7N/F7M proof files, and relevant source files
- `rg` searches for `use_flux_advection`, `constant_k_deformation`, `disable_guards`, `positive_definite`, `damp_opt`, `time_step_sound`, `advect_w`, `flux3`, `calc_p_rho`, `smdiv`, `horizontal_diffusion`, `vertical_diffusion`, and related symbols
- `python -m json.tool` / small read-only Python snippets to inspect JSON proof metadata and verdict fields
- `find` and `ls` reads under `proofs/f7*`, `proofs/m9`, and `/mnt/data/wrf_gpu2/wrf_truth`

## Proof Objects Reviewed

- `proofs/f7/DYCORE_STATUS.md`
- `proofs/f7n/skamarock_bubble_verdict.md`
- `proofs/f7n/straka_density_current_verdict.md`
- `proofs/f7n/regression_recheck.json`
- `proofs/f7n/touchdown_fix.md`
- `.agent/sprints/2026-05-29-f7n-touchdown-substep-diff/worker-report.md`
- `proofs/f7m/wrf_vs_jax_straka_front.json`
- `proofs/f7a2/*`
- `proofs/m9/wrf_em_grav2d_x_front_savepoints.json`
- `/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_front_savepoints.json`
- WRF source under `/home/enric/src/wrf_pristine/WRF/dyn_em/` and `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`

## Handoff

objective: pre-close adversarial review of F7 dry dycore before declaring it DONE.

files changed: `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-f7-dycore-close-critique/gpt-preclose-findings.md` only.

commands run: read-only source/proof inspection commands listed above; no code modification commands.

proof objects produced: this `gpt-preclose-findings.md` report.

unresolved risks: no fresh executable rerun was performed during this critique; findings are based on committed code/proof/source inspection. The highest risks are operational-path mismatch, non-WRF momentum diffusion, canonical WRF Straka mismatch, and false-green pytest gates.

next decision needed: either block F7 close and assign fixes for the four P0 items, or explicitly narrow the milestone claim to a flat/periodic idealized prototype and prevent Phase B real-case work from treating it as a closed WRF dry dycore.

F7_PRECLOSE_COMPLETE

CLOSE-BLOCKED-pending-operational-path-unification+canonical-WRF-Straka-rerun+WRF-momentum-diffusion+CI-pass-gates
