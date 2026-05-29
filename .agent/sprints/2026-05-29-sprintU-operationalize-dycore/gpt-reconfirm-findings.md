# Sprint U Reconfirm Findings

Reviewer: GPT-5.5 xhigh (Codex)
Date: 2026-05-29
Branch reviewed: `worker/opus/f7d-pressure-mass-fix` at `bc5660b`
Scope: final focused reconfirm of the two prior rejected findings only, per `gpt-reconfirm-prompt.md`.

## Verdict

**CLOSE-REJECTED-pending-P0-1-operational-entry-force-fp64** confidence 9/10.

The flux-advection dtype remediation itself is real, and the P0-2 deferral is now acceptable as an explicitly documented Phase-B item. But P0-1 is still not completely closed for the actual operational API: `daily_pipeline` calls `run_forecast_operational`, and `run_forecast_operational` still initializes its scan carry with `_enforce_operational_precision(state)` instead of `_enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))`.

## Findings

### P0-1 STILL NOT CLOSED: public operational entry ignores `force_fp64` at scan initialization

`_build_real_case` sets `force_fp64=True` in the real Canary namelist at `src/gpuwrf/integration/daily_pipeline.py:179` through `src/gpuwrf/integration/daily_pipeline.py:195`, and the daily path calls `run_forecast_operational` via `_default_forecast_fn` at `src/gpuwrf/integration/daily_pipeline.py:268` through `src/gpuwrf/integration/daily_pipeline.py:270`.

But the operational entry points still create their initial carry without passing the namelist flag:

- `run_forecast_operational`: `src/gpuwrf/runtime/operational_mode.py:1553`
- `run_forecast_operational_with_limiter_diagnostics`: `src/gpuwrf/runtime/operational_mode.py:1606`
- `run_forecast_operational_debug`: `src/gpuwrf/runtime/operational_mode.py:1659`

That is not a harmless omission after commit `3ee8d94`. `_physics_boundary_step` now enforces `force_fp64` at the end of each step at `src/gpuwrf/runtime/operational_mode.py:1471`, while the initial scan carry is still built from default storage dtypes. JAX scan carry input/output dtypes therefore disagree when `namelist.force_fp64=True`.

Direct verification, pinned/capped as requested:

```text
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY'
...
case, _ = _build_real_case(DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10))
nl = dataclasses.replace(case.namelist, run_physics=False, run_boundary=False, radiation_cadence_steps=999999)
out = run_forecast_operational(case.state, nl, float(nl.dt_s) / 3600.0)
...
PY
```

Observed:

```text
RUN_ERROR TypeError
scan body function carry input and carry output must have equal types, but they differ:
```

The dtype audit shows the exact root cause:

```text
jax_force_fp64_namelist True
u raw float32 force_true float64 force_default float32 carry_default float32
v raw float32 force_true float64 force_default float32 carry_default float32
w raw float64 force_true float64 force_default float64 carry_default float64
theta raw float32 force_true float64 force_default float32 carry_default float32
p raw float64 force_true float64 force_default float64 carry_default float64
ph raw float64 force_true float64 force_default float64 carry_default float64
mu raw float64 force_true float64 force_default float64 carry_default float64
```

So `_enforce_operational_precision(..., force_fp64=True)` itself is fixed, but the actual operational entry does not call it during initialization.

### The new proof artifacts are real, but they bypass the operational entry point

`proofs/sprintU/real_case_smoke.json` records all seven checked prognostics as float64 and finite at `proofs/sprintU/real_case_smoke.json:33` through `proofs/sprintU/real_case_smoke.json:111`. `proofs/sprintU/guards_off_operational_proof.json` records the 50-step real Canary guards-off run as all float64 and finite at `proofs/sprintU/guards_off_operational_proof.json:11` through `proofs/sprintU/guards_off_operational_proof.json:56`.

Those are not fake artifacts, but they are not dispositive for the real operational API. The scripts manually pre-upcast and call `_physics_boundary_step` directly:

- `scripts/sprintU_real_case_smoke.py:94` builds `fp64_state` manually.
- `scripts/sprintU_real_case_smoke.py:124` through `scripts/sprintU_real_case_smoke.py:133` builds `initial_operational_carry(_enforce_operational_precision(..., force_fp64=True))` and loops `_physics_boundary_step` directly.
- `scripts/sprintU_guards_off_proof.py:65` through `scripts/sprintU_guards_off_proof.py:72` does the same for the 50-step guards-off real-case proof.

This contradicts the proof note that "the operational run forces fp64 via `_enforce_operational_precision(..., force_fp64=True)`" at `proofs/sprintU/operational_path_unification.md:89` through `proofs/sprintU/operational_path_unification.md:92`. The proof script forces fp64; the public operational entry still does not.

The "0 warnings + final prognostics are fp64" evidence is therefore not dispositive. A proof can show final float64 state after manual pre-upcast, while the production API either starts from fp32/default carry or, currently, fails the scan type check outright.

### Flux-advection dtype remediation is genuine

The package-level x64 bootstrap is present at `src/gpuwrf/__init__.py:14` through `src/gpuwrf/__init__.py:16`. `State.replace(_cast=False)` is present at `src/gpuwrf/contracts/state.py:566` through `src/gpuwrf/contracts/state.py:588`, and `_enforce_operational_precision(force_fp64=True)` uses it at `src/gpuwrf/runtime/operational_mode.py:299` through `src/gpuwrf/runtime/operational_mode.py:316`.

The flux-advection scatter buffers no longer allocate at a stale fp32 field dtype:

- `rom` uses promoted cumulative dtype at `src/gpuwrf/dynamics/flux_advection.py:152` through `src/gpuwrf/dynamics/flux_advection.py:160`.
- scalar vertical flux uses `jnp.result_type(field.dtype, rom.dtype)` at `src/gpuwrf/dynamics/flux_advection.py:216` through `src/gpuwrf/dynamics/flux_advection.py:220`.
- `_mass_to_full_levels` uses `jnp.result_type(field_mass.dtype, fzm.dtype, fzp.dtype)` at `src/gpuwrf/dynamics/flux_advection.py:399` through `src/gpuwrf/dynamics/flux_advection.py:402`.
- `_vertical_flux_div_3` uses `jnp.result_type(field_mass.dtype, romq.dtype)` at `src/gpuwrf/dynamics/flux_advection.py:430` through `src/gpuwrf/dynamics/flux_advection.py:436`.
- `_vertical_flux_div_w` uses `jnp.result_type(w.dtype, rom.dtype)` for both `vflux` and `tend` at `src/gpuwrf/dynamics/flux_advection.py:495` through `src/gpuwrf/dynamics/flux_advection.py:532`.

Direct microtest with fp32 fields plus fp64 metrics/transport reported:

```text
warning_count 0
ru float64
rv float64
rom float64
scalar float64
mass_full float64
div3 float64
divw float64
```

This part is closed.

### Same-class precision-drop risk remains in non-flux operators if the entry state is fp32

I did not find another flux-advection-style scatter bug on the already-pre-upcast path. However, the actual operational-entry bug leaves `theta/u/v` at default fp32 during initialization, and several core operators intentionally allocate or cast from the incoming field dtype:

- `vertical_implicit_solver.build_epssm_column_coefficients` casts `theta_coefficient` and `dz_m` to `theta.dtype`, then allocates coefficient buffers at `theta.dtype` at `src/gpuwrf/dynamics/vertical_implicit_solver.py:36` through `src/gpuwrf/dynamics/vertical_implicit_solver.py:52`.
- `conservative_constant_k_diffusion_tendency` casts `dz_m` to `field.dtype` and allocates vertical `flux` at `field.dtype` at `src/gpuwrf/dynamics/explicit_diffusion.py:196` through `src/gpuwrf/dynamics/explicit_diffusion.py:200`.
- `constant_k_diffusion_tendency` starts from `jnp.zeros_like(field)` at `src/gpuwrf/dynamics/explicit_diffusion.py:145`.

These are fine if `force_fp64` actually upcasts the scan carry before the first step. They are not safe evidence that the current public path is fp64, because the public path currently initializes `theta/u/v` as fp32 and then fails when the first step returns fp64.

Other inspected named modules did not show a new blocker beyond this entry-state issue: `advance_w` pressure-gradient scratch is keyed to `p.dtype` at `src/gpuwrf/dynamics/core/advance_w.py:114` through `src/gpuwrf/dynamics/core/advance_w.py:128` and the real pressure fields are already fp64; `acoustic_wrf` `dpn` buffers are keyed to `pressure_perturbation.dtype` at `src/gpuwrf/dynamics/acoustic_wrf.py:296` through `src/gpuwrf/dynamics/acoustic_wrf.py:334`; `mu_t_advance` zero rows key off mass tendency at `src/gpuwrf/dynamics/mu_t_advance.py:179` through `src/gpuwrf/dynamics/mu_t_advance.py:184`; `damping` keys the Rayleigh profile to `w.dtype` at `src/gpuwrf/dynamics/damping.py:74` through `src/gpuwrf/dynamics/damping.py:81`.

### Numeric regression evidence: artifact is green, but I did not rewrite close-gate proofs

`proofs/sprintU/fp64_regression_gate.txt:1` through `proofs/sprintU/fp64_regression_gate.txt:3` reports `4 passed in 556.75s`. The current committed diagnostics show the expected pass values:

- Warm bubble PASS: rise `1924.3475674059687`, max|w| `11.680187856366164`, theta' max `1.9200603163246228`, mass drift `0.0`.
- Straka PASS: front `14150.0`, theta' min `-9.970995032353471`, max|w| `14.574919073012929`, mass drift `2.2509470252373515e-9`.

I did not rerun `tests/idealized/test_dycore_close_gate.py -m close_gate` because that test archives proof files into `proofs/sprintU/close_gate` at `tests/idealized/test_dycore_close_gate.py:40` through `tests/idealized/test_dycore_close_gate.py:43`, and this reconfirm prompt says only to write this report. This is not the blocker; P0-1 is.

### P0-2 deferral is acceptable as documented

The deferral is now explicit and honest. `DYCORE_STATUS.md` says the implemented deformation operator is only the 2D one-row `u/w` subcase and full 3D `u/v/w` deformation is deferred to Phase B at `proofs/f7/DYCORE_STATUS.md:18` through `proofs/f7/DYCORE_STATUS.md:32`. It also states the operational real-case path uses `diff_6th_opt=2`, does not enable `km_opt` deformation diffusion, and defers full `defor11/22/33/12/13/23` plus `horizontal_diffusion_u/v/w_2` to Phase B at `proofs/f7/DYCORE_STATUS.md:51` through `proofs/f7/DYCORE_STATUS.md:66`.

The real-case proof JSON agrees that `deformation_momentum_diffusion_P0_2` is false at `proofs/sprintU/real_case_smoke.json:8`, while the operational namelist sets `diff_6th_opt=2` at `src/gpuwrf/integration/daily_pipeline.py:187` through `src/gpuwrf/integration/daily_pipeline.py:190`.

So P0-2 is acceptable as a documented deferral, not as an implemented full WRF u/v/w deformation tensor.

## Commands Run

```text
sed -n '1,220p' PROJECT_CONSTITUTION.md
sed -n '1,220p' AGENTS.md
sed -n '1,260p' .agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-reconfirm-prompt.md
sed -n '1,260p' .agent/sprints/2026-05-29-sprintU-operationalize-dycore/worker-report.md
sed -n '1,260p' .agent/skills/validating-physics/SKILL.md
git status --short
git branch --show-current
git log --oneline -8
git show --stat --oneline 3ee8d94
git show --stat --oneline bc5660b
rg/nl/sed/jq inspections of the cited source, proof, and test files
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... x64 import audit ... PY
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... real-case dtype audit ... PY
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... flux-advection dtype/warnings microtest ... PY
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... one-step run_forecast_operational real-case smoke ... PY
```

## Close Decision

Do not close Sprint U yet. Required remediation is small and specific: the three operational entry points must initialize the carry with `force_fp64=bool(namelist.force_fp64)`, then the real-case public entry (`daily_pipeline -> run_forecast_operational`) must pass a one-step and multi-step pure-dycore dtype/finite proof without manual pre-upcast bypass. The existing direct `_physics_boundary_step` proofs can remain supporting evidence, but they cannot substitute for the operational API.

SPRINTU_RECONFIRM_COMPLETE
**CLOSE-REJECTED-pending-P0-1-operational-entry-force-fp64** confidence 9/10
