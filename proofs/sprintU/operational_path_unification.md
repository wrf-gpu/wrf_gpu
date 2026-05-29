# Sprint U / P0-1 — Operational/real-case path unified with the F7-closed dycore

Date: 2026-05-29
Branch: `worker/opus/f7d-pressure-mass-fix`

## Finding being closed (GPT pre-close P0-1)

> The passing dycore path is not the operational/default path.
> `daily_pipeline._build_real_case` built an `OperationalNamelist.from_grid(...)`
> with `use_flux_advection=False`, `force_fp64=False`, `w_damping=0`,
> `damp_opt=0`, `const_nu=0` — BYPASSING the just-closed F7 advection fixes.
> `_augment_large_step_tendencies` only takes the F7 flux-form advection branch
> when `use_flux_advection` is True; otherwise it uses the pre-F7 primitive
> `compute_advection_tendencies` path.

## Root cause

`daily_pipeline._build_real_case` (`src/gpuwrf/integration/daily_pipeline.py:156`)
constructed the real-case namelist with the bare defaults, so the real path ran
the pre-F7 primitive advection, fp32, no WRF damping, and the rigid-lid `advect_w`
(missing the open-top face). The idealized PASS path set the F7 controls
explicitly in `ic_generators/idealized.py`; the operational path did not.

## Fix (before → after)

`_build_real_case` now constructs the real-case namelist with the **same F7
operators** the idealized gates use:

| control | before (`from_grid` default) | after (real-case) | WRF basis |
|---|---|---|---|
| `use_flux_advection` | `False` (primitive `u du/dx`) | **`True`** (advect_u/v/w + advect_scalar, h=5/v=3, incl. F7N vertical-momentum sign fix) | `module_advect_em.F:126/1530/3029/4364` |
| `force_fp64` | `False` (fp32-gated) | **`True`** | F7 acoustic-solve fp64 requirement (ADR-007 defers perf downcast) |
| `diff_6th_opt` / `diff_6th_factor` | `0` / `0.12` | **`2`** / `0.12` | operational d02 numerical filter, `module_big_step_utilities_em.F:6504` |
| `w_damping` | `0` | **`1`** | Gen2 d02 namelist |
| `damp_opt` | `0` | **`3`** (implicit Rayleigh) | Gen2 d02 namelist |
| `zdamp` / `dampcoef` | `5000` / `0.0` | `5000` / **`0.2`** | Gen2 d02 namelist |
| `top_lid` | `False` | `False` (OPEN top → advect_w top-face P1-5 active) | `module_advect_em.F:6014-6028` |
| `disable_guards` | `False` | `False` (production guard safety net kept) | — |

The namelist metadata block in `_build_real_case` now records every one of these
so the pipeline run JSON is self-documenting.

## Evidence

### 1. The operational entry point runs the SAME dycore as the idealized PASS path (bitwise)

`run_forecast_operational(...)` and the idealized harness segment
(`ic_generators.idealized._run_segment_jit`) over the same 50 warm-bubble steps:

```
theta linf = 0.0   w linf = 0.0   SAME DYCORE (bitwise) = True
```

The idealized harness was never a "special dycore" — it calls the identical
`_physics_boundary_step` the operational scan calls. The only difference was the
namelist, which is now unified.

### 2. The full idealized warm bubble PASSES through the operational entry point

Running the full 500 s warm bubble through `run_forecast_operational` (NOT the
harness) reproduces the closed F7N verdict 6/6:

| check | value | passed |
|---|---|---|
| all_snapshots_finite | 1.0 | ✅ |
| theta_prime_max_500s | 1.920 K | ✅ |
| max_abs_w_500s | 11.68 m/s | ✅ |
| thermal_rise_500s | 1924.3 m | ✅ |
| horizontal_drift_500s | 1.8e-12 m | ✅ |
| relative_mass_drift | 0.0 | ✅ |

### 3. Real Canary d02 case builds with the F7 operators and the dycore is finite

`scripts/sprintU_real_case_smoke.py` → `proofs/sprintU/real_case_smoke.json`:

* real Canary d02 replay grid **44 × 66 × 159** (3D, multi-row, open top);
* active-operator audit confirms `flux_adv=True, fp64=True, top_lid=False,
  damp_opt=3, w_damping=1, diff6=2`;
* the flux-form branch is provably taken (`flux_vs_primitive_theta_linf > 0`,
  finite);
* 6 operational dycore steps (physics/boundary off to isolate the dycore) leave
  every prognostic finite → `verdict: PASS`.

## Honest scope notes

* The smoke run isolates the dycore (physics/boundary off). The production real
  path keeps physics + lateral boundaries + the guard safety net on; guards-off
  stability of the bare dycore is proven separately (Sprint U P1-6).
* The real-case IC build emits a benign float64→float32 cast warning in the
  replay-state assembly; the operational run forces fp64 via
  `_enforce_operational_precision(..., force_fp64=True)` so the dycore itself is
  fp64. (Tidying the replay-state dtype is out of P0-1 scope.)
* The 3D real-case lateral-boundary / map-factor / terrain operators remain a
  Phase-B gate (flux_advection/rhs_ph still document the periodic/unit-map scope);
  P0-1 proves the *advection/diffusion/damping/precision* operators are unified,
  not that terrain/map-factor coupling is closed.
