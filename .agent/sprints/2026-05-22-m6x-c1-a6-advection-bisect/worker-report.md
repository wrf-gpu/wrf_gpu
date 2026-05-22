# c1-A6 Advection Internals Bisect Worker Report

## Objective

Bisect the proven advection-only c1 dycore failure at per-operator granularity without modifying `src/gpuwrf/dynamics/advection.py`.

Baseline cited from `/tmp/wrf_gpu2_main_cp/.agent/sprints/2026-05-22-m6x-empirical-bisection/worker-report.md`: acoustic-only stayed finite through 360 steps; advection-only first produced nonfinites at step 30.

## Per-Probe Results

All probes used the raw pre-sanitize candidate state with sanitize, physics, boundary replay, acoustic, and mu-continuity disabled unless the row says otherwise. Probe artifacts are under `artifacts/m6/performance/c1_a6_advection_bisect/`.

| Probe | Enabled advection subset | First nonfinite step | First nonfinite fields | Proof object |
|---|---|---:|---|---|
| `phase1_advection_only` | scalar + momentum, horizontal + vertical | 30 | `p/ph/qv/theta/u/v/w` | `phase1_advection_only.json` |
| `phase1_scalar_only` | scalar only | 189 | `theta` | `phase1_scalar_only.json` |
| `phase1_momentum_only` | momentum only | 30 | `u/v/w` | `phase1_momentum_only.json` |
| `phase1_horizontal_only` | horizontal only | 30 | `p/ph/qv/theta/u/v/w` | `phase1_horizontal_only.json` |
| `phase1_vertical_only` | vertical only | 112 | `qv/theta/u/v` | `phase1_vertical_only.json` |
| `phase2_horizontal_momentum_only` | horizontal momentum only | 30 | `u/v/w` | `phase2_horizontal_momentum_only.json` |
| `phase2_horizontal_scalar_only` | horizontal scalar only | none through 360 | none | `phase2_horizontal_scalar_only.json` |
| `phase3_horizontal_u_only` | horizontal `u` tendency only | 30 | `u` | `phase3_horizontal_u_only.json` |
| `phase3_horizontal_v_only` | horizontal `v` tendency only | 41 | `v` | `phase3_horizontal_v_only.json` |
| `phase3_horizontal_w_only` | horizontal `w` tendency only | none through 360 | none | `phase3_horizontal_w_only.json` |
| `phase3_horizontal_momentum_no_u` | horizontal `v+w`, no `u` | 41 | `v/w` | `phase3_horizontal_momentum_no_u.json` |
| `phase4_horizontal_u_x_only` | `u` x/self term only | 31 | `u` | `phase4_horizontal_u_x_only.json` |
| `phase4_horizontal_u_y_only` | `u` y/cross term only | none through 360 | none | `phase4_horizontal_u_y_only.json` |

Current `advection.py` does not advect `qc/qr/qi/qs/qg`; scalar advection in this file is `theta`, `qv`, perturbation pressure `p - pb`, and horizontal `ph`.

## Bisection Conclusion

The step-30 advection failure is in **momentum advection**, not scalar advection.

The step-30 path is **horizontal**, not vertical. Horizontal scalar-only is finite through 360 steps; horizontal momentum-only reproduces step 30.

The smallest field-level reproducer is **horizontal `u`-face momentum advection**. `u` horizontal alone reproduces step 30. Removing `u` from horizontal momentum delays the first nonfinite to step 41.

The tightest stencil-level evidence points to the **x/self term in `advect_u_face`**:

- `u` x/self term only: first nonfinite at step 31.
- `u` y/cross term only: finite through 360.
- `u` x+y combined: first nonfinite at step 30.

So the defect should be treated as `advect_u_face` horizontal x/self advection with y-cross coupling accelerating the failure by one step.

## Recommended Fix Hint for c1-A7

Target `src/gpuwrf/dynamics/advection.py:373-383`, especially:

- `src/gpuwrf/dynamics/advection.py:381`: `state.u * derivative5_upwind(state.u, state.u, _dx(grid), axis=2)`
- `src/gpuwrf/dynamics/advection.py:382`: y-cross term is not sufficient alone, but accelerates the x/self failure when combined.

Do not start with scalar flux form, vertical `derivative3_upwind_vertical`, `ph`, or hydrometeor fields for the step-30 bug.

## Files Changed

- `scripts/m6_full_domain_batching.py`
- `src/gpuwrf/coupling/driver.py`
- `.agent/sprints/2026-05-22-m6x-c1-a6-advection-bisect/worker-report.md`
- `artifacts/m6/performance/c1_a6_advection_bisect/*.json`

## Commands Run

```bash
python -m py_compile scripts/m6_full_domain_batching.py src/gpuwrf/coupling/driver.py
git diff --check
PYTHONPATH=src python scripts/m6_full_domain_batching.py --help
PYTHONPATH=src pytest -q tests/test_m4_advection.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --hours 0.008333333333333333 --probe-label smoke_advection_only --probe-log-interval 1
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --hours 1 --probe-label phase1_advection_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-momentum --hours 1 --probe-label phase1_scalar_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --hours 1 --probe-label phase1_momentum_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-horizontal --hours 1 --probe-label phase1_vertical_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-vertical --hours 1 --probe-label phase1_horizontal_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --hours 1 --probe-label phase2_horizontal_momentum_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-momentum --disable-advection-vertical --hours 1 --probe-label phase2_horizontal_scalar_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --disable-advection-v --disable-advection-w --hours 1 --probe-label phase3_horizontal_u_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --disable-advection-u --disable-advection-w --hours 1 --probe-label phase3_horizontal_v_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --disable-advection-u --disable-advection-v --hours 1 --probe-label phase3_horizontal_w_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --disable-advection-u --hours 1 --probe-label phase3_horizontal_momentum_no_u --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --disable-advection-v --disable-advection-w --disable-advection-y --hours 1 --probe-label phase4_horizontal_u_x_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --disable-advection-v --disable-advection-w --disable-advection-x --hours 1 --probe-label phase4_horizontal_u_y_only --probe-log-interval 30
```

## Proof Objects Produced

- `artifacts/m6/performance/c1_a6_advection_bisect/smoke_advection_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase1_advection_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase1_scalar_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase1_momentum_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase1_vertical_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase1_horizontal_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase2_horizontal_momentum_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase2_horizontal_scalar_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase3_horizontal_u_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase3_horizontal_v_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase3_horizontal_w_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase3_horizontal_momentum_no_u.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase4_horizontal_u_x_only.json`
- `artifacts/m6/performance/c1_a6_advection_bisect/phase4_horizontal_u_y_only.json`

## Unresolved Risks

- The x-only `u` self-advection term fails at step 31, while x+y combined fails at step 30. That means the smallest exact step-30 reproducer is `advect_u_face` horizontal x+y; the strongest single-term suspect is the x/self term.
- Bisection instrumentation intentionally transfers one scalar nonfinite count per step to stop at the first raw failure. This is diagnostic-only and is documented in every probe JSON.
- The custom bisection RK path mirrors production RK for disabled components, but it is not a production timestep path.

## Next Decision Needed

Dispatch c1-A7 to fix only the `advect_u_face` horizontal `u` x/self advection path, with the y-cross interaction kept in the regression probe.
