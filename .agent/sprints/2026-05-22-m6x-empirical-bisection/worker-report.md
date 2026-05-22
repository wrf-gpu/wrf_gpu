# M6x Empirical Bisection Worker Report

## Objective

Empirically isolate the first raw nonfinite source in the coupled d02 timestep by progressively disabling boundary replay, physics, and dycore subcomponents under sanitize-disabled 1 h probes.

## Per-Probe Results

All probes used `dt_s=10`, `n_acoustic=2`, `hours=1`, raw candidate state before sanitize, and Gen2 run `20260520_18z_l3_24h_20260521T045847Z`.

| Phase | Probe | Enabled subset | `first_nonfinite_step` | Proof object | Diagnosis |
|---|---|---|---:|---|---|
| 1 | baseline | dycore + Thompson + MYNN + surface + RRTMG + boundary | 25 | `artifacts/m6/performance/empirical_bisection/phase1_baseline.json` | Full coupled path fails before radiation cadence; first nonfinite fields include `theta/qv/u/v/qke` plus physics outputs. |
| 1 | no boundary | dycore + physics, boundary disabled | 25 | `artifacts/m6/performance/empirical_bisection/phase1_no_boundary.json` | Boundary replay is not the trigger. |
| 1 | no physics | dycore + boundary, all physics disabled | 26 | `artifacts/m6/performance/empirical_bisection/phase1_no_physics.json` | Physics is not required for the raw failure. |
| 1 | dycore only | dycore only, physics and boundary disabled | 25 | `artifacts/m6/performance/empirical_bisection/phase1_dycore_only.json` | Dycore alone is sufficient. Phase 2 skipped. |
| 3 | acoustic only | acoustic enabled, advection and mu-continuity disabled | none through 360 | `artifacts/m6/performance/empirical_bisection/phase3_acoustic_only.json` | Acoustic alone is finite for the 1 h probe. Phase 4 skipped. |
| 3 | advection only | advection enabled, acoustic and mu-continuity disabled | 30 | `artifacts/m6/performance/empirical_bisection/phase3_advection_only.json` | Advection alone is sufficient to generate raw nonfinites. |
| 3 | mu-continuity only | mu-continuity enabled, advection and acoustic disabled | none through 360 | `artifacts/m6/performance/empirical_bisection/phase3_mu_only.json` | No raw nonfinite from the mu-only path in this checked-out code. |

## Bisection Conclusion

The instability is in the dycore advection path. It is not caused by boundary replay or physics. Acoustic-only stays finite, but full dycore fails at step 25 while advection-only fails at step 30, so acoustic coupling accelerates an advection-generated instability rather than being the standalone source.

## Recommended Fix Hint

Next worker should target `src/gpuwrf/dynamics/advection.py`, starting at `compute_advection_tendencies` (`src/gpuwrf/dynamics/advection.py:262`) and then bisecting its fanout:

- scalar advection: `advect_mass_scalar` and `derivative5_upwind` (`src/gpuwrf/dynamics/advection.py:76`, `src/gpuwrf/dynamics/advection.py:171`)
- face-velocity advection: `advect_u_face`, `advect_v_face`, `advect_w_face` (`src/gpuwrf/dynamics/advection.py:202`, `src/gpuwrf/dynamics/advection.py:215`, `src/gpuwrf/dynamics/advection.py:228`)

No single line was proven by this sprint; the empirical component target is advection.

## Files Changed

- `scripts/m6_full_domain_batching.py`
- `src/gpuwrf/coupling/driver.py`
- `.agent/sprints/2026-05-22-m6x-empirical-bisection/worker-report.md`
- `artifacts/m6/performance/empirical_bisection/*.json`

## Commands Run

- `python -m py_compile scripts/m6_full_domain_batching.py src/gpuwrf/coupling/driver.py`
- `PYTHONPATH=src pytest -q tests/test_m6_tier2_coupled.py tests/test_m6_dycore_cap_lift.py`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --hours 1 --probe-label phase1_baseline --probe-log-interval 30`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-boundary --hours 1 --probe-label phase1_no_boundary --probe-log-interval 30`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --hours 1 --probe-label phase1_no_physics --probe-log-interval 30`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --hours 1 --probe-label phase1_dycore_only --probe-log-interval 30`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-advection --disable-mu-continuity --hours 1 --probe-label phase3_acoustic_only --probe-log-interval 30`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --hours 1 --probe-label phase3_advection_only --probe-log-interval 30`
- `PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-advection --disable-acoustic --hours 1 --probe-label phase3_mu_only --probe-log-interval 30`

## Proof Objects Produced

- `artifacts/m6/performance/empirical_bisection/phase1_baseline.json`
- `artifacts/m6/performance/empirical_bisection/phase1_no_boundary.json`
- `artifacts/m6/performance/empirical_bisection/phase1_no_physics.json`
- `artifacts/m6/performance/empirical_bisection/phase1_dycore_only.json`
- `artifacts/m6/performance/empirical_bisection/phase3_acoustic_only.json`
- `artifacts/m6/performance/empirical_bisection/phase3_advection_only.json`
- `artifacts/m6/performance/empirical_bisection/phase3_mu_only.json`

## Unresolved Risks

- The role prompt expected a known step-45 symptom, but this worktree measured step 25 on the current checked-out code. The report uses the observed proof objects, not the prior expectation.
- This code’s mu-continuity path has no observable standalone update in the probe; `mu-only` is therefore a negative control for this worktree, not proof that future c1-style mu continuity is stable.
- The bisection probe intentionally transfers one scalar nonfinite count per step to stop at the first raw failure. This is diagnostic-only instrumentation, not a production timestep path.

## Next Decision Needed

Dispatch the next worker to instrument or fix `src/gpuwrf/dynamics/advection.py` at per-field/per-operator granularity.
