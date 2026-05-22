# c1-A8 Worker Report -- Vertical Momentum Eta-Metric Bisect

## Objective

Bisect the residual c1-A7 vertical momentum failure, apply one surgical WRF-cited fix per probe, and verify whether the fix closes the 1h coupled stability gate.

## Files Changed

- `src/gpuwrf/dynamics/advection.py`
- `tests/test_m4_advection.py`
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json`

## Implementation

The failing single-component path was vertical `w` momentum: baseline `c1_a8_phase1_vertical_w_only` failed at step 158, while vertical `u` and `v` alone were finite through 360. The accepted fix changes the vertical momentum flux-divergence metric sign for `u`, `v`, and `w` when applying WRF-form vertical momentum advection on this codebase's positive geometric `dz`.

WRF basis: `module_advect_em.F` vertical momentum branches subtract `rdzw/rdzu * (vflux(k+1)-vflux(k))`, while `module_initialize_real.F:3733-3734` defines `dnw = znw(k+1)-znw(k)` and `rdnw = 1/dnw`; `znw` decreases upward. The reduced c1 positive-`dz` geometry therefore needs the opposite vertical metric sign for vertical momentum divergence. Existing WRF `-vel` upwind-selection behavior is preserved.

Rejected probe: zeroing physical/top-bottom `w` or just negating vertical transport velocity regressed the isolated `w` probe, so those changes were not kept.

## Commands Run

```bash
python -m py_compile src/gpuwrf/dynamics/advection.py
PYTHONPATH=src pytest -q tests/test_m4_advection.py::test_momentum_vertical_fluxes_use_eta_metric_sign tests/test_m4_advection.py::test_momentum_self_advection_is_closed_flux_divergence tests/test_m6x_fallback_c1_coupling.py::test_bisection_scalar_disabled_probe_uses_production_momentum_flux_form
PYTHONPATH=src pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-horizontal --hours 1 --probe-label c1_a8_phase1_vertical_momentum_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-horizontal --disable-advection-v --disable-advection-w --hours 1 --probe-label c1_a8_phase1_vertical_u_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-horizontal --disable-advection-u --disable-advection-w --hours 1 --probe-label c1_a8_phase1_vertical_v_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-horizontal --disable-advection-u --disable-advection-v --hours 1 --probe-label c1_a8_phase1_vertical_w_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-horizontal --disable-advection-u --disable-advection-v --hours 1 --probe-label c1_a8_post_fix_vertical_w_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-horizontal --hours 1 --probe-label c1_a8_post_fix_vertical_momentum_only --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-w --hours 1 --probe-label c1_a8_post_fix_full_uv_no_w --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --hours 1 --probe-label c1_a8_post_fix_full_momentum_scalar_disabled --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a8_post_fix_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a8_post_fix_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
git diff --check
```

## Proof Objects Produced

- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_phase1_vertical_momentum_only.json`: baseline vertical momentum failed at step 157.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_phase1_vertical_u_only.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_phase1_vertical_v_only.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_phase1_vertical_w_only.json`: failed at step 158.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_post_fix_vertical_w_only.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_post_fix_vertical_momentum_only.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_post_fix_full_uv_no_w.json`: finite through 360 after extending eta metric sign to u/v.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a8_post_fix_full_momentum_scalar_disabled.json`: still nonfinite at step 188.
- `artifacts/m6x-fallback-c1/c1_a8_post_fix_1h.json`: verdict FAIL; speedup 44.33x, Tier-2 fail, sanitize fired on 313/360 steps.
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json`: tracked Tier-2 artifact from the red 1h verdict.

## Result

Status: **PARTIAL / NOT GREEN**.

The isolated vertical momentum bug is fixed: vertical momentum only and `w` vertical only are finite through 360 steps. The broader scalar-disabled momentum probe is improved versus c1-A7's step-106 failure but still fails at step 188 when all momentum components are coupled. The 1h coupled path remains red, so I did not run 24h.

## Unresolved Risks

- Remaining failure requires cross-component momentum coupling: `w` alone and `u+v` without `w` are stable, but `u+w` and `v+w` pair probes still fail around 193-197 steps.
- The 1h coupled sanitize rate remains high at 86.94%, and final state diagnostics hit clip bounds/nonfinite leaves.
- The transfer audit remains 167,904 bytes post-init, unchanged by this patch.

## Next Decision Needed

c1-A8 did not close M6.x. Per the sprint prompt, manager/user should decide whether to dispatch another cross-component momentum-coupling investigation, escalate to c2, or pivot the end-goal criteria.
