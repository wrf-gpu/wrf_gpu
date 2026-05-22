# c1-A7 Worker Report -- Momentum Flux Form

## Objective

Convert `advect_u_face`, `advect_v_face`, and `advect_w_face` away from pointwise advective products such as `u * derivative5_upwind(u, u, dx)` and onto WRF-style mass-flux divergence. Keep scalar advection untouched except for using the already-existing scalar flux helpers as references. Run the c1-A7 bisection probe and the 1h coupled probe; run 24h and speedup only if green.

## Files Changed

- `src/gpuwrf/dynamics/advection.py`
- `src/gpuwrf/coupling/driver.py`
- `tests/test_m4_advection.py`
- `tests/test_m6x_fallback_c1_coupling.py`
- Proof JSONs under `artifacts/m6/performance/c1_a6_advection_bisect/`
- Proof JSONs under `artifacts/m6x-fallback-c1/`
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json`

## Implementation

Momentum advection now builds density-weighted face fluxes and returns `-(1/rho_face) div(rho_face * velocity_face * interpolated_momentum)`. The horizontal path reuses the WRF 5th-order flux interpolant already used by c1-A2 scalar advection. The vertical momentum path uses the WRF eta-coordinate sign convention for the upwind interpolant: WRF `module_advect_em.F:4310-4315` passes `-vel` for vertical flux interpolation, and WRF `advect_u` uses the same pattern around `module_advect_em.F:1404-1435`. The w-face rigid top/bottom tendencies are held at zero so vertical advection does not move impermeable lid faces.

The c1-A6 diagnostic driver had copied the old pointwise momentum formulas for filtered probes. Its directional helpers now delegate to the production `advect_*_face_directional` functions, so `--disable-advection-scalar` probes test the fixed momentum operator instead of stale duplicated code.

## Commands Run

```bash
python -m py_compile src/gpuwrf/dynamics/advection.py src/gpuwrf/coupling/driver.py
PYTHONPATH=src pytest -q tests/test_m4_advection.py::test_momentum_self_advection_is_closed_flux_divergence tests/test_m6x_fallback_c1_coupling.py::test_bisection_scalar_disabled_probe_uses_production_momentum_flux_form
PYTHONPATH=src pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-vertical --hours 1 --probe-label c1_a7_horizontal_momentum_post_fix_horizontal_only
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --hours 1 --probe-label c1_a7_horizontal_momentum_post_fix
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a7_post_fix_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a7_post_fix_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
```

## Proof Objects Produced

- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a7_horizontal_momentum_post_fix_horizontal_only.json`: horizontal momentum only stayed finite through 360/360 steps.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a7_horizontal_momentum_post_fix.json`: full momentum with scalar/acoustic/physics/boundary/mu disabled delayed first raw nonfinite to step 106, from the c1-A6 step 30/31 failure.
- `artifacts/m6x-fallback-c1/c1_a7_post_fix_1h.json`: 1h coupled verdict FAIL. Speedup ratio was 43.90x, but Tier-2 failed, final state contained nonfinite leaves, and sanitize fired on 310/360 steps.
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json`: Tier-2 coupled invariant audit for the 1h run.

## Result

Status: **PARTIAL / NOT GREEN**.

The isolated horizontal momentum bug is fixed: the horizontal-only post-fix bisection stays finite through 360 steps. The broader scalar-disabled momentum probe no longer fails at step 30/31, but it still fails at step 106, with residual instability involving vertical momentum coupling. The 1h coupled probe remains red, so I did not run the 24h or speedup follow-on probes.

## Unresolved Risks

- Vertical momentum advection is still not robust enough under the diagnostic configuration with acoustic, scalar, physics, boundary, sanitize, and mu-continuity disabled.
- The production 1h coupled path still saturates finite guards and produces nonfinite leaves, so M6.x cannot close green from this patch.
- The 167,904 byte transfer-audit finding remains unchanged.

## Next Decision Needed

Dispatch should continue from the new evidence: horizontal momentum flux form fixed the c1-A6 smoking gun, but remaining failure is now vertical/coupled momentum stability rather than the isolated `advect_u_face` horizontal x/self term.
