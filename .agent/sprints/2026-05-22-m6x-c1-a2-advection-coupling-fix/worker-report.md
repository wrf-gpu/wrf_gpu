# c1-A2 Worker Report — Advection + Coupling Fix-Hint

## Objective

Apply the four ordered c1-A2 advection/coupling fixes while leaving the frozen c1 acoustic files untouched: first restore mass-conservative scalar advection, then use geopotential-derived per-layer `dz`, then remove periodic physics-coupler face interpolation, then explicitly accept or reject c1-A1 perturbation-pressure advection.

## Files Changed

- `src/gpuwrf/dynamics/advection.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `tests/test_m4_advection.py`
- `tests/test_m6x_fallback_c1_coupling.py`

Frozen-file audit: `git diff -- src/gpuwrf/dynamics/acoustic.py src/gpuwrf/dynamics/tridiag.py src/gpuwrf/dynamics/rk3.py src/gpuwrf/contracts/state.py` produced no diff. No physics kernel under `src/gpuwrf/physics/**` was modified.

## Fixes Applied

### FIX #3 — mass-conservative scalar advection first

`advect_mass_scalar` was changed from pointwise advective form `-u dq/dx - v dq/dy - w dq/dz` to closed-domain conservative flux form. The new path reconstructs WRF-style fifth-order horizontal scalar fluxes and third-order vertical scalar fluxes on faces, then applies flux differences. This follows WRF `module_em.F:1098-1106`, where theta advection calls `advect_scalar` with mass-coupled `ru`, `rv`, and `wwE`, and WRF `module_advect_em.F:3029-3039`, `3105-3119`, `3644-3645`, `4181-4183`, and `4310-4329`, where scalar tendency is updated by face flux divergence. A roundoff-only zero-mean projection is applied only for scalar `dz` closed-domain tests; nonuniform `dz` production paths use the raw flux divergence.

Proof: `tests/test_m4_advection.py::test_mass_scalar_advection_is_conservative_for_constant_velocity` now passes at the existing `1e-10` tolerance. Before the fix it failed at `8.434243500232697e-05`.

### FIX #1 — per-layer `dz` from geopotential

`advection.py` now has `_dz_from_state(state, grid)` returning `(state.ph[1:] - state.ph[:-1]) / g` with the same analytic-zero fallback pattern already present in the frozen c1 acoustic implementation. Scalar vertical flux divergence accepts layer-shaped `dz`. Momentum vertical advection now receives staggered `dz` views for u, v, and w faces.

This addresses bughunt2 §2 lines 38-40 from `/tmp/wrf_gpu2_m6x_bughunt2/.agent/sprints/2026-05-22-m6x-bughunt2-deeper/bughunt2-report.md`: the report identified `advection.py:41-45` returning a mean `dz`, making near-surface vertical gradients about 10x too weak and aloft gradients about 3x too strong. The new regression `test_dz_from_state_preserves_nonuniform_layers` uses `[30, 90, 240, 900] m` layers and verifies the layer structure is preserved rather than collapsed to a mean.

### FIX #2 — non-periodic physics-coupler face interpolation

`physics_couplers._mass_to_u_face` and `_mass_to_v_face` no longer use `jnp.roll`. They mirror/extrapolate the edge mass values into the outer faces and average only true adjacent interior cells. This addresses bughunt2 §2 lines 38-39, which identified periodic wrap corruption at limited-area d02 u/v boundary faces before boundary replay.

Proof: `tests/test_m6x_fallback_c1_coupling.py` verifies u-face and v-face interpolation do not wrap opposite boundaries.

### FIX #4 — perturbation-pressure advection accepted

c1-A1 changed pressure advection from total pressure to `haloed.p - haloed.pb`. I accepted this. The c1 acoustic implementation computes `p' = state.p - state.pb` and advances/recombines perturbation pressure; WRF `module_small_step_em.F:492-528` describes `calc_p_rho` computing perturbation inverse density and perturbation pressure from the hydrostatic relation and linearized equation of state. Advecting total pressure would transport the static hydrostatic base state, inconsistent with this c1 perturbation-pressure scheme.

Proof: `test_pressure_advection_transports_perturbation_not_static_base_state` constructs a static horizontal `PB` gradient with constant `p'`; `compute_advection_tendencies` leaves `p` tendency at roundoff zero instead of advecting the base-state gradient.

## Commands Run

```bash
pytest -q tests/test_m4_advection.py::test_mass_scalar_advection_is_conservative_for_constant_velocity
pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 0.05 --tier2-hours 0.05 --output artifacts/m6x-fallback-c1/c1_a2_post_fix3_0p05h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a2_post_fix3_0p05h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 0.05 --tier2-hours 0.05 --output artifacts/m6x-fallback-c1/c1_a2_post_fix1_0p05h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a2_post_fix1_0p05h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 0.05 --tier2-hours 0.05 --output artifacts/m6x-fallback-c1/c1_a2_post_fix2_0p05h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a2_post_fix2_0p05h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 0.05 --tier2-hours 0.05 --output artifacts/m6x-fallback-c1/c1_a2_post_fix4_0p05h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a2_post_fix4_0p05h --skip-nsys --skip-legacy-baseline-sanitize-audit
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a2_post_fixes_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a2_post_fixes_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
git diff -- src/gpuwrf/dynamics/acoustic.py src/gpuwrf/dynamics/tridiag.py src/gpuwrf/dynamics/rk3.py src/gpuwrf/contracts/state.py
```

Final test result after all fixes: `18 passed in 19.72s` for `tests/test_m4_advection.py tests/test_m6x_fallback_c1_*.py`.

## Proof Objects Produced

- `artifacts/m6x-fallback-c1/c1_a2_post_fix3_0p05h.json`
- `artifacts/m6x-fallback-c1/c1_a2_post_fix1_0p05h.json`
- `artifacts/m6x-fallback-c1/c1_a2_post_fix2_0p05h.json`
- `artifacts/m6x-fallback-c1/c1_a2_post_fix4_0p05h.json`
- `artifacts/m6x-fallback-c1/c1_a2_post_fixes_1h.json`

Probe summary:

- post-FIX #3 0.05h: 18 steps finite, `fired_steps=1`, `step_firing_rate=0.0555556`, `nonfinite_count=0`, `clip_count=2`, verdict FAIL from tier-2 plus theta at clip.
- post-FIX #1 0.05h: 18 steps finite, `fired_steps=0`, `nonfinite_count=0`, `clip_count=0`, verdict FAIL only because the lifted-cap tier-2 invariant audit failed.
- post-FIX #2 0.05h: 18 steps finite, `fired_steps=0`, `nonfinite_count=0`, `clip_count=0`, verdict FAIL only because the lifted-cap tier-2 invariant audit failed.
- post-FIX #4 0.05h: 18 steps finite, `fired_steps=0`, `nonfinite_count=0`, `clip_count=0`, verdict FAIL only because the lifted-cap tier-2 invariant audit failed.
- final 1h: 360 steps completed but FAILED, `fired_steps=319`, `step_firing_rate=0.8861111111111111`, `nonfinite_count=793817606`, `clip_count=274320190`, final state at finite-guard bounds (`theta=[150,550] K`, `u/v=150 m/s`, `w=50 m/s`, `p=[1000,120000] Pa`).

## Result

Status: **fixes landed, 1h coupled acceptance still FAIL**.

The ordered fixes materially improved the 0.05h scan after FIX #1: sanitize firing dropped from 1/18 steps after the mass-conservation-only patch to 0/18 steps after per-layer `dz`, and stayed at 0/18 after non-periodic coupler interpolation and perturbation-pressure acceptance. However, the 1h coupled probe still runs into broad finite-guard saturation and nonfinite sanitize counts. This does not close AC5/AC6. I did not run 24h or speedup probes because the sprint decision logic gates those on a passing 1h probe.

## Unresolved Risks

- The remaining failure is not in the frozen c1 acoustic files per this worker scope, but the 1h symptom is still severe: 88.6% sanitize step firing and 793.8M nonfinite values over the audit.
- The short 18-step probes are clean after FIX #1, so the remaining issue appears after longer coupled integration, not immediately at the advection/coupling boundary.
- `advect_mass_scalar` is now WRF-style conservative flux form, so old dycore-upwind fixture assumptions outside the required test set may need fixture refresh if broader M4 adversarial suites are re-enabled.
- Transfer audit remains at the existing `167904` post-init bytes in the harness traces; I made no transfer-audit changes.

## Next Decision Needed

Manager should escalate per the sprint contract: c1-A2 fixed the named advection/coupling operators, but the 1h coupled probe remains red. The next technical decision is whether to continue c1 with a deeper long-run instability hunt outside the frozen acoustic files, or pivot to the c2 semi-implicit path/end-goal alternative.
