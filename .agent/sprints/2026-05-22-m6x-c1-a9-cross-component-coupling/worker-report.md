# c1-A9 Worker Report -- Cross-Component Momentum Coupling

## Objective

Bisect the c1-A8 residual `u+w` / `v+w` cross-component momentum instability, check `u←w` versus `w←u` reflexivity against WRF `advect_u` / `advect_w`, and apply a surgical fix only if supported by proof.

## Files Changed

- `src/gpuwrf/coupling/driver.py`
- `scripts/m6_full_domain_batching.py`
- `tests/test_m6x_fallback_c1_coupling.py`
- `.agent/sprints/2026-05-22-m6x-c1-a9-cross-component-coupling/worker-report.md`

No production advection physics change was retained. WRF mass-flux-first variants were tested and rejected because they did not close the gate and one worsened it.

## Implementation

Added bisection-only switches:

- `--disable-advection-u-vertical-by-w`
- `--disable-advection-v-vertical-by-w`
- `--disable-advection-w-horizontal-by-u`
- `--disable-advection-w-horizontal-by-v`

These switches isolate cross terms without disabling whole momentum tendencies.

WRF audit:

- `module_advect_em.F:1336-1435`: `advect_u` vertical term uses horizontally averaged `rom` and subtracts `rdzw(k) * (vflux(k+1)-vflux(k))`.
- `module_advect_em.F:2813-2920`: `advect_v` vertical term is analogous but includes the `msfvy/msfvx` correction.
- `module_advect_em.F:5004-5018` and `:5384-5429`: `advect_w` horizontal x term uses vertically interpolated `ru` with `fzm/fzp` and subtracts the x flux divergence.
- `module_advect_em.F:10685-10689`: WRF defines `fzm/fzp` from adjacent `dnw/dn` layer metrics.

Key finding: the c1-A8 "u+w" / "v+w" probes were not pure two-component probes. `--disable-advection-v` disables the `v` tendency but did not disable `v` as an advecting velocity in `w←v`; likewise `--disable-advection-u` did not disable `w←u`.

## Commands Run

```bash
python -m py_compile src/gpuwrf/coupling/driver.py scripts/m6_full_domain_batching.py tests/test_m6x_fallback_c1_coupling.py
PYTHONPATH=src pytest -q tests/test_m6x_fallback_c1_coupling.py
PYTHONPATH=src pytest -q tests/test_m4_advection.py tests/test_m6x_fallback_c1_coupling.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-v --disable-advection-u-vertical-by-w --hours 1 --probe-label c1_a9_phase1_uw_no_u_vertical_by_w --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-v --disable-advection-w-horizontal-by-u --hours 1 --probe-label c1_a9_phase1_uw_no_w_horizontal_by_u --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-u --disable-advection-v-vertical-by-w --hours 1 --probe-label c1_a9_phase1_vw_no_v_vertical_by_w --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-u --disable-advection-w-horizontal-by-v --hours 1 --probe-label c1_a9_phase1_vw_no_w_horizontal_by_v --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-v --disable-advection-w-horizontal-by-v --hours 1 --probe-label c1_a9_phase1_uw_true_no_v_terms --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --disable-advection-u --disable-advection-w-horizontal-by-u --hours 1 --probe-label c1_a9_phase1_vw_true_no_u_terms --probe-log-interval 30
PYTHONPATH=src python scripts/m6_full_domain_batching.py --bisection-probe --disable-sanitize --disable-physics --disable-boundary --disable-acoustic --disable-mu-continuity --disable-advection-scalar --hours 1 --probe-label c1_a9_post_fix3_full_momentum_scalar_disabled --probe-log-interval 30
git diff --check
```

Additional rejected-probe commands produced artifacts listed below.

## Proof Objects Produced

- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase1_uw_no_u_vertical_by_w.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase1_uw_no_w_horizontal_by_u.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase1_vw_no_v_vertical_by_w.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase1_vw_no_w_horizontal_by_v.json`: finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase1_uw_true_no_v_terms.json`: true `u+w` finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase1_vw_true_no_u_terms.json`: true `v+w` finite through 360.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase2_full_no_v_vertical_by_w.json`: full momentum finite through 360 when `v←w` is disabled.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase2_full_no_w_horizontal_by_u.json`: full momentum finite through 360 when `w←u` is disabled.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_phase2_full_no_w_horizontal_by_v.json`: full momentum finite through 360 when `w←v` is disabled.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_post_fix3_full_momentum_scalar_disabled.json`: rejected mass-flux variant still failed at step 190.
- `artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_post_fix4_full_momentum_scalar_disabled.json`: rejected horizontal mass-flux variant worsened to step 134.

## Result

Status: **NOT GREEN / NO MODEL FIX RETAINED**.

The pure `u+w` and pure `v+w` cross probes are stable once disabled background advecting terms are actually removed. The full scalar-disabled momentum probe still fails, and one-at-a-time full bisection points to a three-way cross-coupling problem, strongest around `v←w` plus `w←u/w←v`.

## Unresolved Risks

- Full momentum coupling remains nonfinite around step 188-190 on the retained code.
- The WRF `advect_v` vertical branch has a map-factor correction (`msfvy/msfvx`) that the current reduced `GridSpec` cannot represent.
- The current bisection API now exposes true cross-term switches, but the older c1-A8 pair labels should not be used as pure pair evidence.
- AC2/AC3 were not run because AC1/full momentum did not close.

## Next Decision Needed

Manager/user should escalate before further model edits. The next likely decision is whether to add map-factor fields / ADR-backed metric support to the reduced grid, or to open a new sprint on full `u/v/w` kinetic-energy/reflexivity diagnostics with an explicit WRF fixture or analytic energy oracle.
