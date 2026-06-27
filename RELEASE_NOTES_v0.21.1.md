# Release notes — wrf_gpu v0.21.1

**v0.21.1 is a focused stability point release off v0.21.0.** It fixes a lateral-boundary
instability that affected the most extreme steep-terrain (Mont-Blanc-class, ~1000 m of relief
per grid cell) configurations.

## What changed

On the most extreme steep-terrain fixtures, v0.21.0 could run without an immediate
floating-point overflow, but the coarse-domain solution was physically invalid: the physical
vertical velocity (`W`) diverged at the outer specified-boundary corner while the interior
stayed bounded. The growth was approximately `44 → 288 → 2066 → 14531 → 99325 m/s`.

v0.21.1 fixes the boundary mechanism, WRF-faithfully:

- standalone `wrfbdy` roots now use WRF specified-boundary update cadence and edge advection
  degradation;
- hydrometeor and number scalar `wrfbdy` leaves are decoded and applied when the boundary file
  actually carries the corresponding WRF side base/tendency arrays;
- specified-domain zero-gradient `W` (`zero_grad_bdy`) is applied to the reconstructed physical
  `W` by solving the acoustic work value that the small-step finish reconstructs back to WRF's
  copied physical boundary value;
- WRF's corner source-index behavior is preserved (y-side rows own the corners and use the
  nearest interior source column).

This is a WRF boundary-copy semantics fix, **not** a limiter. There is no `W` masking,
`nan_to_num`, finite guard, or value clip/clamp in the fix.

## Validation

- Mont-Blanc-class native-dt run (2 h): coarse-domain max `|W|` bounded at
  `2.31, 4.13, 3.57, 3.62, 3.58 m/s`; inner 1 km domain max `|W| ≤ 8.04 m/s`.
- Steep-Alpine 3-domain nest: no regression vs CPU-WRF — T / U / V correlations
  `0.999995 / 0.999839 / 0.999751`.
- Canary 3-domain nest: no regression (all domains finite, all outputs present).
- Focused CPU test suite: 40 passed, 3 skipped.

## Scope

v0.21.1 is `v0.21.0 + the Mont-Blanc boundary fix` only. It does not change the v0.21.0
AOT / fused-cache behavior. Longer-horizon memory and broader performance work continue in the
v0.22 line.
