# M6B0-R `calc_coef_w` Defect Analysis

## Finding

The defect is a formulation and unit/staggering mismatch. WRF `calc_coef_w` builds coefficients in hybrid eta pressure-mass units using `MUT`, `c1h/c2h`, `c1f/c2f`, `rdn`, `rdnw`, `c2a`, and `cqw` (`module_small_step_em.F:624-649`). The previous JAX comparator path used the MPAS-family `build_epssm_column_coefficients(theta, dz_m)` in geometric meters and thermodynamic sound-speed proxies (`src/gpuwrf/dynamics/vertical_implicit_solver.py:35-82`). That converted eta-pressure operators into meter-space second-derivative coefficients, producing `a` O(100) instead of O(1e-3), and `alpha/gamma` forward-sweep values near MPAS tridiagonal ratios instead of WRF's Thomas recurrence.

There is also fixture metadata drift: the savepoint metadata says `top_lid=True`, but the source run namelist has `TOP_LID= 11*F` and the saved top row matches WRF line 620 with `lid_flag=1`. The fixed comparator therefore uses `top_lid=False` for this run.

## WRF Line Table

| WRF line(s) | Variable | Computation | Units | Stagger | Dependency | JAX equivalent | Status |
|---|---|---|---|---|---|---|---|
| 570-579 | signature | `a, alpha, gamma, mut, c1/c2, cqw, rdn/rdnw, c2a, dts, g, epssm, top_lid, dims` | mixed | mass and w | caller | `calc_coef_w_wrf_coefficients` at `acoustic_wrf.py:598-663` | Match after fix |
| 587-591 | `c2a`, `cqw`, `a`, `alpha`, `gamma` | 3D arrays `(i,k,j)` | native / dimensionless | mass for `c2a`, w for `cqw` and outputs | caller | defaults `c2a=1`, `cqw=1` at `acoustic_wrf.py:620-621` for M6B0-R fixture | Match fixture |
| 592-599 | `mut`, `rdn`, `rdnw`, `c1/c2` | hybrid-coordinate inputs | Pa, eta^-1, Pa terms | mass/vertical vectors | WRF grid/state | `mut` savepoint plus `load_wrfinput_metrics` in comparator, consumed at `acoustic_wrf.py:623-626` | Match after fix |
| 613-618 | loop bounds | interior `i/j`, `k_end=kte-1` | index | tile | domain dims | array whole-slice operation | Match |
| 619-620 | `lid_flag` | `1`, then `0` only if `top_lid` | dimensionless | scalar | namelist | `top_lid=False` for Canary d02 at comparator lines 46-55 | Match run |
| 621-623 | loops | outer `j`, top `k=kde-1` | index | w top | bounds | vectorized over `(ny,nx)` | Match |
| 624 | `cof` | `(.5*dts*g*(1.+epssm))**2` | m^2 s^-2 times timestep^2 | scalar | `dts,g,epssm` | `acoustic_wrf.py:628` | Match after fix |
| 625 | `a(i,2,j)` | lower boundary lower-diagonal seed `0` | dimensionless | w face 2 | none | `acoustic_wrf.py:636` | Match after fix |
| 626 | `a(i,kde,j)` | `-2*cof*rdnw(kde-1)^2*c2a*lid_flag / ((c1h*MUT+c2h)*(c1f*MUT+c2f))` | dimensionless | top w face | `cof`, hybrid mass denominators | `acoustic_wrf.py:635-637` | Match after fix |
| 627 | `gamma(i,1,j)` | `0` | dimensionless | bottom w face | none | `acoustic_wrf.py:638` | Match after fix |
| 629-634 | `a(i,kk,j)` | interior lower diagonal with `cqw*cof*rdn(kk)*rdnw(kk-1)*c2a(kk-1)` over hybrid denominator | dimensionless | w face `kk` | prior `cof`, metrics | `acoustic_wrf.py:640-644` | Match after fix |
| 635-639 | `b`, `c` | WRF diagonal and upper diagonal using two asymmetric hybrid denominators | dimensionless | current w face | `a`, `cof`, `cqw`, `rdn/rdnw`, `c2a`, `MUT` | `acoustic_wrf.py:646-654` | Match after fix |
| 640-641 | `alpha`, `gamma` | Thomas forward recurrence `alpha=1/(b-a*gamma(k-1))`, `gamma=c*alpha` | dimensionless | current w face | previous `gamma` | `acoustic_wrf.py:655-657` | Match after fix |
| 644-649 | top `alpha/gamma` | top diagonal `b=1+2*cof*rdnw^2*c2a/denom`, `c=0`, forward close | dimensionless | top w face | top `a`, previous `gamma` | `acoustic_wrf.py:659-662` | Match after fix |
| 651-652 | end loops/subroutine | no computation | n/a | n/a | n/a | function return at `acoustic_wrf.py:663` | Match |

## Specific Discrepancies

- Equation form: JAX used `I - lambda*d2/dz2` from `dz_m` and sound speed (`vertical_implicit_solver.py:37-82`), while WRF uses eta-pressure coefficients with `(c1h*MUT+c2h)*(c1f*MUT+c2f)` denominators (`module_small_step_em.F:626,632,637-639,646`).
- Unit interpretation: old JAX `rdzw=1/dz_m` was m^-1 (`vertical_implicit_solver.py:41`); WRF `rdn/rdnw` are eta-coordinate inverse layer intervals and are scaled by pressure-mass denominators (`module_small_step_em.F:632,637-639`).
- Staggering/order: old JAX populated all interior rows symmetrically from geometric lower/upper layers (`vertical_implicit_solver.py:58-81`), including a nonzero row at JAX index 1. WRF explicitly seeds `a(i,2,j)=0` before interior `kk=3..kde-1` (`module_small_step_em.F:625,629-633`).
- Coefficient assembly: old comparator converted raw tridiagonal `b/c` to `alpha/gamma` with `1/b` and `c/b`; WRF performs a forward recurrence using prior `gamma` (`module_small_step_em.F:640-641`). The fixed helper computes `alpha/gamma` in WRF order.

## Proof Summary

Before: `PARITY-DEFECT-LOCALIZED`, worst deltas `a=278.0947834046479`, `alpha=0.9950100403383089`, `gamma=0.4798648364335092`.

After: `PASS`, all tiers and fields have max absolute delta `0.0`, which is within `1e-6` and the stricter ladder `1e-11`.
