# Part 2 — M6B0-R Python `_wrf_calc_coef_w` vs WRF Fortran `:570-652`

Source under audit:
- Fortran: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:570-652`
- Python:  `/tmp/wrf_gpu2_reprod/scripts/m6b0r_wrf_savepoint_extract.py:122-163`
  function `_wrf_calc_coef_w`

Note (important framing): the M6B0-R "Python reproduction" lives in the **extractor**, not in the **comparator** (`scripts/m6b0r_jax_vs_wrf_compare.py`). The comparator simply diffs JAX output against HDF5 savepoint values written by the extractor; the extractor itself is what computes the "expected" `a/alpha/gamma` via the Python translation of `calc_coef_w`. The savepoints therefore reflect the extractor's Python reading of WRF — not Fortran-emitted truth.

## Index convention table

WRF: `kts=1, kte=nz_mass, kde=nz_mass+1`. In the Canairy d02 column savepoint:
`nz_mass=44, kde=45`.

| Array  | Fortran range | Python length | Mapping |
|--------|---------------|---------------|---------|
| `c1h, c2h, rdn, rdnw` | 1..nz_mass=44 | 44 | F(k) <-> P[k-1] |
| `c1f, c2f` | 1..kde=45 | 45 | F(k) <-> P[k-1] |
| `a, alpha, gamma` (W faces) | 1..kde=45 | 45 | F(k) <-> P[k-1] |
| `c2a, cqw` (mass) | 1..nz_mass | 44 | F(k) <-> P[k-1] |

## Line-by-line audit

Two driver loops + two boundary rows. Indices below: F=Fortran, P=Python.

### Initialization & constants

| WRF Fortran (line) | Python (line) | Status |
|--------------------|--------------|--------|
| `cof = (.5*dts*g*(1.+epssm))**2` (624) | `cof = (0.5 * dts * g * (1.0 + epssm)) ** 2` (137) | MATCH |
| `lid_flag=1; IF(top_lid) lid_flag=0` (619-620) | `lid_flag = 1.0` (138) | **MISMATCH** — Python hardcodes 1.0; ignores `state["attrs"]["top_lid"]`. In the actual run, `top_lid=True` (extractor default since wrfout lacks attr), so WRF would set lid_flag=0. This zeroes `a(kde)` in Fortran but Python computes a non-zero `a[nz]`. |
| `a(i,2,j) = 0.` (625) | `a[1,:,:] = 0.0` (140) | MATCH |
| `gamma(i,1,j) = 0.` (627) | `gamma[0,:,:] = 0.0` (144) | MATCH |

### Top boundary `a(i,kde,j)` (line 626) — k=kde-1=nz_mass

Fortran (with `k=kde-1`):
```
a(i,kde,j) = -2.*cof*rdnw(kde-1)**2 * c2a(i,kde-1,j) * lid_flag
             / ((c1h(k)*MUT+c2h(k))*(c1f(k)*MUT+c2f(k)))
```
where `c1h(k)=c1h(kde-1)=c1h(nz)` and `c1f(k)=c1f(kde-1)=c1f(nz)`.

Python (line 141-143):
```python
k_top = nz - 1                                   # P[43] = F(44) = F(kde-1)  OK
denom_top = (c1h[k_top]*mut+c2h[k_top]) * (c1f[nz]*mut+c2f[nz])
                                                  # c1f[nz]=c1f[44]=F(45)=F(kde)  WRONG
a[nz,:,:] = -2.0 * cof * rdnw[nz-1]**2 * c2a[nz-1] * lid_flag / denom_top
```
**MISMATCH** — Python's `denom_top` for `a[nz]` uses `c1f[nz]` (Fortran `c1f(kde)`) instead of `c1f[nz-1]` (Fortran `c1f(kde-1)`). Fortran wants both `c1h` and `c1f` factors at `k=kde-1`.

Combined with the `lid_flag` bug above, Python computes a wrong, non-zero `a[nz]`; correct Fortran would compute `a(kde)=0` (because top_lid=True drives lid_flag=0).

### Interior `a` loop (Fortran 629-633): `DO kk=3,kde-1, k=kk-1`

Fortran:
```
a(i,kk,j) = -cqw(i,kk,j)*cof*rdn(kk)*rdnw(kk-1)*c2a(i,kk-1,j)
            / ((c1h(k)*MUT+c2h(k))*(c1f(k)*MUT+c2f(k)))
```

Python (line 146-149):
```python
for kk in range(2, nz):           # kk_P in 2..nz-1=43  <-> kk_F in 3..nz=44   OK
    k = kk - 1                     # k_P in 1..42       <-> k_F in 2..43       OK
    denom = (c1h[k]*mut+c2h[k]) * (c1f[k]*mut+c2f[k])  # OK
    a[kk,:,:] = -cqw[kk]*cof*rdn[kk]*rdnw[kk-1]*c2a[kk-1] / denom  # OK
```
**MATCH** — index translation verified: F `rdn(kk_F)` <-> P `rdn[kk_P]`, F `rdnw(kk-1)` <-> P `rdnw[kk-1]`, F `c2a(...,kk-1,...)` <-> P `c2a[kk-1]`, F `c1h(k)` <-> P `c1h[k]`, F `c1f(k)` <-> P `c1f[k]`.

### Interior `b/c/alpha/gamma` recurrence (Fortran 635-643): `DO k=2,kde-1`

Fortran:
```
b = 1 + cqw(i,k,j)*cof*rdn(k)*( rdnw(k)*c2a(i,k,j)/( (c1h(k)*MUT+c2h(k))*(c1f(k)*MUT+c2f(k)) )
                              + rdnw(k-1)*c2a(i,k-1,j)/( (c1h(k-1)*MUT+c2h(k-1))*(c1f(k)*MUT+c2f(k)) ) )
c = -cqw(i,k,j)*cof*rdn(k)*rdnw(k)*c2a(i,k,j) / ( (c1h(k)*MUT+c2h(k))*(c1f(k+1)*MUT+c2f(k+1)) )
alpha(i,k,j) = 1/(b - a(i,k,j)*gamma(i,k-1,j))
gamma(i,k,j) = c*alpha(i,k,j)
```

Python (line 151-158):
```python
for k in range(1, nz):              # k_P in 1..43 <-> k_F in 2..44 = 2..kde-1  OK
    denom1 = (c1h[k]*mut+c2h[k])*(c1f[k]*mut+c2f[k])              # OK
    denom0 = (c1h[k-1]*mut+c2h[k-1])*(c1f[k]*mut+c2f[k])          # OK
    denomp = (c1h[k]*mut+c2h[k])*(c1f[k+1]*mut+c2f[k+1])          # OK
    b = 1.0 + cqw[k]*cof*rdn[k]*(rdnw[k]*c2a[k]/denom1
                                + rdnw[k-1]*c2a[k-1]/denom0)      # OK
    c = -cqw[k]*cof*rdn[k]*rdnw[k]*c2a[k]/denomp                  # OK
    alpha[k,:,:] = 1.0 / (b - a[k]*gamma[k-1])                    # OK
    gamma[k,:,:] = c*alpha[k]                                     # OK
```
**MATCH** — all index translations verified.

### Top boundary `b/alpha/gamma` (Fortran 644-650): `k=kde`

Fortran:
```
b = 1 + 2.*cof*rdnw(kde-1)**2*c2a(i,kde-1,j) / ( (c1h(k-1)*MUT+c2h(k-1))*(c1f(k)*MUT+c2f(k)) )
   ! k=kde: c1h(k-1)=c1h(kde-1)=c1h(nz), c1f(k)=c1f(kde)
c = 0
alpha(i,kde,j) = 1/(b - a(i,kde,j)*gamma(i,kde-1,j))
gamma(i,kde,j) = 0
```

Python (line 160-162):
```python
b_top = 1.0 + 2.0*cof*rdnw[nz-1]**2 * c2a[nz-1] / denom_top
        # denom_top = (c1h[nz-1]*mut+c2h[nz-1])*(c1f[nz]*mut+c2f[nz])
        # MATCHES Fortran for the b_top numerator/denominator (c1f[nz]=c1f[44]=F(45)=F(kde) OK)
alpha[nz,:,:] = 1.0/(b_top - a[nz]*gamma[nz-1])                  # MATCH
gamma[nz,:,:] = 0.0                                              # MATCH
```
**MATCH** for the `b_top`/`alpha[nz]`/`gamma[nz]` calculation in isolation. The bug is that the SAME `denom_top` was wrongly reused at line 143 for the top `a` row, where Fortran wants `c1f(kde-1)`.

## Summary of discrepancies

| # | Location | Severity | Magnitude impact (column tier) |
|---|----------|----------|---------------------------------|
| 1 | `lid_flag` hardcoded to 1.0 (line 138) — should be `0.0 if top_lid else 1.0` | bug | only `a[nz]` and `alpha[nz]` affected |
| 2 | Top `a` denom uses `c1f[nz]` (line 142) — should use `c1f[nz-1]` | bug | only `a[nz]`, `alpha[nz]` |
| 3 | `cqw` placeholder = ones (line 132) | DESIGN, not bug — same on both sides of compare | dwarfed by formulation difference |
| 4 | `c2a` placeholder = ones (line 133) | DESIGN, not bug — same on both sides | dwarfed |

Bugs 1+2 together produce: `a[44]` delta = 1.03e-2, `alpha[44]` delta = 4.95e-5 vs strictly-correct WRF reading.
These are TINY compared to the JAX-vs-anything deltas (`a` = 259.66, `alpha` = 0.995, `gamma` = 0.479)
and they live at index 44, while the JAX deltas peak at indices 2 (a, alpha) and 40 (gamma).

## Verdict on the M6B0-R Python reproduction

**PARTIAL** faithfulness to WRF Fortran `:570-652`:
- The arithmetic body (interior loops, recurrence, b_top) is a faithful translation.
- Two localized bugs in the top boundary row (`lid_flag` hardcode + `c1f[nz]` instead of `c1f[nz-1]`).
- Two pre-existing simplifications (`cqw=1`, `c2a=1`) that the worker report already
  flags as harness limitations — these are operational caveats, not bugs in the
  reproduction's translation of the `calc_coef_w` arithmetic.

The bugs do NOT explain the JAX-vs-WRF gap. They are interior to k=44 only; the
JAX deltas dominate at interior levels (k=2, 40).
