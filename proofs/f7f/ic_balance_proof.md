# F7F — WRF fixed-mass IC ph-rebalance derivation + the calc_p_rho_phi geopotential-term fix

## 1. WRF idealized-init hydrostatic rebalance (ground truth)

`WRF/dyn_em/module_initialize_ideal.F`, `quarter_ss` (`:1103-1130`) and `grav2d_x`
/ Straka (`:1278-1313`) both do, per column, with **mu' = 0** (dry, fixed mass):

```
! perturb theta, recompute full inverse density (EOS), al = alt - alb
grid%t_1   += delt * cos(.5*pi*RAD)**2                          ! :1112
grid%alt    = (r_d/p0)*(t_1+t0)*qvf*((p+pb)/p0)**cvpm           ! :1114-1116
grid%al     = grid%alt - grid%alb                              ! :1117
! rebalance hydrostatically (ph_1 = perturbation geopotential ph', ph_1(1)=0)
DO k = 2,kte
  grid%ph_1(k) = grid%ph_1(k-1)
      - dnw(k-1)*( (c1h*mub+c2h + c1h*mu')*al(k-1) + c1h*mu'*alb(k-1) )   ! :1124-1129
  grid%ph0(k)  = grid%ph_1(k) + grid%phb(k)
ENDDO
```

Base state (`:982`): `phb(k) = phb(k-1) - dnw(k-1)*(c1h*mub+c2h)*alb(k-1)`, `ph_1(1)=0`.

So WRF's intended buoyancy source is the **balanced perturbation fields** (`ph'`, full-θ
`alt`, the EOS `grid%p`), not "base ph + θ". `mu' = 0` is correct (Hypothesis A refuted).

## 2. IC code: explicit, WRF-faithful, bit-identical

`src/gpuwrf/ic_generators/idealized.py::_make_state` now integrates exactly the WRF
recurrence (pure sigma c1h=1, c2h=0, mub=mu, mu'=0):

```
alt_full = EOS(theta_full, p+pb);   alb = EOS(theta0, p+pb);   al = alt_full - alb
phb(k+1)    = phb(k)    + |dnw(k)|*mu*alb(k)          # base, WRF :982
ph_pert(k+1)= ph_pert(k)+ |dnw(k)|*mu*al(k)           # WRF :1124-1129 (mu'=0)
ph_total = phb + ph_pert ;  ph_perturbation = ph_pert ;  mu_perturbation = 0
```

(`|dnw|` upward-positive == WRF's `-dnw` with dnw<0.)  This is algebraically identical
to the previous `ph_col - ph_base_col` formulation (verified), but now matches WRF
line-for-line and the misleading "base ph + θ is the buoyancy source" comment is gone.
`mu_perturbation == 0` is asserted by `proofs/f7f/rwtend_after_fix.json`
(`max_abs_mu_prime = 0`).

## 3. The decisive operator bug found in F7F: calc_p_rho_phi dropped the geopotential term

`pg_buoy_w` consumes WRF `grid%p` = the `rk_step_prep`/`calc_p_rho_phi` **absolute
perturbation-pressure diagnostic** (`module_em.F:1361`).  WRF `calc_p_rho_phi`
(`module_big_step_utilities_em.F:1029, :1083-1087`), dry:

```
al(k) = -1/(c1*muts+c2) * ( alb(k)*c1*mu'  +  rdnw(k)*(ph'(k+1)-ph'(k)) )      # :1029
p(k)  = p0*( Rd*(t0+theta')/(p0*(al(k)+alb(k))) )**cpovcv - pb(k)             # :1083-1087
```

The JAX equivalent `acoustic_wrf.diagnose_pressure_al_alt` **dropped the
`rdnw*(ph'(k+1)-ph'(k))` term** and used base θ in the EOS:

```
# BEFORE (bug):
al = -(alb*c1h*mu') / (c1h*muts+c2h)            # geopotential term MISSING
p  = EOS(theta_base, al+alb) - pb               # base theta, not full theta
```

For the dry rebalanced bubble (`mu'=0`, `ph'!=0`) this gave `al == 0` and
`p == 0` **regardless of the rebalanced ph'** — i.e. the warm/cold thermal never
produced a perturbation pressure (a dead bubble on the real pressure path).  This was
the actual reason Sprint B reached for the synthetic absolute `p_buoy`.

### After (F7F fix)

```
al = -( alb*c1h*mu' + rdnw*(ph'(k+1)-ph'(k)) ) / (c1h*muts+c2h)
p  = p0*( Rd*(t0+theta')/(p0*(al+alb)) )**cpovcv - pb        # full theta
```

Verified (`/tmp` probe, fp64, cuda:0):
- Balanced warm-bubble IC: `max|grid%p| = 1.512e3 Pa`, `max|al| = 6.87e-3` (physical).
- Neutral base column: `max|grid%p| = 2.9e-11 Pa` (≈0, correctly balanced).
- Unbalanced (base-ph) IC: `max|grid%p| = 7.50e2 Pa` — reproduces the historic
  ~744 Pa artifact that Sprint B's synthetic pressure had baked in (AC2 discriminator).

## 4. Before / after summary (warm bubble, fp64, cuda:0)

| quantity                              | pre-F7F (synthetic p_buoy) | F7F (this sprint)            |
|---------------------------------------|----------------------------|-----------------------------|
| direct stage-const pg_buoy_w max\|rw\|| 0.6147 m/s² (9.40× analytic)| 0.0 m/s² (AC1 PASS)         |
| max\|p_buoy\| fed to pg_buoy_w        | 743.97 Pa (synthetic)      | live calc_p_rho work p (~0) |
| c1f·mu' term                          | 0 (mu'=0)                  | 0 (mu'=0)                   |
| warm-bubble max\|w\| growth           | linear ≈ 0.615·t → NaN@80s | no 0.615·t; NaN@~190s       |
| calc_p_rho_phi grid%p (balanced)      | n/a (path unused)          | 1.512e3 Pa (now nonzero)    |

## 5. Open item (why AC3/AC4 still FAIL) — see worker-report §unresolved-risks

Removing the synthetic p_buoy eliminates the 9.4× over-forcing (AC1 PASS) but leaves the
bubble weakly forced and a top-boundary gravity-wave mode that detonates (warm ~190s,
straka ~30s).  Feeding the now-correct `calc_p_rho_phi` `grid%p` into `pg_buoy_w`
over-forces the OTHER way (max\|w\| ≈ 1.2·t) because the in-solver `advance_w`
`c2a·alt·t_2ave − c1f·muave` buoyancy does not subtract the same hydrostatic reference
(`muave = 0` when `mu' = 0`), so `pg_buoy_w(grid%p)` double-counts the perturbation
column weight.  Reconciling the large-step `pg_buoy_w` pressure reference with the
small-step `advance_w` buoyancy reference is the binding open F7F item.
