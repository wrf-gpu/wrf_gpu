# F7H — the geopotential/pressure restoring coupling: what was broken, the WRF
# truth, and the before/after trace

## TL;DR (the reframe was right that 9.4× is a red herring; the leading
## ph-carry hypothesis was the WRONG mechanism)

`ph_perturbation` looking "frozen at 131.83" is a **max-statistic artefact**, not a
broken carry: the initial hydrostatic bubble perturbation (131.83) stays the
largest value while the live work geopotential `ph_work` *does* change every
substep (4e-4 … 3.7e-3) and *is* carried (`proofs/f7h/ph_carry_trace.json`,
independently confirmed by the parallel GPT bug-hunt §1).  The real driver of the
linear-in-t `w` runaway is a **pressure-restoring inconsistency**: the vertical
buoyancy/PGF source (`pg_buoy_w`) and the carried state perturbation pressure were
both built from the small-step WORK-DELTA pressure (~O(1–10 Pa)) instead of the
WRF `calc_p_rho_phi` diagnostic `grid%p` (~O(1e3–1e4 Pa) once `ph'` evolves), so
the restoring gradient was suppressed and the spurious `−c1f·mu'` column-weight
term dominated, growing without bound as `mu'` ramped from 0.

## 1. Phase-1 empirical traces (cuda:0, fp64, taskset -c 0-3)

### 1a. `proofs/f7h/rwtend_source.json` — the once-per-stage `pg_buoy_w` forcing GROWS
```
step  RK  mu'(Pa)   pgf=g·rdn·Δp'   mu=g·c1f·mu'   rw_tend(max)
1     RK2 1.12e-2   0.591           0.110          10.19   <- already O(10)=~g
2     RK3 1.20e-1   1.024           1.175          43.13
3     RK3 3.23e-1   1.709           3.164          77.00
4     RK3 6.21e-1   2.381           6.095          111.2
5     RK3 1.01e+0   3.040           9.950          145.7
6     RK3 1.50e+0   3.688          14.71           180.6
```
`rw_tend` is dominated by the `−c1f·mu'` term, and `mu'` ramps 0 → 1.5 Pa (it
should stay ~0 for a fixed-mass bubble). `pgf` (the `rdn·Δp'` PGF) does **not**
cancel the `c1f·mu'` weight because the work-delta `p` carries none of the
`ph'`/`mu'` hydrostatic structure.

### 1b. `proofs/f7h/full_p_compare.json` — full grid%p balances the interior much better
For the SAME entry state, interior net `|pgf − c1f·mu'|`:
```
step6 RK3:  work_p net = 18.40    full_p net = 5.98   (~3× smaller with full grid%p)
```

### 1c. `f7h_buoyancy_sign.py` — at the IC the bubble has NO buoyancy on either path
At the rest IC (`mu'=0`): `max|full_p| = 2.9e-11 Pa`, `max|rw_tend| = 1.5e-8 m/s²`
vs the parcel estimate `g·θ'/θ0 ≈ 0.065 m/s²`. The IC is constructed
hydrostatically balanced so `alt = alt_full` ⇒ `p' ≡ 0`; buoyancy only emerges as
the dynamics evolve `ph'`, `t_2`, and `grid%p`.

### 1d. `f7h_wfield_structure.py` — pre-fix w was a DOWNWARD-at-center mode
Pre-fix, `w@bubble-center` was negative (−5.3 m/s) — the spurious `−c1f·mu'` weight
overwhelmed the (zero) buoyancy. Post-fix the center develops a coherent updraft
but a vertical 2Δz mode appears (see §4).

## 2. WRF ground truth (source audit; binary build deferred — see wrf_vs_jax_warmbubble.json)

- `module_em.F:1362` — `CALL pg_buoy_w(rw_tend, p, cqw, mu, mub, …)`: the `p`
  actual argument is **`grid%p` = the FULL-perturbation `calc_p_rho_phi`
  diagnostic**, and `mu` = `grid%mu_2` (full perturbation dry mass).
- `module_big_step_utilities_em.F:1029` — `al = −1/(c1·muts+c2)·(alb·c1·mu +
  rdnw·(ph(k+1)−ph(k)))` built from the **FULL `ph'`, `mu'`**.
- `:1083-1087` — `p = p0·(Rd·(t0+θ')/(p0·(al+alb)))^cpovcv − pb` (full θ').
- `module_em.F:49-227` (`rk_step_prep`) does **NOT** recompute `grid%p`; it only
  re-derives `al`/`alt` via `calc_alt`. So `pg_buoy_w` consumes the `grid%p` that
  the PREVIOUS RK step's closing `calc_p_rho_phi` left
  (`solve_em.F:6180`, `:7542`).
- `module_big_step_utilities_em.F:2553-2573` — `pg_buoy_w` interior face
  `rw_tend = (1/msfty)·g·(rdn·(p(k)−p(k−1)) − c1f·mu')` (dry cq1=1,cq2=0); top face
  `2·rdnw·(−p)` is irrelevant under the idealized rigid lid (advance_w sets
  `w_top=0`).

So WRF's restoring force lives in `rdn·Δ(grid%p)` with the **full** diagnostic
`grid%p`; that term grows to O(kPa) and balances the column weight, bounding `w`.

## 3. The fix (operational_mode.py; both WRF-faithful, regression-safe)

### Fix A — `pg_buoy_w` stage forcing from the FULL grid%p
`_acoustic_core_state_from_prep`: build `rw_tend_stage` from the
full-perturbation `grid%p` via `diagnose_pressure_al_alt` (the F7F-fixed JAX
`calc_p_rho_phi`) instead of `calc_p_rho_wrf(prep).p` (the work-delta). Matches
`module_em.F:1362`.

### Fix B — refresh `grid%p` at the RK-stage boundary (NEW `_refresh_grid_p_from_finished`)
After `small_step_finish_wrf`, recompute `state.p_perturbation`/`p_total` from the
**finished physical `ph'` and θ** via `calc_p_rho_phi`, exactly as WRF closes each
RK step (`solve_em.F:6180`, `:7542`). Previously the operational carry kept
`state.p_perturbation = calc_p_rho_step` work pressure (O(1–10 Pa)); the WRF
diagnostic is O(1e3–1e4 Pa) once `ph'` moves (GPT bug-hunt §1 divergence table:
work=4.5 Pa vs calc_p_rho_phi=554 Pa at 10 s). The stale O(1) pressure starved the
next stage's vertical + horizontal PGF of the restoring gradient.

### Reverted — theta-coupled-work carry (acoustic.py)
GPT §4 flagged a real ~6.5e-2/substep theta re-coupling drift, and feeding the
carried `theta_coupled_work` directly to `advance_mu_t` *did* activate theta
transport (bubble rose faster). But it **broke the bare-core `test_m4_acoustic`
path** and made the bubble detonate EARLIER (~90 s vs ~180 s). Per the AC5
"nothing weakened / no regression" rule it was reverted; it remains a documented
secondary lead for a follow-up sprint with a proper bare-core/operational split.

## 4. Before / after (warm bubble, fp64, cuda:0, `max|w|` at t)

| t (s) | pre-F7H (HEAD) | +Fix A (full-p pg_buoy) | +Fix A+B (this sprint) |
|------:|---------------:|------------------------:|-----------------------:|
| 30    | 2.53           | 1.07                    | 1.19  (centroid +59 m) |
| 60    | 13.6           | 6.00                    | 10.2  (centroid +282 m, rising) |
| 100   | 44.7 (→NaN@190)| 21.0                    | 4.3 → … detonates ~180 s |
| —     | linear-in-t, bubble does NOT rise | halved, still no rise | bubble RISES; restoring p' now O(100–400 Pa); residual 2Δz vertical mode |

Net: the WRF-faithful pressure-restoring fixes cut `max|w|@100s` by ~10×, made the
thermal physically RISE (centroid 2000 → 2173 m by 160 s), and lifted the
diagnostic restoring pressure from O(1) to O(100–400) Pa. **But the cases still go
non-finite** (`skamarock_bubble_verdict.md`, `straka_density_current_verdict.md`):
a residual **vertical 2Δz acoustic mode at the bubble-center column**
(`w@center(k)` alternates −9.88 / +8.59 between adjacent levels at t=60 s) grows and
detonates ~180 s.

## 5. Remaining gap (→ F7H_PARTIAL; STOP per "no 8th workaround")

The residual is a vertical 2Δz mode in the implicit-`w`/`ph` acoustic solve — a
coefficient-level question (`c2a`, `epssm` off-centering, `calc_coef_w`
tridiagonal) that the **WRF em_quarter_ss savepoint comparison is the definitive
arbiter for** (deferred; seeds M9). No clamp/cap/epssm-tune was applied (hard rule
3). The two committed fixes are correct, evidence-backed, and regression-safe; the
remaining mode needs the WRF binary to localize at the operator/coefficient level.
