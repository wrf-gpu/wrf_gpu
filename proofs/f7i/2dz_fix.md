# F7I — root-cause analysis of the warm-bubble vertical blow-up

**Status: F7I_PARTIAL.** The "residual 2Δz acoustic mode in the implicit w/ph
solve" framing is **REFUTED**. The growing center-column vertical structure is a
deep (≈12-level wavelength) **buoyancy-driven vertical standing mode** that grows
exponentially (e-folding ≈ 29 s, fixed modal structure, amplitude-growing) and
detonates ≈ 180 s. It is driven by the once-per-RK-stage `pg_buoy_w` large-step
forcing, **not** by the implicit acoustic solve. No WRF-grounded single fix tested
(epssm off-centering, w-advection in `rw_tend`, WRF `km_opt=2` diffusion) removes
it. The WRF `em_quarter_ss` binary — the definitive arbiter — could not be built
in bounded effort on the NVHPC-GPU Gen2 fork (see `wrf_em_quarter_ss_savepoints.json`
build-deferral note). Deltas below localize the residual for the next sprint.

## What the mode actually is (not a 2Δz checkerboard)

`proofs/f7i/center_column_w_trace.json` — center column (ic=40, x=10.1 km),
warm bubble, fp64, cuda:0:

| t (s) | max\|w\|center | zz_energy | sign_alt (adjacent-flip fraction) |
|------:|---------------:|----------:|----------------------------------:|
| 10  | 0.025 | 1.7e-4 | 0.000 |
| 30  | 0.182 | 5.6e-3 | 0.050 |
| 60  | 0.985 | 0.213  | 0.075 |
| 100 | 4.386 | 4.07   | 0.075 |
| 150 | 17.27 | 48.6   | 0.075 |
| 170 | 26.55 | 124.6  | 0.075 |
| 180 | NaN   | —      | —     |

`sign_alt` is **constant at 0.075** (only ~3 of 40 faces flip sign) — the
structure is FIXED while the amplitude grows exponentially. The center-column
profile at t=100 s is a smooth multi-lobe wave (`+0.74,+1.85,+1.94,+1.35,+0.07,
-1.61,-3.22,-4.26,-4.39,-3.54,-1.95,-0.04,+1.73,+3.02,+3.55,+2.42,...`),
**not** an adjacent-level `(-1)^k` checkerboard. Growth: `max|w| ~ exp(t/29s)`,
per-acoustic-substep (dts=0.01 s) factor ≈ 1.00034.

## Proof it is NOT the implicit w/ph acoustic solve

1. **epssm-independent** (`proofs/f7i/epssm_sensitivity.json`): going from
   epssm=0.1 → 1.0 (maximal/backward-Euler off-centering) barely changes the
   growth (t=100 s max\|w\| 4.386 → 4.208; zz_energy 4.07 → 2.72). A genuine
   acoustic mode in the off-centered implicit solve would be crushed by
   epssm→1. It is not. (Independently reproduced by the parallel GPT audit:
   4.386/4.305/4.208.)
2. **Injected-mode probe (GPT audit, independent)**: feeding a pure interior
   `w(k)=(-1)^k` mode into `advance_w_wrf` with zero forcing yields an amplitude
   ratio of **0.996–0.993 per substep (DAMPING)** across eps=0…1. The implicit
   operator + `calc_coef_w` correctly damp the 2Δz mode.
3. **Source line-by-line audit (GPT, corroborated here)**: JAX `calc_coef_w`
   (`acoustic_wrf.py:648-686`) and `advance_w_wrf` off-centering
   (`advance_w.py:230-432`) match WRF `module_small_step_em.F:624-649` and
   `:1341-1584` term-for-term, including `(1±epssm)/2` weights, signed
   `rdn/rdnw`, and the `ph` work-delta convention (WRF `:275-276`
   `ph_2 = ph_1 - ph_2` ⇒ JAX `prep.ph_work ≈ 0` at stage entry — FAITHFUL).

## Proof the driver is the once-per-stage `pg_buoy_w` `rw_tend`

`proofs/f7i/term_ablation.json` — operational warm bubble to t=100 s, one
large-step term zeroed at a time (diagnostic monkeypatch, no production change):

| ablation | max\|w\| | zz_energy |
|----------|---------:|----------:|
| baseline | 4.386 | 4.07 |
| zero `rw_tend_pg_buoy` | **0.0013** | 7.8e-8 |
| zero `theta_tend` (advection) | 0.323 | 0.018 |

Zeroing the stage `pg_buoy_w` forcing removes the entire mode (and the bubble
rise). The mode IS the warm bubble's vertical response — but it grows
exponentially instead of saturating.

`proofs/f7i/rwtend_profile_*.json` — the smoking gun: at the IC (t=0) the bubble
is hydrostatically balanced (`grid_p_full ≈ 0`, `rw_tend ≈ 0`); by t=3 s a
smooth, uniformly-downward `rw_tend ≈ -115 m/s²` has developed against a smooth
warm `θ′ = +2 K` blob; by t=30 s `rw_tend` has acquired the multi-lobe vertical
oscillation that matches the developing `w`. `grid_p_full` and `θ′` stay smooth
throughout — the oscillation is in the **forcing↔response feedback**, not in the
input fields.

## WRF-faithfulness gaps found (real, but none is the fix)

1. **`advect_w` is missing from `rw_tend`.** WRF `rk_tendency` builds
   `rw_tend = advect_w(w) + pg_buoy_w(grid%p)` (`module_em.F:1011-1059` then
   `:1362`). JAX `_acoustic_core_state_from_prep` sets
   `rw_tend_pg_buoy = pg_buoy_w_dry(...)` only — the large-step vertical
   advection of `w` is computed (`compute_advection_tendencies`,
   `advection.py:270`) but never folded into `rw_tend`.
   **Tested** (`proofs/f7i/wadv_fix_probe.json`): adding the coupled
   `tendencies.w` into `rw_tend` changes t=100 s negligibly (4.386 → 4.364) and
   still detonates before t=500 s. Faithfulness fix, not the stability fix.
2. **`rhs_ph` (large-step geopotential horizontal advection) is never
   computed.** `ph_tend` is initialised to zero (`operational_state.py:94`) and
   never updated; `accumulate_ph_tend` (`small_step_scratch.py:66`) is a stub.
   WRF computes it via `rhs_ph` (`module_em.F:1254`,
   `module_big_step_utilities_em.F:1365+`). NOT YET TESTED as a fix (a
   substantial operator; deferred to avoid speculative un-WRF-verified code).
3. **Diffusion / top-damping mismatch vs WRF `em_quarter_ss`.** JAX warm bubble
   uses `const_nu=0` + `damp_opt=3` (small-step top damper). WRF `em_quarter_ss`
   uses `diff_opt=2, km_opt=2, khdif=kvdif=500` + `damp_opt=2` (large-step
   Rayleigh `rk_rayleigh_damp`). **Tested**: enabling constant-K diffusion at the
   WRF value (nu=500, and 75/200) makes the run finite at 180 s but **still
   detonates before 500 s**. Diffusion masks/delays the growth; it does not
   remove the underlying energy source.

## Conclusion / next decision

The implicit w/ph solve and `calc_coef_w` are WRF-faithful and 2Δz-stable
(independently proven two ways). The exponential mode is a vertical
momentum/buoyancy **balance discretization** problem in the large-step path:
the `pg_buoy_w` vertical PGF and the in-solver `c2a·alt·t_2ave` buoyancy do not
discretely telescope into a saturating response. The WRF `em_quarter_ss` binary
center-column savepoint comparison (`a/alpha/gamma`, `w`, `ph`, `p`, `t_2ave`,
`muave` after IC + steps 1-3) is the definitive arbiter and is the gating
deliverable for the next sprint; the build harness here
(`/home/enric/src/wrf_ideal_f7i/WRF` + conda `wrfbuild` env) is ~75% landed and
documented for that follow-up.
