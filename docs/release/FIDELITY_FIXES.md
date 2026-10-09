# WRF-fidelity fixes found during v0.25 → v0.3 (preparation record; status to verify on the final tree)

Method that found them: unmodified WRF v4 Fortran (pristine build) run as an **oracle** on the same column/operator
inputs, plus full GPU-vs-CPU-WRF forecasts scored against frozen tolerances. A fix counts only with a gate test that
fails when the fix is removed. IDs refer to `.agent/kernel/FINDINGS.md`.
Status legend: **main** = merged on the default path · **fast path** = fixed in the fp32 kernels that are the v0.3
default (RC defaults) · **open** = known, not fixed in v0.3. `[verify]` until the final-tree audit at release.

| id | what differed from WRF | effect | status |
|---|---|---|---|
| B17 | theta_m coupling, Thompson sedimentation band/density, Exner rcp, KF phy_prep rho, option-0 theta | P0 core physics/dynamics fixes | main |
| B19 | radiation called at the wrong steps (1, 34, 67 in WRF), no first-step heating | radiation phase | main (WRF cadence default) [verify] |
| B13 | acoustic UV ring updates and advance_w surface convention not WRF's | boundary acoustic | fast path [verify] |
| B21 | MYNN CLOUDMIX=1 missing (no cloud-water mixing / latent heating in PBL) | low cloud, rain pattern | fast path [verify] |
| B22 | RRTMG clear-sky flux: per-layer instead of whole-column cloud flag (50 W/m²) | clear-sky radiation | fast path [verify] |
| B24 | KF shallow-convection TIMEC | cumulus timing | fast path (KF kernel) [verify] |
| B25 | MYNN PBL height threshold 1.0 K over land (WRF 1.25 K) | PBLH −31 m over land | fast path [verify] |
| B26 | W advection had a bottom-face term WRF does not have | W near terrain | fast path [verify] |
| B29 | acoustic finish used saved instead of live dry mass | mass coupling | fast path [verify] |
| B32 | microphysics before instead of after RK3 transport (WRF order) → negative number concentrations, 400 M guard repairs | rain/Nr on WN3 | main (WRF-order default, D21) |
| B34 | base-state PHB/PB drift through separate rounding | static fields | main |
| B35 | microphysics increment applied twice with sst_update | runaway rain on WN3 d03 | main (e5d66b834) |
| B36 | nest boundaries forced only QV, not the other moist/number species | nest edges, all nested runs | main |
| B37 | root-domain Ni/Nr not advected | number concentrations d01 | main |
| B38 | missing WRF h_diabatic pair in the acoustic steps | intermediate heating response | main |
| B39 | MYNN surface layer got default inputs (MOL, fluxes, QSFC, PBLH, dx) | surface fluxes → T2/U10 | main |
| B39b | surface constants (cp 1004 vs 1004.5, ep1, ep2) not WRF's | surface layer, Noah-MP coupling | main |
| B40 | microphysics also ran in the specified boundary zone | edge-only | main |
| B42 | GWDO pressure/height inputs not WRF's (up to 12 % of the drag increment) | gravity-wave drag | main |
| B44 | specified root used a periodic sixth-order filter in the relax zone | d01 relax-zone U/V/θ | main |
| B45 | root lateral BC relaxed all moist species (WRF: only QV + flow-dependent others) | d01 boundaries | main |
| B46 | GWDO on the post-PBL state and as an Euler increment | gravity-wave drag | main |
| B47 | land Q2 not from Noah-MP | Q2 over land | main |
| B49 | water T2/Q2 bulk form with Noah-MP | T2/Q2 over water | main (in-step diagnostics) [verify] |
| B51 | pg_buoy_w fed rounded mu′ instead of WRF's mu_2 | vertical momentum | main (fab6b3c92) |
| B52 | horizontal diffusion missing for moist and number scalars when diff_opt=1 | dry near-surface air over high terrain; Q2/RH2 | main (a8a95424f + 51056861a); final-tree validation pending |
| B53 | MYNN qsq/qc_bl/qi_bl/cldfra_bl discarded at the RK seam | missing subgrid clouds in radiation, GLW/SWDOWN/T2 | main (a95f2d757); final-tree validation pending |
| B54/B57 | omitted land state, flux accumulations, held radiation diagnostics and WRF metadata in history | missing or constant output variables | merged in FINAL; PROD all-frame audit has no missing variables or CPU globals, with [five trace exceptions](INTEGRITY_EXCEPTIONS.md); WN3 checks pending |
| B55/B56 | Thompson process rates used post-warm state; mp8 graupel constants and sedimentation preparation differed from WRF | cold collection and graupel fall speeds | native corrections merged; remaining heat/cloud and entry-state residuals disclosed [verify] |
| glacier init | NOAHMP_INIT ice-soil moisture, temperature cap, snow floor and XICE predicate were incomplete | Swiss initial soil/snow fields | main (`d65839f74`); real Swiss initial fields and clause-deletion controls verified |
| snow45 | missing snow-layer TSNO write-back, FICEOLD and WRF water/energy coupling order | snow temperature, phase change and water balance | merged in FINAL; pristine Swiss snow-column multi-step oracle and deletion controls reviewed; full Swiss forecast validation pending |
| glacier runtime | WRF's separate NOAHMP_GLACIER evolution is unported | glacier cells use standard Noah-MP SFLX | open, planned v0.3.1; no glacier-runtime fidelity claim |

The first-night land T2 bias improves in the six-hour checks after B52/B53: for the 0227/0608 cases at 06Z,
mean GPU minus CPU T2 over land changes from −0.70/−0.64 K to −0.04/+0.00 K on d01, and from −0.42/−0.56 K
to −0.02/−0.02 K on d02 (validation-lane report, 2026-10-03 21:35Z).

The 0614 24 h follow-up also checks the second night against original CPU-WRF: d02 land-mean T2 at +24 h
improves from −0.96 K to **−0.11 K [M]**, and d03 from −0.08 K to −0.04 K. On the islands the d02 residual is
−0.04 K; the remaining colder spread is mainly over African barren land. This is the G05 component-validation
run on main `b9f43aa26`, with the q2/cloud fixes, not the final release tree (fid-q2 report, 2026-10-04 01:08Z;
receipt `fid-q2/G05/20260614_18z_a1/receipt.json`, reduction `fid-q2/N01/main_0614_24h.txt`).
The final-tree 24/72 h D6, strict integrity and formal twin verdict remain required.
