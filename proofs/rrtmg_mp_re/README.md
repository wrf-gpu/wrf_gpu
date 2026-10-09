# RE01 — pristine RRTMG true-caller oracle (GPUWRF_RRTMG_MP_RE, v0.3.3)

Request: MAILBOX 2026-10-05T17:31:57Z (LL01 finding 3). The frozen tier-1 RRTMG fixtures came from a synthetic harness
(`scripts/wrf_rrtmg_harness.f90`: has_req=1 with constant 10/30/75 µm, cldovrlp=1, ghg_input=0, o3input=0). PROD/WN3
call pristine RRTMG with different flags. This oracle runs the unchanged pristine V4.7.1 routines (`libwrflib.a`) with
the true caller wiring on 186 real CPU-WRF 0227 columns.

## Caller audit (pristine V4.7.1)

| item | WRF source | PROD/WN3 value | port today |
|---|---|---|---|
| has_reqc/i/s | module_physics_init.F:984–1024 (use_mp_re=1 default, mp8 + ra 4) | 1/1/1 | radii fixed 10/30/75 µm |
| radii | mp_gt_driver module_mp_thompson.F:1466–1477 → calc_effectRad :5594–5699, clamps 50/125/999 µm; Nc = Nt_c (is_aerosol_aware=F) | Thompson diagnosed | constant |
| LW/SW consumption | module_ra_rrtmg_lw.F:12179–12245 (sw :10776–10855): inflg 3/4/5, recloud=max(2.5,re·1e6) (≤2.5 & cloudy → 10.5 ocean / 7.5 land), reice=max(5,·) (retab(T) fallback), resnow=max(10,·) | — | — |
| has_req=0 (use_mp_re=0) | same block | 5/10/10 µm, inflg 2 / iceflg 3 | (not 10/30/75) |
| cloud overlap | Registry.EM_COMMON:2502 cldovrlp default 2; CPU history attribute CLDOVRLP=2 | maximum-random | random only (kernels/rad_mcica.py, rrtmg_lw.py:2421) |
| radiation P/P8W | first_rk_step_part1.F:280 P=p_hyd, P8W=p_hyd_w (phy_prep :4943–4970), PI=pi_phy, T=t_phy, T8W=t8w, RHO=grid%rho | hydrostatic | (port uses hydrostatic profiles) |
| icloud_bl merge | radiation_driver: CLDFRA=CLDFRA_BL, qc += QC_BL (qc<1e-6 & CLDFRA_BL>0.001), qi += QI_BL (qi<1e-8) | icloud_bl=1 | physics_couplers.py ~1250 |
| ozone | o3input=2: oznini/ozn_time_int/ozn_p_int on id 1 only (radiation_driver :1803); nests get o3rad by Registry rdf=(p2c) (level copy of the parent cell) | d01 profile on every nest | each domain its own lat/p_hyd (possible separate seam, unmeasured) |
| solar | radconst(julian) solcon/declin; calc_coszen(julian, xtime+radt/2, gmt) | — | — |
| gases | ghg_input=1 CLWRF (CAMtr_volume_mixing_ratio), yr 2026 | — | wrf_clwrf_ghg |

## Operands (extract.py) and disclosed approximations

State = CPU-WRF 0227 history at τ (the state the radiation call at the step starting at τ sees). Real-structured,
not a replay: (1) CLDFRA = history CLDFRA = CLDFRA_BL of the τ−30 min call (CLDFRA_BL at τ is not in CPU history);
(2) the QC_BL merge uses GPU W9 QC_BL/CLDFRA_BL at τ ≤ 24 (101 columns; port QC_BL is specific humidity, WRF mixing
ratio — b-phys seam ≈ 1 %); QI_BL is not in history → qi_rad = qi; (3) calc_effectRad gets P = p_phy from history
(WRF: the pre-MP pressure); (4) O3 from the d01 parent cell's p_hyd at the same frame. Checks: p_hyd_w[0] vs history
PSFC median 0.09 Pa (max 1.5); calc_coszen at xtime−15 min vs history COSZEN 8.7e-5 (the history holds the τ−30 call).

## Arms (re01_driver.F90) and results (parse.py → re01_summary.json, frozen tier-1 bounds flux 1 W/m² + 5 %, heating 1e-4 + 5 %)

A = WRF truth (Thompson radii, has_req=1, cldovrlp=2) · B = has_req=0 · C = constant 10/30/75, cldovrlp=2 ·
D = C + cldovrlp=1 (= the port today) · E = A + cldovrlp=1. Clear columns (27) are bitwise identical in every arm.

| vs A | columns outside | GLW max / mean | OLR max | SWDOWN max / mean | heating LW / SW K/d |
|---|---|---|---|---|---|
| B has_req=0 | 83 / 186 | 9.9 / +0.42 | 23.6 | 115 / +0.85 | 8.8 / 3.9 |
| C radii only | 76 | 12.3 / +0.68 | 18.3 | 89 / +2.0 (liq_thin_ocean +17.3) | 7.0 / 3.8 |
| E overlap only | 41 | 43.9 / +3.0 (liq_thin_ocean +4.6) | 12.5 | 282 / −7.4 | 26.2 / 10.6 |
| D port today | 93 | 43.7 / +3.6 | 19.5 | 233 / −4.3 | 25.9 / 9.7 |

Table counts use parse.py's set (scalar fluxes + heating). With the reader's full gate set (scalars + interfaces
0..nz + heating): B 101, C 100, E 48, D 117 columns outside vs A; the port today 119 vs A and 0 vs D
(tests/v025/fid_q2/test_rrtmg_mp_re_port_today.py). Interface convention: WRF stores layer fluxes for k = kts..kte+2,
so index nz+1 is the FIRST of RRTMG's 13 buffer interfaces above p_top (deltap 4 hPa, module_ra_rrtmg_lw.F:11565,
:12815) while the port's last index is the TOA → gates compare interfaces 0..nz; TOA via the *t scalars.

Port probe (port_probe.py, legacy CPU column solvers on the same operands): port == D on all 186 columns
(GLW .88, OLR .94, SWDOWN 1.74, SWUPT 1.23 W/m², heating .81 K/d max; 0 outside). Port heating_rate is dT/dt;
WRF RTHRATEN = dT/dt / pi_phy.

CPU-WRF history evidence (validate_hist.py, τ−30 call, cloudy columns; statistical, 30-min lag noise): SW
transmissivity rms A .070 / B .093 / C .087 / D .093 / E .085, TOA albedo A .058 / .077 / .072 / .074 / .066,
LWDNB rms A 9.7 / random-overlap arms 10.7–10.9 → CPU-WRF ran has_req=1 + maximum-random.

## Files

`census.py` (186 picks: d02/d03 × τ12–36 × liq_thin_ocean/liq_thick/liq_land/ice/snow/mixed/sgs_only/clear),
`extract.py` (operands → re01_input.bin sha c6610bee…), `re01_driver.F90` + `build.sh` (TH08 libwrflib recipe; output
re01_output.bin sha 327ea5c0…), `parse.py` (→ fixture, summary), `validate_hist.py`, `port_probe.py`.
Fixture: `tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz` (sha256 bba1a3e0…), reader
`tests/v025/fid_q2/rrtmg_mp_re_fixture.py`, tests `tests/v025/fid_q2/test_rrtmg_mp_re_{oracle,port_today}.py`.
Raw binaries + build: `<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE01/`.
