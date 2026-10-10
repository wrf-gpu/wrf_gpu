# CAM01 — pristine WRF CAM radiation true-caller oracle (ra_lw_physics = ra_sw_physics = 3), lane o1-camrad

The old v0.18 `ra3_wrf_real.json` was an output-only history column (no operands) and is no longer in the tree, so it
cannot support column parity. CAM01 runs the **unchanged pristine V4.7.1 routines** (`libwrflib.a`:
`camradinit`, `camrad`, plus the module's own `radctl`/`radclwmx` call chain) with the radiation_driver wiring.

## Caller wiring reproduced (cam01_driver.F90)

| item | WRF source |
|---|---|
| init | `module_physics_init.F:2249-2269` camradinit(R_D,R_V,CP,G,STBOLT,EP_2, shalf=ZNU, pptop=p_top/1000 (z2sigma eta branch), levsiz=59, n_ozmixm=13, paerlev=29, n_aerosolc=13) |
| CLWRF gases | `module_physics_init.F:929-941` read_CAMgases(.true., "CAM") at init, then camrad reads it each call (ghg_input=1) |
| LW / SW | `module_radiation_driver.F:1972` CAMRAD(dolw=T,dosw=F) and `:2533` CAMRAD(dolw=F,dosw=T) — two separate calls |
| doabsems | `module_radiation_driver.F:1601` every `cam_abs_freq_s`; arm H = held REAL abstot/absnxt/emstot of arm L on a perturbed state |
| coszen / solcon | radconst(julian), calc_coszen(julian, xtime + radt/2, gmt) |
| operands | first_rk_step_part1: P = p_hyd, P8W = p_hyd_w, PI = pi_phy, T = t_phy; Thompson moisture branch (qice = qi + qs) |

Columns: the 186 RE01 census picks (WN3 0227 CPU-WRF d02/d03, tau 12..36: liq_thin_ocean/liq_thick/liq_land/ice/
snow/mixed/sgs_only/clear, day 12Z and night 00Z/06Z) + 24 augmented real-structured columns (synthetic CLDFRA +
condensate patterns forcing 3-10 maximum-overlap regions, tied fractions, lowest/top-layer cloud, thin layer).
Extra dumps: every radclwmx intermediate (radinp/radtpl/radoz2/trcpth/radems/radabs) and the SW operands
(aqsat RH, get_aerosol AEROSOLt) for unit parity. Consistency check inside the driver: the r8 replay of camrad's
preprocessing + radctl reproduces camrad's REAL outputs exactly (0 mismatches on L/S/H).

Regenerate: `python3 extract.py ... && bash build.sh && (cd run && ../cam01_driver.exe in out lwint swint) &&
python3 parse.py ... && python3 compact.py ...` → `data/fixtures/cam01-compact-v1.npz` (40 columns, committed).

## WRF quirks the port reproduces (each would otherwise break parity)

* radtpl/radems/radabs read the MODULE `co2vmr` = REAL 3.55e-4 (never updated by camrad), trcpth and SW use the
  CLWRF `co2mmr` (431 ppm in 2026);
* unsuffixed literals in r8 code are single precision; gfortran folds REAL constant sub-expressions in single precision
  (`1.15*3.42217e3`, `1./293.`, `1./.3205`);
* held absorptivities are REAL state arrays (`abstot_3d`...): the held path widens float32 values;
* the maximum-random overlap sorts cloud layers with a Shell sort (`sortarray`): emulated exactly as stable chain sorts;
* strat_volcanic = .false. → all LW aerosol transmissions are exactly 1; aerosol_indirect has no outputs.

## Results (full 210-column fixture; tests run on the compact one)

LW: GLW, OLR, LWUPT/C, LWDNT/C, LWUPB/C, LWDNB/C bit-identical (0/210) in both the fresh and the held arm; held REAL
abstot/absnxt/emstot bit-identical; RTHRATENLW 4/9240 values differ by <= 2 float32 ulp; r8 intermediates <= 1e-14
relative. SW (after the aerosol-index fix): rh/aerosol/tauxcl/tauxci bitwise, fluxes <= 2.5e-13 relative, flux
differences <= 1.1e-11, heating <= 8.1e-11 (XLA vs glibc exp ulps amplified by the divergence; <= 1.9e-13 of the
incident flux), REAL RTHRATENSW 0 mismatches; SWIDX (76 stress columns: 33 maximum-overlap fallback, 34 with > 15
configurations, 29 tied) at the same gates.

Oracle history: CAM01 v1 left the Registry aerosolc indices P_sul..P_volc at their module default 1 (a standalone driver
never runs set_scalar_indices_from_config) -> all CAM aerosol species collapsed into one slot. Found by the SW port,
fixed by setting P_* = 2..13 (inc/scalar_indices.inc, package active); every LW output was unchanged by the fix.

OFF-path: release RRTMG 4/4 (fast defaults, Pallas interpret) and legacy RRTM1/Dudhia1 physics+boundary step jaxprs
(ordinary + radiation) identical between main 464e3e51f and the branch (cam_off_identity.py).

Coupled CPU smoke (cam_swiss_smoke.py, Swiss 42x42 d01, Noah-MP/Thompson/MYNN, ra_lw = ra_sw = 3 via the namelist API,
legacy defaults, pure-JAX Thomas solve in the harness): 3 steps incl. a radiation step, all carry leaves finite, census
guards 0, held RTHRATEN -14.9..+0.8 K/day (night start), CAM GLW to Noah-MP 187-269 W/m2; first step 864 s (compile),
then 14 s/step on a contended core.

## Operational deviations (disclosed)

1. Absorptivities/emissivity recomputed at every radiation call = WRF with `cam_abs_freq_s <= radt` (WRF default
   21600 s holds them); the held path is implemented and oracle-tested in the kernel, its operational carry is open.
   Size on the CAM01 H arm (state +-0.8 K / +-5 % qv away from the abs/ems state): LW heating <= 0.38 K/day (rms
   0.035), GLW <= 0.6 W/m2, OLR <= 0.34 W/m2.
2. XICE = 0 (no sea-ice field in the operational State); SNOW = Noah-MP SNEQV over land.
3. On the legacy (non RRTMG 4/4) radiation path the land surface and history SWDOWN/GLW now come from CAM, but the other
   radiation flux history fields (TOA/surface up/down, OLR) remain RRTMG-derived — pre-existing for every non-RRTMG
   scheme.
4. CPU-qualified only; GPU and coupled-forecast qualification pending. CLI binding of ra_* pending lane o1-nlbind.
