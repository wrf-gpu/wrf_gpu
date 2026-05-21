# ADR-009 - RRTMG JAX Transfer Solver State

Date: 2026-05-21
Author: M5-S3.x worker amendment (Codex gpt-5.5); M5-S3.y non-acceptance update (Codex gpt-5.5); M5-S3.zzz LW closeout update (Codex gpt-5.5); M5-S3.zzzzz LW cldprmc/rtrnmc parity update (Codex gpt-5.5)
Status: SW-PARTIAL/UNKNOWN, LW-PARITY after M5-S3.zzzzz
Scope: M5-S3.x/M5-S3.y RRTMG shortwave and longwave radiation column rewrite.

## Decision

M5-S3.x replaces the M5-S3 hand-rolled SW reflection stack and fabricated gas curve with a real transfer-solver skeleton:

- SW now computes RRTMG-style molecular columns from pressure interfaces and gas VMRs, multiplies them by extracted reduced-g absorption coefficients, combines gas/Rayleigh/cloud optical properties, applies Joseph-Wiscombe-Weinman delta scaling, evaluates the Meador-Weaver/Joseph Eddington two-stream coefficients, and solves the vertical adding problem using the WRF `vrtqdr_sw` recurrence.
- LW now computes RRTMG-style molecular columns, evaluates per-band/g-point optical depths, applies the `rtrnmc` band diffusivity-angle convention, runs top-down/down-up source recurrences with surface emissivity/reflection, and sums g-points with the extracted quadrature weights.
- The table asset now stores original SW cloud extinction/SSA/asymmetry from WRF source tables instead of only pre-delta-scaled cloud coefficients, so delta scaling is visible in the JAX solver.

This is a real transfer rewrite relative to M5-S3. It is **not accepted as full RRTMG parity** because strict Tier-1 still fails. The remaining blocker is the incomplete optical-depth/source path: the NPZ still exposes reduced reference-pressure absorption profiles, not the full `setcoef` + band-specific `taumol` interpolation state, and LW still lacks full Planck-fraction (`fracref*`) interpolation.

M5-S3.y forced the local SW oracle to Eddington (`module_ra_rrtmg_sw.F:2632`, `kmodts=1`) and rebuilt the WRF harness, then exposed native reduced SW `absa/absb/selfref/forref/sfluxref/Rayleigh` tables and LW `totplnk/totplk16` Planck tables as JAX table leaves. This is still **not PARITY**: strict broadband Tier-1 remains false, the WRF harness still does not emit per-band TOA/surface fluxes, and the SW native-table path increases SW HLO beyond the 500 KB budget.

M5-S3.zzz closes the LW gas/source-input gap at the intermediate-oracle boundary: all 16 LW `taumol` branches now use WRF-native reduced `absa/absb/selfref/forref/minorref/fracref` data and pass per-band `taug` and `fracs` checks against `data/fixtures/rrtmg-intermediate-oracle-v1.npz` at `abs<=1e-8 + rel<=1e-4`. This still does **not** promote ADR-009 to parity. Strict LW Tier-1 broadband flux remains false after the gas/input closure (`flux_down` max abs `59.568065480560136 W m-2`, `flux_up` max abs `46.99548470324402 W m-2`, `toa_up` max abs `23.93536747974062 W m-2`), so the remaining root cause is downstream of `taumol`: the JAX LW transfer/source recurrence is not yet bound to WRF `cldprmc`/`rtrnmc` intermediate oracles.

M5-S3.zzzzz closes that LW downstream gap. The harness now dumps LW-specific `cldprmc_lw_*` and `rtrnmc_*` records at the `cldprmc` to `rtrnmc` boundary, and the JAX LW path ports the WRF MCICA cloud mask, `cldprmc` optical-depth assembly, climatological ozone profile, top-buffer temperature adjustment, `rtrnmc` source lookup, surface reflection, and per-g-point flux recurrence. All 16 LW bands pass `lw_cldprmc_*` and `lw_rtrnmc_*` intermediate gates, and strict LW Tier-1 now passes. ADR-009 is therefore **LW-PARITY** for this fixture and implementation scope. SW remains **PARTIAL/UNKNOWN** in this branch because `artifacts/m5/tier1_rrtmg_sw_parity.json` is still `pass=false`; the parallel M5-S3.zzzz SW sprint owns that closeout.

## WRF Source Mapping

The local WRF source declares 14 shortwave bands and 112 reduced shortwave g-points in `module_ra_rrtmg_sw.F:31-37`; it declares 16 longwave bands and 140 reduced longwave g-points in `module_ra_rrtmg_lw.F:76-82`.

The WRF wrappers bound by the harness are `RRTMG_SWRAD` at `module_ra_rrtmg_sw.F:10034-10100` and `RRTMG_LWRAD` at `module_ra_rrtmg_lw.F:11570-11607`. They call the internal AER drivers at `module_ra_rrtmg_sw.F:11462-11484` and `module_ra_rrtmg_lw.F:12768-12778`.

SW source mappings:

- Molecular-column and interpolation setup follows `setcoef_sw`: pressure/temperature interpolation factors and scaled molecular columns are defined at `module_ra_rrtmg_sw.F:2843-3099`.
- Band/g-point optical depth is delegated by WRF to `taumol_sw` at `module_ra_rrtmg_sw.F:3190-4653`; the current JAX code does not yet port each band-specific branch.
- Incoming solar g-point source is accumulated in `spcvmc_sw` at `module_ra_rrtmg_sw.F:8470-8476`.
- Optical properties and delta scaling are combined at `module_ra_rrtmg_sw.F:8554-8558` and `module_ra_rrtmg_sw.F:8603-8610`, with the cloud delta-scaling branch at `module_ra_rrtmg_sw.F:8638-8644`.
- Eddington two-stream coefficients are present in `reftra_sw` at `module_ra_rrtmg_sw.F:2647-2652`; homogeneous and particular reflectance/transmittance expressions continue through `module_ra_rrtmg_sw.F:2672-2802`.
- WRF compiled oracle caveat: local `reftra_sw` sets `kmodts=2` at `module_ra_rrtmg_sw.F:2632`, so the WRF fixture uses the PIFM branch at `module_ra_rrtmg_sw.F:2652-2655`, while the M5-S3.x contract requested the Eddington branch. This mismatch is now explicit.
- Vertical adding follows `vrtqdr_sw` at `module_ra_rrtmg_sw.F:8035-8159`.

LW source mappings:

- Molecular columns, precipitable water, CFC cross-section inputs, and surface emissivity setup are in the LW input path at `module_ra_rrtmg_lw.F:11204-11490`.
- LW pressure/temperature/gas interpolation state follows `setcoef` at `module_ra_rrtmg_lw.F:3556-3921`.
- Band-specific optical-depth and Planck-fraction interpolation is delegated by WRF to `taumol` at `module_ra_rrtmg_lw.F:4824-7942`; M5-S3.zzz ports and validates all 16 reduced-g `taugb*` branches at the intermediate-oracle boundary.
- WRF wrapper ozone climatology and top-buffer temperatures are applied at `module_ra_rrtmg_lw.F:12329-12418` using `INIRAD/O3DATA` from `module_ra_rrtmg_lw.F:12842-13035`.
- LW cloud optical depth follows `cldprmc` at `module_ra_rrtmg_lw.F:2738-3027`.
- `rtrnmc` diffusivity angles use the band rules at `module_ra_rrtmg_lw.F:3253-3261`.
- LW downward/upward correlated-k source recurrences and surface reflection are at `module_ra_rrtmg_lw.F:3317-3436` and `module_ra_rrtmg_lw.F:3439-3468`.
- LW band/g-point flux accumulation and heating conversion are at `module_ra_rrtmg_lw.F:3475-3515`.

## Implemented Formulas

SW delta scaling follows Joseph, Wiscombe, and Weinman (1976): `f = g^2`, `tau' = (1 - f omega) tau`, `omega' = (1 - f) omega / (1 - f omega)`, and `g' = (g - f) / (1 - f)`. WRF applies the same algebra in `module_ra_rrtmg_sw.F:8603-8610` and `module_ra_rrtmg_sw.F:8638-8644`.

SW Eddington coefficients follow the contract and the Eddington branch in WRF `reftra_sw`: `gamma1 = (7 - omega (4 + 3g))/4`, `gamma2 = -(1 - omega (4 - 3g))/4`, `gamma3 = (2 - 3 mu0 g)/4`, and `gamma4 = 1 - gamma3` (`module_ra_rrtmg_sw.F:2647-2661`). The homogeneous and non-conservative reflectance/transmittance expressions mirror `module_ra_rrtmg_sw.F:2714-2802`. The vertical adding recurrence mirrors `module_ra_rrtmg_sw.F:8108-8156`.

LW follows Mlawer et al. (1997) correlated-k structure at the reduced-g-point level. M5-S3.zzz computes WRF-style `setcoef` columns/indices, 16 per-band gas optical-depth branches, minor gas continua, CFC/CCl4 cross sections, and `fracrefa/fracrefb` Planck fractions from the pinned WRF DATA payload. M5-S3.zzzzz adds the WRF wrapper ozone/top-buffer inputs plus `cldprmc` and `rtrnmc` boundary validation, so the LW path is accepted as parity for the current strict Tier-1 fixture.

## Validation And Gate Status

Latest regenerated proof objects after M5-S3.zzzzz:

- `artifacts/m5/rrtmg_intermediate_validation.json`: `pass=true`; all 16 LW `taug/fracs`, `lw_cldprmc_*`, and `lw_rtrnmc_*` per-band gates pass.
- `artifacts/m5/rrtmg_per_band_status.json`: all LW bands and all `lw_cldprmc_bands` / `lw_rtrnmc_bands` entries are `FULL_BRANCH_ACCEPTED`; no LW band is left on nearest-pressure fallback.
- `artifacts/m5/tier1_rrtmg_sw_parity.json`: `pass=false`; M5-S3.zz/S3.zzzz SW broadband debt remains outside this ADR update.
- `artifacts/m5/tier1_rrtmg_lw_parity.json`: `pass=true`; max residuals include `flux_down=0.00011974164334560555 W m-2`, `flux_up=0.00009836681255137592 W m-2`, `toa_up=0.000033088221243815497 W m-2`, `column_net_heating=0.00007607716526081276 W m-2`, and `heating_rate=3.577208162086794e-08 K s-1`.
- `artifacts/m5/tier2_rrtmg_invariants.json`: `pass=true`.
- `artifacts/m5/rrtmg_profile.json`: M5-S3.zzzzz HLO sizes are `1409481` bytes SW and `3943207` bytes LW; raw launch marker count is honestly `454` (`54` SW + `400` LW), so launch and HLO budgets still fail. No `min(raw, cap)` launch reporting is used.
- `artifacts/m5/rrtmg_gate_result.json`: `gate_status=FALLBACK`, `tier1_lw_pass=true`, `tier1_sw_pass=false`, `tolerance_regime=strict`, `oracle_regime=wrf-driver`.

## Consequences

Positive consequence: M5-S3.x removes the fabricated `log1p` SW gas curve, uses WRF-style molecular columns, implements visible SW delta scaling and Eddington adding, and moves LW to an explicit reduced-g correlated-k recurrence. The new tests lock those formulas.

Blocking consequence: full M5 RRTMG completion remains blocked until the parallel SW closeout proves SW Tier-1 parity and the launch/HLO budget decision is handled. LW no longer blocks on correctness for the current Tier-1 fixture, but it carries performance debt: the production LW HLO is much larger and has many more launch markers after the explicit WRF-faithful `cldprmc`/`rtrnmc` path.
