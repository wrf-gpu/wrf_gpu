# ADR-009 - RRTMG JAX Transfer Solver State

Date: 2026-05-21
Author: M5-S3.x worker amendment (Codex gpt-5.5); M5-S3.y non-acceptance update (Codex gpt-5.5)
Status: PROPOSED worker draft, M5-S3.y still NOT PARITY
Scope: M5-S3.x/M5-S3.y RRTMG shortwave and longwave radiation column rewrite.

## Decision

M5-S3.x replaces the M5-S3 hand-rolled SW reflection stack and fabricated gas curve with a real transfer-solver skeleton:

- SW now computes RRTMG-style molecular columns from pressure interfaces and gas VMRs, multiplies them by extracted reduced-g absorption coefficients, combines gas/Rayleigh/cloud optical properties, applies Joseph-Wiscombe-Weinman delta scaling, evaluates the Meador-Weaver/Joseph Eddington two-stream coefficients, and solves the vertical adding problem using the WRF `vrtqdr_sw` recurrence.
- LW now computes RRTMG-style molecular columns, evaluates per-band/g-point optical depths, applies the `rtrnmc` band diffusivity-angle convention, runs top-down/down-up source recurrences with surface emissivity/reflection, and sums g-points with the extracted quadrature weights.
- The table asset now stores original SW cloud extinction/SSA/asymmetry from WRF source tables instead of only pre-delta-scaled cloud coefficients, so delta scaling is visible in the JAX solver.

This is a real transfer rewrite relative to M5-S3. It is **not accepted as full RRTMG parity** because strict Tier-1 still fails. The remaining blocker is the incomplete optical-depth/source path: the NPZ still exposes reduced reference-pressure absorption profiles, not the full `setcoef` + band-specific `taumol` interpolation state, and LW still lacks full Planck-fraction (`fracref*`) interpolation.

M5-S3.y forced the local SW oracle to Eddington (`module_ra_rrtmg_sw.F:2632`, `kmodts=1`) and rebuilt the WRF harness, then exposed native reduced SW `absa/absb/selfref/forref/sfluxref/Rayleigh` tables and LW `totplnk/totplk16` Planck tables as JAX table leaves. This is still **not PARITY**: strict broadband Tier-1 remains false, the WRF harness still does not emit per-band TOA/surface fluxes, and the SW native-table path increases SW HLO beyond the 500 KB budget.

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
- Band-specific optical-depth and Planck-fraction interpolation is delegated by WRF to `taumol` at `module_ra_rrtmg_lw.F:4824-7942`; the current JAX code does not yet port each `taugb*` branch.
- `rtrnmc` diffusivity angles use the band rules at `module_ra_rrtmg_lw.F:3253-3261`.
- LW downward/upward correlated-k source recurrences and surface reflection are at `module_ra_rrtmg_lw.F:3317-3436` and `module_ra_rrtmg_lw.F:3439-3468`.
- LW band/g-point flux accumulation and heating conversion are at `module_ra_rrtmg_lw.F:3475-3515`.

## Implemented Formulas

SW delta scaling follows Joseph, Wiscombe, and Weinman (1976): `f = g^2`, `tau' = (1 - f omega) tau`, `omega' = (1 - f) omega / (1 - f omega)`, and `g' = (g - f) / (1 - f)`. WRF applies the same algebra in `module_ra_rrtmg_sw.F:8603-8610` and `module_ra_rrtmg_sw.F:8638-8644`.

SW Eddington coefficients follow the contract and the Eddington branch in WRF `reftra_sw`: `gamma1 = (7 - omega (4 + 3g))/4`, `gamma2 = -(1 - omega (4 - 3g))/4`, `gamma3 = (2 - 3 mu0 g)/4`, and `gamma4 = 1 - gamma3` (`module_ra_rrtmg_sw.F:2647-2661`). The homogeneous and non-conservative reflectance/transmittance expressions mirror `module_ra_rrtmg_sw.F:2714-2802`. The vertical adding recurrence mirrors `module_ra_rrtmg_sw.F:8108-8156`.

LW follows Mlawer et al. (1997) correlated-k structure at the reduced-g-point level: per-band/g-point optical depth, band diffusivity angle, source recurrence, surface reflection, and quadrature accumulation. The implemented recurrence follows the `rtrnmc` control flow at `module_ra_rrtmg_lw.F:3317-3515`, but the source function is still a band-weighted Stefan-Boltzmann approximation because the full WRF `fracref*`/Planck interpolation path is not yet exposed in `rrtmg-tables-v1.npz`.

## Validation And Gate Status

Latest regenerated proof objects:

- `artifacts/m5/tier1_rrtmg_sw_parity.json`: `pass=false`; M5-S3.y max residuals include `flux_down=135.97104293374935 W m-2`, `flux_up=79.21669171335645 W m-2`, `toa_down=67.04542671926288 W m-2`, and `heating_rate=3.632664522070731e-05 K s-1`.
- `artifacts/m5/tier1_rrtmg_lw_parity.json`: `pass=false`; M5-S3.y max residuals include `flux_down=67.25107985479801 W m-2`, `flux_up=44.33055076399509 W m-2`, `column_net_heating=73.67198882864889 W m-2`, and `heating_rate=6.091121542523097e-05 K s-1`.
- `artifacts/m5/tier2_rrtmg_invariants.json`: `pass=true`.
- `artifacts/m5/rrtmg_profile.json`: M5-S3.y HLO sizes are `1312209` bytes SW and `154560` bytes LW; raw launch marker count is honestly `52` (`36` SW + `16` LW), so launch and HLO budgets fail. No `min(raw, cap)` launch reporting is used.
- `artifacts/m5/rrtmg_gate_result.json`: `gate_status=FALLBACK`, `tolerance_regime=strict`, `oracle_regime=wrf-driver`.
- `artifacts/m5/tier1_rrtmg_per_band.json`: `produced=false`; the WRF harness per-band flux extension was not completed, so no per-band parity claim is made.

## Consequences

Positive consequence: M5-S3.x removes the fabricated `log1p` SW gas curve, uses WRF-style molecular columns, implements visible SW delta scaling and Eddington adding, and moves LW to an explicit reduced-g correlated-k recurrence. The new tests lock those formulas.

Blocking consequence: M6 coupled validation remains blocked. Strict Tier-1 does not pass and the raw launch count exceeds the M5-S3.x budget. The next implementation decision is whether to port full SW/LW `setcoef` + `taumol` + LW Planck-fraction interpolation into JAX tables, or to call/translate the existing WRF low-level oracle into generated device-resident optical-depth fixtures for a narrower validation boundary.
