# ADR-009 - RRTMG JAX Transfer Solver State

Date: 2026-05-21
Author: M5-S3.x worker amendment (Codex gpt-5.5); M5-S3.y non-acceptance update (Codex gpt-5.5); M5-S3.zzzz SW parity amendment (Codex gpt-5.5)
Status: SW-PARITY, LW-NOT-PARITY
Scope: M5-S3.x/M5-S3.y RRTMG shortwave and longwave radiation column rewrite.

## Decision

M5-S3.x replaces the M5-S3 hand-rolled SW reflection stack and fabricated gas curve with a real transfer-solver skeleton:

- SW now computes RRTMG-style molecular columns from pressure interfaces and gas VMRs, multiplies them by extracted reduced-g absorption coefficients, combines gas/Rayleigh/cloud optical properties, applies Joseph-Wiscombe-Weinman delta scaling, evaluates the Meador-Weaver/Joseph Eddington two-stream coefficients, and solves the vertical adding problem using the WRF `vrtqdr_sw` recurrence.
- LW now computes RRTMG-style molecular columns, evaluates per-band/g-point optical depths, applies the `rtrnmc` band diffusivity-angle convention, runs top-down/down-up source recurrences with surface emissivity/reflection, and sums g-points with the extracted quadrature weights.
- The table asset now stores original SW cloud extinction/SSA/asymmetry from WRF source tables instead of only pre-delta-scaled cloud coefficients, so delta scaling is visible in the JAX solver.

This is a real transfer rewrite relative to M5-S3. It is **not accepted as full RRTMG parity** because strict Tier-1 still fails. The remaining blocker is the incomplete optical-depth/source path: the NPZ still exposes reduced reference-pressure absorption profiles, not the full `setcoef` + band-specific `taumol` interpolation state, and LW still lacks full Planck-fraction (`fracref*`) interpolation.

M5-S3.y forced the local SW oracle to Eddington (`module_ra_rrtmg_sw.F:2632`, `kmodts=1`) and rebuilt the WRF harness, then exposed native reduced SW `absa/absb/selfref/forref/sfluxref/Rayleigh` tables and LW `totplnk/totplk16` Planck tables as JAX table leaves. This is still **not PARITY**: strict broadband Tier-1 remains false, the WRF harness still does not emit per-band TOA/surface fluxes, and the SW native-table path increases SW HLO beyond the 500 KB budget.

M5-S3.zzzz closes the SW broadband blocker. The WRF harness now emits `cldprmc_sw` MCICA cloud optics plus `spcvmc_sw` clear/cloud/blended two-stream and pre-broadband per-g-point flux dumps. JAX SW now uses WRF-native single-precision `setcoef_sw`/`taumol_sw`/cloud-optics arithmetic where lookup-bin sensitivity matters, WRF climatological ozone for `o3input=0`, WRF exponential lookup semantics, no optical-depth cap fudge, and the WRF `spcvmc_sw` clear/cloud `reftra_sw` then output-blend order. Strict SW Tier-1 now passes; LW remains not accepted by this ADR because M5-S3.zzzz did not edit the LW implementation.

## WRF Source Mapping

The local WRF source declares 14 shortwave bands and 112 reduced shortwave g-points in `module_ra_rrtmg_sw.F:31-37`; it declares 16 longwave bands and 140 reduced longwave g-points in `module_ra_rrtmg_lw.F:76-82`.

The WRF wrappers bound by the harness are `RRTMG_SWRAD` at `module_ra_rrtmg_sw.F:10034-10100` and `RRTMG_LWRAD` at `module_ra_rrtmg_lw.F:11570-11607`. They call the internal AER drivers at `module_ra_rrtmg_sw.F:11462-11484` and `module_ra_rrtmg_lw.F:12768-12778`.

SW source mappings:

- Molecular-column and interpolation setup follows `setcoef_sw`: pressure/temperature interpolation factors and scaled molecular columns are defined at `module_ra_rrtmg_sw.F:2843-3099`.
- Band/g-point optical depth is delegated by WRF to `taumol_sw` at `module_ra_rrtmg_sw.F:3190-4653`; the current SW JAX path ports the branch structure needed by the M5-S3.zzzz oracle and validates all 14 SW bands against the WRF intermediate dump.
- Incoming solar g-point source is accumulated in `spcvmc_sw` at `module_ra_rrtmg_sw.F:8470-8476`.
- Optical properties and delta scaling are combined at `module_ra_rrtmg_sw.F:8554-8558` and `module_ra_rrtmg_sw.F:8603-8610`, with the cloud delta-scaling branch at `module_ra_rrtmg_sw.F:8638-8644`.
- Eddington two-stream coefficients are present in `reftra_sw` at `module_ra_rrtmg_sw.F:2647-2652`; homogeneous and particular reflectance/transmittance expressions continue through `module_ra_rrtmg_sw.F:2672-2802`.
- WRF `reftra_sw` is bound with `kmodts=1` at `module_ra_rrtmg_sw.F:2632`, so the fixture and JAX path both use the Eddington branch.
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

- `artifacts/m5/tier1_rrtmg_sw_parity.json`: `pass=true`; max residuals are `flux_down=0.07147216796875 W m-2`, `flux_up=0.0467681884765625 W m-2`, `toa_up=0.0267333984375 W m-2`, `column_absorbed=0.0353851318359375 W m-2`, and `heating_rate=3.421277171294793e-08 K s-1`.
- `artifacts/m5/rrtmg_intermediate_validation.json`: SW `setcoef`, `taur`, all 14 `taug` bands, `sfluxzen`, and `cldprmc` pass; A1 keeps the WRF `max(0.01, cldfrac)` denominator floor; A2 keeps WRF clear/cloud `reftra_sw` followed by output blending. The remaining SW intermediate residuals are recorded for `spcvmc` bands 10 and 13, but broadband/pre-band Tier-1 SW is within strict tolerance.
- `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`: pins harness SHA `43ab8af87e869f002f162f5cfc4311802957b312738d80f95b4a73a4e2dbc1a8`, oracle SHA `eeef60540bdcf1a20d90cecefd4ef264f54fa012ce18d23ab3edfbb52d4f4aca`, and `nm_symbol_sha256=2dd3acc839cecd3657037150dcd111ff2440aa2ebf32d1f5636be2d7f61c476d`.
- `artifacts/m5/tier1_rrtmg_lw_parity.json`: `pass=false`; LW remains sister-sprint scope and this branch has no diff in `src/gpuwrf/physics/rrtmg_lw.py`.
- `artifacts/m5/tier2_rrtmg_invariants.json`: `pass=true`.
- `artifacts/m5/rrtmg_gate_result.json`: `gate_status=FALLBACK`, `tier1_sw_pass=true`, `tier1_lw_pass=false`, `tier2_pass=true`; the fallback also records the raw launch count honestly (`443`, above threshold). No `min(raw, cap)` launch reporting is used.

## Consequences

Positive consequence: SW broadband and per-field Tier-1 parity are accepted under the strict WRF-driver fixture. The new cldprmc/spcvmc oracle makes the former R-8/R-9 hypotheses directly testable and leaves numeric residuals visible instead of clipping or capping them.

Blocking consequence: combined M5 RRTMG parity still depends on the M5-S3.zzz LW closeout and later launch/HLO optimization. M6 coupled validation remains blocked until LW Tier-1 also passes and the gate no longer reports fallback.
