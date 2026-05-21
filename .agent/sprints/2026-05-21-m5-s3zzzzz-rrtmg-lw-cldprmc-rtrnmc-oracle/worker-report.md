# M5-S3.zzzzz Worker Report - RRTMG LW cldprmc/rtrnmc Oracle

## Objective

Close the LW-only M5-S3.zzzzz scope from the manager interface freeze: add WRF-faithful intermediate oracles at the `cldprmc_lw` to `rtrnmc` boundary, validate every LW band 1-16 per quantity, and bring strict LW Tier-1 broadband/heating parity to PASS without touching `src/gpuwrf/physics/rrtmg_sw.py`.

The sprint outcome is LW-PARITY for the current strict Tier-1 fixture. Overall M5 RRTMG is not complete in this branch because SW Tier-1 remains false and is owned by the parallel M5-S3.zzzz sprint.

## Files Changed

- `scripts/wrf_rrtmg_harness.f90`
  - Added `cldprmc_lw_*` and `rtrnmc_*` binary records with interface-freeze names.
  - Added direct WRF `mcica_subcol_lw`, `cldprmc`, and `rtrnmc` boundary capture.
  - Added WRF `INIRAD/O3DATA` ozone profile use for LW low-level oracles.
  - Added the WRF wrapper top-buffer temperature adjustment used before `rrtmg_lw`.
- `scripts/m5_generate_rrtmg_fixture.py`
  - Extended the binary parser and manifest writer for the new `lw_*` NPZ leaves.
- `src/gpuwrf/physics/rrtmg_lw.py`
  - Added LW cloud table extraction for WRF `cldprmc`.
  - Added MCICA KISS cloud-mask generation, LW cloud optical depth assembly, WRF climatological ozone, top-buffer temperature handling, and `rtrnmc` per-g-point recurrence.
  - Production LW flux now comes from summed per-g-point `rtrnmc` outputs.
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`
  - Added required validators: `validate_lw_cldprmc_taucmc`, `validate_lw_cldprmc_cldfmc`, `validate_lw_rtrnmc_per_gpoint_flux`, `validate_lw_rtrnmc_source_recurrence`, and `validate_lw_rtrnmc_tfn_tbl`.
  - Added `lw_cldprmc`, `lw_rtrnmc`, `lw_cldprmc_bands`, and `lw_rtrnmc_bands` artifact sections.
- `data/fixtures/rrtmg-intermediate-oracle-v1.npz`
  - Added `lw_cldprmc_cldfmc`, `lw_cldprmc_taucmc`, `lw_rtrnmc_pfracs`, `lw_rtrnmc_plansum`, `lw_rtrnmc_tfn_tbl_output`, `lw_rtrnmc_zfd_per_gpoint`, and `lw_rtrnmc_zfu_per_gpoint`.
- `fixtures/manifests/*rrtmg*`, `fixtures/samples/analytic-rrtmg-lw-column-v1.npz`, `data/fixtures/analytic-rrtmg-lw-column-v1/full.npz`, and `artifacts/m5/*`
  - Regenerated proof objects from the rebuilt harness.
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`
  - Amended to `SW-PARTIAL/UNKNOWN, LW-PARITY`.
- `tests/test_m5_rrtmg_*.py`
  - Extended tests for new LW oracle leaves and LW Tier-1 pass posture.

No changes were made to `src/gpuwrf/physics/rrtmg_sw.py`; `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py` is empty.

## WRF Citations

- WRF LW wrapper top-buffer temperature profile: `module_ra_rrtmg_lw.F:12329-12378`.
- WRF LW ozone climatology call and profile integration: `module_ra_rrtmg_lw.F:12398-12418` and `module_ra_rrtmg_lw.F:12842-13035`.
- WRF MCICA stochastic cloud generation: `module_ra_rrtmg_lw.F:2236-2242`, `module_ra_rrtmg_lw.F:2389-2401`, `module_ra_rrtmg_lw.F:2449-2457`, and KISS `module_ra_rrtmg_lw.F:2688-2706`.
- WRF LW cloud-optics assembly: `cldprmc`, `module_ra_rrtmg_lw.F:2738-3027`.
- WRF LW transfer/source recurrence and per-band flux/heating accumulation: `rtrnmc`, `module_ra_rrtmg_lw.F:3253-3515`.

## Per-Band LW Intermediate Table

| Band | cldprmc | taucmc max abs | rtrnmc | pfracs max abs | tfn max abs | zfd max abs | zfu max abs | debt |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | PASS | 1.804e-05 | PASS | 0.000e+00 | 4.349e-07 | 2.666e-06 | 2.250e-06 | none |
| 2 | PASS | 2.467e-05 | PASS | 0.000e+00 | 1.462e-05 | 2.005e-06 | 2.237e-06 | none |
| 3 | PASS | 2.602e-05 | PASS | 8.436e-09 | 1.429e-05 | 2.613e-06 | 2.176e-06 | none |
| 4 | PASS | 2.308e-05 | PASS | 8.144e-09 | 1.271e-04 | 1.265e-05 | 1.192e-05 | none |
| 5 | PASS | 2.235e-05 | PASS | 7.407e-09 | 1.111e-04 | 1.106e-05 | 1.163e-05 | none |
| 6 | PASS | 1.754e-05 | PASS | 0.000e+00 | 1.389e-05 | 5.078e-06 | 5.478e-06 | none |
| 7 | PASS | 1.502e-05 | PASS | 1.192e-06 | 1.462e-05 | 1.218e-05 | 2.839e-05 | none |
| 8 | PASS | 1.258e-05 | PASS | 0.000e+00 | 1.333e-05 | 2.121e-06 | 2.550e-06 | none |
| 9 | PASS | 1.354e-05 | PASS | 1.341e-08 | 9.606e-05 | 1.028e-04 | 6.480e-05 | none |
| 10 | PASS | 1.414e-05 | PASS | 0.000e+00 | 7.773e-06 | 1.522e-06 | 1.267e-06 | none |
| 11 | PASS | 1.959e-05 | PASS | 0.000e+00 | 5.648e-05 | 2.977e-06 | 3.008e-06 | none |
| 12 | PASS | 8.871e-06 | PASS | 1.835e-08 | 1.333e-05 | 9.924e-07 | 1.081e-06 | none |
| 13 | PASS | 1.073e-05 | PASS | 2.119e-08 | 1.091e-05 | 3.579e-07 | 3.408e-07 | none |
| 14 | PASS | 8.488e-06 | PASS | 0.000e+00 | 4.103e-08 | 6.074e-07 | 5.783e-07 | none |
| 15 | PASS | 5.357e-06 | PASS | 4.817e-09 | 1.154e-05 | 2.508e-07 | 2.643e-07 | none |
| 16 | PASS | 1.144e-05 | PASS | 2.834e-08 | 1.221e-05 | 1.860e-07 | 1.740e-07 | none |

## Commands Run

- `bash scripts/wrf_rrtmg_harness_build.sh`
- `nm data/scratch/wrf_rrtmg_harness | grep -E "rrtmg_lw_(rad|cldprmc|rtrnmc|taumol)|spcvmc_sw|cldprmc_sw" | head`
- `PYTHONPATH=src python scripts/m5_generate_rrtmg_fixture.py`
- `JAX_PLATFORM_NAME=cpu PYTHONPATH=src python -m gpuwrf.validation.rrtmg_intermediate_oracles`
- `PYTHONPATH=src python scripts/m5_run_rrtmg.py`
- `PYTHONPATH=src python scripts/m5_gate_rrtmg.py`
- `cat artifacts/m5/rrtmg_intermediate_validation.json | jq '.lw_cldprmc.pass, .lw_rtrnmc.pass'`
- `cat artifacts/m5/rrtmg_per_band_status.json | jq '([.lw_cldprmc_bands[].intermediate_gate] | unique), ([.lw_rtrnmc_bands[].intermediate_gate] | unique)'`
- `cat artifacts/m5/tier1_rrtmg_lw_parity.json | jq '.pass, .per_field_max_abs_err'`
- `PYTHONPATH=src pytest -q tests/test_m5_rrtmg_*.py`
- `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py`

## Proof Objects Produced

- `artifacts/m5/rrtmg_intermediate_validation.json`: `pass=true`, `lw_cldprmc.pass=true`, `lw_rtrnmc.pass=true`.
- `artifacts/m5/rrtmg_per_band_status.json`: 16/16 `lw_cldprmc_bands` PASS and 16/16 `lw_rtrnmc_bands` PASS.
- `artifacts/m5/tier1_rrtmg_lw_parity.json`: `pass=true`; maximum LW flux residual is `1.1974164334560555e-4 W m-2`, maximum LW heating-rate residual is `3.577208162086794e-8 K s-1`.
- `artifacts/m5/tier1_rrtmg_sw_parity.json`: `pass=false`; this is outside this worker's code ownership and remains sister-sprint/manager scope.
- `artifacts/m5/tier2_rrtmg_invariants.json`: `pass=true`.
- `artifacts/m5/rrtmg_gate_result.json`: `gate_status=FALLBACK`, with `tier1_lw_pass=true`, `tier1_sw_pass=false`, and launch count `454`.
- `pytest -q tests/test_m5_rrtmg_*.py`: 16 passed.

## Unresolved Risks

- Overall RRTMG is not complete until SW Tier-1 passes in the parallel M5-S3.zzzz workstream.
- The LW path is correctness-closed for this fixture but has performance debt: `rrtmg_profile.json` reports `400` LW launch markers and `3,943,207` LW HLO bytes. This is honest fallback evidence, not a GPU performance acceptance claim.
- There is a plain-intermediate versus jitted-production MCICA seed sensitivity caused by XLA expression fusion around the pressure seed path. The committed proof objects preserve the explicit WRF boundary oracle and the production Tier-1 artifact separately. Future performance cleanup should remove this sensitivity while preserving both proof objects.

## Next Decision Needed

Manager should wait for the SW M5-S3.zzzz closeout. If SW passes, ADR-009 can move from `SW-PARTIAL/UNKNOWN, LW-PARITY` to `SW-PARITY, LW-PARITY`. If SW remains false, keep ADR-009 split and dispatch the next SW-focused corrective sprint. In parallel, a performance/restructuring sprint is needed before claiming M5 launch/HLO budget success for LW.
