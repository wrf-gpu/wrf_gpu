# M5-S3.zzzz Worker Report - RRTMG SW cldprmc/spcvmc Oracle

## Objective

Close the M5-S3.zzzz shortwave contract under reviewer Option A: add WRF intermediate oracle dumps for `cldprmc_sw` and `spcvmc_sw`, use them to confront A1/R-8 and A2/R-9, regenerate the pinned oracle/manifest proof, and bring strict SW Tier-1 broadband/per-field parity to PASS without touching LW implementation scope.

## Files Changed

- `scripts/wrf_rrtmg_harness.f90`: added `cldprmc_sw` dumps for `pcldfmc`, `ptaucmc`, `pasycmc`, `pomgcmc`, and `ptaormc`; added `spcvmc_sw` stage dumps for clear/cloud/blended `zref/ztra/zrefd/ztrad`, direct-beam transmittance, raw `zfd/zfu`, and weighted pre-broadband per-g-point flux. The harness now calls the WRF low-level `reftra_sw`/`vrtqdr_sw` path to preserve source semantics and keeps the actual `spcvmc_sw` symbol call for nm verification.
- `scripts/m5_generate_rrtmg_fixture.py`: parses the appended cloudy/spcvmc binary records, includes low-cloud A1 cases with cloud fraction in `(0, 0.01)`, and persists the harness nm symbol SHA in `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`.
- `scripts/extract_rrtmg_tables.py`: corrected the SW liquid-cloud radius grid so WRF `index=int(radliq-1.5)` maps to the correct table row.
- `src/gpuwrf/physics/rrtmg_sw.py`: aligned SW setcoef/taumol/cloud optical assembly with WRF native real precision, added WRF climatological ozone for `o3input=0`, removed the prior optical-depth cap path from reftra, uses WRF exponential lookup semantics, uses WRF MCICA seed rounding, keeps the WRF cloud denominator floor, uses cloud-active `reftra` masks matching `pcldfmc > repclc`, and reports `column_absorbed` as WRF top net minus surface net.
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`: added the required cldprmc/spcvmc validation helpers and A1/A2 decision records.
- `tests/test_m5_rrtmg_*.py`: updated tests for SW Tier-1 PASS while leaving LW fallback expectations intact.
- `data/fixtures/rrtmg-intermediate-oracle-v1.npz`, analytic SW/LW fixture samples/full fixtures, table fixture, manifests, and M5 artifacts were regenerated.
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`: amended to `SW-PARITY, LW-NOT-PARITY`.
- `src/gpuwrf/physics/rrtmg_lw.py`: not modified; `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_lw.py` is empty.

## A1-A5 Evidence

A1/R-8 cloud floor: WRF source uses the 0.01 cloud-fraction denominator floor in the RRTMG input path at `module_ra_rrtmg_sw.F:11030-11033` and `11064-11065`. The regenerated oracle includes 6 fixture cells with `cloud_box in (0, 0.01)`. JAX `ptaucmc/pasycmc/pomgcmc/ptaormc` matches WRF within the required single-precision floor, so the decision is `keep_floor_matches_wrf`. The earlier suspicion was not caused by the floor.

A2/R-9 reftra ordering: WRF `spcvmc_sw` computes clear and cloudy `reftra_sw` separately, then blends output `zref/ztra/zrefd/ztrad` at `module_ra_rrtmg_sw.F:8651-8670`. JAX now follows that same order. The rejected alternative, "blend optical inputs then call reftra once", is not WRF semantics. The A2 record is therefore `keep_wrf_clear_cloud_reftra_then_output_blend`. Numeric residuals remain visible in `spcvmc_per_band` for bands 10 and 13; these are lookup-bin/precision residuals and do not block the strict SW broadband Tier-1 pass.

A3 pre-sum flux dump: the oracle now contains `sw_spcvmc_zfd`, `sw_spcvmc_zfu`, `sw_spcvmc_zfd_flux`, and `sw_spcvmc_zfu_flux` before broadband accumulation. These are validated per band through `validate_spcvmc_per_gpoint_flux`.

A4 nm proof: after rebuilding the harness, the required symbol grep output was persisted to `artifacts/m5/rrtmg_harness_nm_symbols.txt`. SHA is `2dd3acc839cecd3657037150dcd111ff2440aa2ebf32d1f5636be2d7f61c476d`, matching `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`. Required symbols include `__rrtmg_sw_cldprmc_MOD_cldprmc_sw`, `__rrtmg_sw_spcvmc_MOD_spcvmc_sw`, `__rrtmg_sw_taumol_MOD_taumol_sw`, `__rrtmg_sw_setcoef_MOD_setcoef_sw`, and `__rrtmg_lw_rtrnmc_MOD_rtrnmc`.

A5/ADR: ADR-009 is amended to `SW-PARITY, LW-NOT-PARITY`. SW passes strict Tier-1; LW remains owned by M5-S3.zzz and this branch leaves the LW physics file untouched.

## Before And After

Before this worker, the most recent M5-S3.zz SW Tier-1 state was not parity. The failing broadband fields were led by `flux_down=23.9379 W m-2`, `flux_up=31.0159 W m-2`, `toa_up=31.0159 W m-2`, and `column_absorbed=87.3318 W m-2`. Root causes found here were not a single R-8/R-9 semantic rewrite. They were a stack of smaller WRF-alignment issues: MCICA pressure seed rounding, liquid cloud table radius indexing, WRF climatological ozone, reftra exponential lookup behavior, an inappropriate optical-depth cap, native-real precision at lookup-sensitive setcoef/taumol/cloud stages, and the SW diagnostic column-absorption definition.

After the fixes, `artifacts/m5/tier1_rrtmg_sw_parity.json` is `pass=true`. Max absolute residuals are:

- `flux_down=0.07147216796875 W m-2`
- `flux_up=0.0467681884765625 W m-2`
- `toa_up=0.0267333984375 W m-2`
- `surface_down=0.0343170166015625 W m-2`
- `surface_up=0.006171226501464844 W m-2`
- `column_absorbed=0.0353851318359375 W m-2`
- `surface_absorbed=0.02814483642578125 W m-2`
- `heating_rate=3.421277171294793e-08 K s-1`

## Proof Objects Produced

- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` is 484 KB, below the 50 MB cap.
- `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml` pins harness SHA `43ab8af87e869f002f162f5cfc4311802957b312738d80f95b4a73a4e2dbc1a8`, oracle SHA `eeef60540bdcf1a20d90cecefd4ef264f54fa012ce18d23ab3edfbb52d4f4aca`, and nm symbol SHA `2dd3acc839cecd3657037150dcd111ff2440aa2ebf32d1f5636be2d7f61c476d`.
- `artifacts/m5/rrtmg_intermediate_validation.json` records SW setcoef, taur, all SW taug bands, sfluxzen, cldprmc, A1, A2, and spcvmc per-band diagnostics.
- `artifacts/m5/tier1_rrtmg_sw_parity.json` records SW PASS.
- `artifacts/m5/rrtmg_oracle_clip_pinning_audit.json` records zero non-finite counts and zero non-mask counts at optical cap sentinels 80/500 for the new cldprmc/spcvmc oracle arrays. `pcldfmc` is binary by design.
- `artifacts/m5/rrtmg_gate_result.json` honestly remains `FALLBACK` because LW Tier-1 is false and raw launch count is 443.

## Commands Run

- `bash scripts/wrf_rrtmg_harness_build.sh`
- `nm data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_|cldprmc_"`
- `python scripts/m5_generate_rrtmg_fixture.py`
- `JAX_PLATFORMS=cpu PYTHONPATH=src JAX_ENABLE_X64=true python scripts/m5_run_rrtmg.py` (exits nonzero because LW/intermediate overall still fail; SW Tier-1 is pass)
- `JAX_PLATFORMS=cpu PYTHONPATH=src JAX_ENABLE_X64=true python scripts/m5_gate_rrtmg.py` (exits nonzero with fallback; `tier1_sw_pass=true`)
- `cat artifacts/m5/rrtmg_intermediate_validation.json | jq '.sw'`
- `cat artifacts/m5/tier1_rrtmg_sw_parity.json | jq '.pass, .per_field_max_abs_err'`
- `JAX_PLATFORMS=cpu PYTHONPATH=src JAX_ENABLE_X64=true pytest -q tests/test_m5_rrtmg_*.py` -> `16 passed`
- `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_lw.py` -> empty

## Unresolved Risks

The combined RRTMG gate is still fallback because LW parity is not complete and launch count is above the fallback threshold. SW intermediate `spcvmc_per_band` still records strict ztra residuals in bands 10 and 13 even though all strict SW Tier-1 fields pass; these should be carried as non-blocking precision/lookup-bin residuals unless a later sprint wants exact layer-transmittance parity beyond broadband acceptance.

## Next Decision Needed

Accept SW as parity for M5-S3.zzzz and let M5-S3.zzz finish LW. After both pass, decide whether M5 closeout should treat launch-count/HLO reduction as a separate optimization gate before M6-S8 operational T2 validation.
