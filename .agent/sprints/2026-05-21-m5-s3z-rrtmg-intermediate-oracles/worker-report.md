# M5-S3.z Worker Report - RRTMG Intermediate Oracles

## objective

Implement the M5-S3.z methodology correction: extract WRF intermediate state before additional JAX branch transcription, generate per-band flux and intermediate-oracle fixtures, validate JAX branches band-by-band against those oracles, complete the LW non-isothermal Planck source hook, and reduce the M5-S3.y SW HLO regression.  The result is **PARTIAL / NOT PARITY**.  The new WRF intermediate oracle is in place and the SW gas-optical-depth branches pass the per-band `taumol_sw` oracle, but strict Tier-1 flux parity still fails, LW gas/fracs are not parity, total launches remain too high, and ADR-009 was intentionally **not** finalized to PARITY.

## files changed

- `scripts/wrf_rrtmg_harness.f90`
  - Preserved existing formatted fixture output.
  - Appended `#RRTMG_ORACLE_V1_BINARY` stream-unformatted records.
  - Added low-level WRF calls to `setcoef_sw`, `taumol_sw`, `spcvmc_sw`, `setcoef`, `taumol`, and `rtrnmc`.
  - Emits SW/LW per-band flux arrays and intermediate arrays.

- `scripts/m5_generate_rrtmg_fixture.py`
  - Parses the appended binary oracle payload in endian-detected Fortran order.
  - Adds per-band flux arrays to SW/LW fixture payloads.
  - Writes `data/fixtures/rrtmg-intermediate-oracle-v1.npz`.
  - Writes SHA-pinned `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`.

- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`
  - New validation module implementing all requested validation entry points.
  - Writes `artifacts/m5/rrtmg_intermediate_validation.json`.
  - Writes `artifacts/m5/rrtmg_per_band_status.json`.

- `src/gpuwrf/physics/rrtmg_sw.py`
  - Added intermediate-state helper exposing JAX SW `setcoef`/`taumol`/`sfluxzen` arrays for oracle checks.
  - Moved production SW transfer back to compact nearest-pressure gas coefficients after the full 14-band branch path was confirmed to keep the HLO/launch regression.  The validated branch helper remains available for evidence, but production no longer carries the 1.31 MB HLO path.

- `src/gpuwrf/physics/rrtmg_lw.py`
  - Added `RRTMGLWIntermediateState`.
  - Added `dplankup` / `dplankdn` source corrections.
  - Added WRF `tfn_tbl`-equivalent source-factor computation.
  - Added `compute_rrtmg_lw_intermediates`.

- `src/gpuwrf/physics/rrtmg_constants.py`
  - Aligned default trace gas constants to the WRF harness defaults used for the M5-S3.z oracle generation.
  - Added LW `tfn_tbl` constants.

- `scripts/m5_run_rrtmg.py`
  - Runs intermediate-oracle validation and includes it in the emitted run record.

- `tests/test_m5_rrtmg_intermediate_oracles.py`
  - New coverage for oracle fixture shapes, helper pass/fail behavior, and honest artifact writing.

- `tests/test_m5_rrtmg_gate.py`, `tests/test_m5_rrtmg_tier1.py`
  - Updated to assert the current honest fallback result instead of the pre-M5-S3.z gray-zone assumption.

- Regenerated proof artifacts:
  - `fixtures/samples/analytic-rrtmg-sw-column-v1.npz`
  - `fixtures/samples/analytic-rrtmg-lw-column-v1.npz`
  - `data/fixtures/analytic-rrtmg-sw-column-v1/full.npz`
  - `data/fixtures/analytic-rrtmg-lw-column-v1/full.npz`
  - `data/fixtures/rrtmg-intermediate-oracle-v1.npz`
  - `fixtures/manifests/*.yaml`
  - `artifacts/m5/*rrtmg*`

## WRF source citations

- SW `setcoef_sw`: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:2843-3099`
- SW `taumol_sw`: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:3190-4653`
- SW spectral solver entry and internal `taumol_sw` call: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:8196-8450`
- SW flux accumulation in `spcvmc_sw`: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:8740-8745`
- LW `rtrnmc` source logic and `dplankup/dplankdn`: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:3253-3409`
- LW `tfn_tbl` construction: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:8054-8070`
- LW `setcoef` Planck interpolation: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:3556-3921`
- LW `taumol`: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:4824-7950`

## AC status

| AC | status | evidence |
| --- | --- | --- |
| AC1 per-band flux emission | PARTIAL | Harness appends SW `(3,4,14)` and LW `(3,4,16)` per-band flux arrays. These are low-level clear-sky `spcvmc_sw` / `rtrnmc` per-band calls, not yet a full cloudy WRF wrapper per-band decomposition. |
| AC2 intermediate-oracle dumps | DONE | `data/fixtures/rrtmg-intermediate-oracle-v1.npz` is 121 KB, under 30 MB. Manifest validates after `sample_slice_path: null`. |
| AC3 JAX intermediate validation | DONE / FAILING | `artifacts/m5/rrtmg_intermediate_validation.json` exists and marks `pass=false`. SW `taug` per-band passes; SW source/setcoef strict state and all LW `taug/fracs` fail. |
| AC4 LW source machinery | PARTIAL | `dplankup/dplankdn` and `tfn_tbl` source factor are wired. LW Planck and dplank oracle checks pass, but LW gas/fracs are still nearest-pressure approximations and fail per-band. |
| AC5 SW launch fusion | PARTIAL | Production SW HLO reduced from 1,312,183 bytes to 497,603 bytes by reverting production gas optical depth to nearest-pressure. Raw launches remain SW=24, LW=18, total=42, so AC5 launch target is not met. No `min(raw, cap)` fudge was introduced; raw equals reported. |
| AC6 strict Tier-1 | FAIL | SW and LW strict Tier-1 artifacts both `pass=false`. SW max flux-down abs error 110.065 W/m2; LW max flux-down abs error 70.602 W/m2. |
| AC7 ADR-009 PARITY | NOT DONE | ADR-009 was not changed because required evidence is missing. Finalizing it would be false. |
| AC8 per-band debt list | DONE | `artifacts/m5/rrtmg_per_band_status.json` records SW branch/source debt and LW per-band debt. |

## intermediate-oracle evidence

SW branch result: all 14 `taumol_sw` gas optical-depth branches pass the new WRF per-band oracle at `abs<=1e-8 + rel<=1e-4`.

| SW band | taug gate | max rel |
| --- | --- | --- |
| 1 | PASS | 3.621e-06 |
| 2 | PASS | 4.206e-06 |
| 3 | PASS | 5.279e-06 |
| 4 | PASS | 3.189e-06 |
| 5 | PASS | 3.633e-06 |
| 6 | PASS | 8.172e-06 |
| 7 | PASS | 2.158e-06 |
| 8 | PASS | 8.941e-06 |
| 9 | PASS | 3.445e-06 |
| 10 | PASS | 4.613e-06 |
| 11 | PASS | 0.000e+00 |
| 12 | PASS | 9.189e-05 |
| 13 | PASS | 2.629e-06 |
| 14 | PASS | 7.861e-06 |

SW residual debt:

- `sw_taur`: PASS, max abs `5.09e-07`, max rel `1.57e-06`.
- `sw_setcoef_state`: FAIL under the contract's float64-roundoff bar. Main residuals are single-precision WRF oracle/harness effects and top-layer `indself`: `fac00 max_abs=1.40e-05`, `fac01 max_abs=1.61e-05`, `colmol max_abs=5.19e-03`, `indself max_abs=1`.
- `sw_sfluxzen`: FAIL. Max abs `14.57`; the current JAX source selection still differs from WRF `sfluxzen` for band/g-point source distribution.

LW result: Planck source pieces pass, but gas and fraction branches remain nearest-pressure debt.

| LW band | taug/fracs gate | max abs taug | max abs fracs |
| --- | --- | --- | --- |
| 1 | FAIL | 1049887.0 | 0.15491 |
| 2 | FAIL | 48642.36 | 0.08279 |
| 3 | FAIL | 4613.01 | 0.09817 |
| 4 | FAIL | 19147.70 | 0.08808 |
| 5 | FAIL | 1503.42 | 0.08193 |
| 6 | FAIL | 9.40394 | 0.16627 |
| 7 | FAIL | 79.98790 | 0.22985 |
| 8 | FAIL | 79.99392 | 0.18941 |
| 9 | FAIL | 9476.33 | 0.12581 |
| 10 | FAIL | 19528.89 | 0.16404 |
| 11 | FAIL | 56337.29 | 0.19259 |
| 12 | FAIL | 6053.94 | 0.12809 |
| 13 | FAIL | 79.99997 | 0.25566 |
| 14 | FAIL | 21493.58 | 0.45202 |
| 15 | FAIL | 80.0 | 0.5 |
| 16 | FAIL | 79.76061 | 0.28507 |

LW residual/source result:

- `lw_planck_state`: PASS.
- `lw_planck_corrections`: PASS.
- `lw_secdiff`: FAIL only under very tight float64 tolerance; max abs is `1.28e-07`, max rel is `8.26e-08`.
- `lw_taug/fracs`: FAIL for every band because true LW `taumol` branch transcription is not implemented.

## flux-output Tier-1 evidence

Strict Tier-1 remains failing.

| family | pass | key max absolute residuals |
| --- | --- | --- |
| SW | false | `flux_down=110.065`, `flux_up=61.751`, `toa_down=67.044`, `toa_up=33.568`, `heating_rate=2.91e-05 K/s` |
| LW | false | `flux_down=70.602`, `flux_up=44.616`, `toa_up=44.616`, `column_net_heating=73.570`, `heating_rate=5.98e-05 K/s` |
| Tier-2 invariants | true | candidate and real-driver closure checks pass |

## before/after performance evidence

Before this worker, M5-S3.y production SW HLO from the full 14-band branch expansion was `1,312,183` bytes with `36` SW raw launches.

After this worker:

- SW HLO: `497,603` bytes, now under 500 KB.
- LW HLO: `197,157` bytes.
- Raw launches: `SW=24`, `LW=18`, combined `42`.
- Reported launches equal raw HLO marker count: `42 == 42`.
- Gate status: `FALLBACK`, rationale `correctness failed`.

This satisfies the "no launch-count fudge" verifier but does not satisfy the AC5 launch target (`SW<=6`, `LW<=4`, combined `<=10`).

## commands run

```bash
bash scripts/wrf_rrtmg_harness_build.sh
nm data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_|cldprmc_" | head -n 20
python scripts/m5_generate_rrtmg_fixture.py
python scripts/m5_run_rrtmg.py
python scripts/m5_gate_rrtmg.py
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml
python scripts/validate_fixture_manifest.py fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml
python scripts/validate_agentos.py
PYTHONPATH=src JAX_ENABLE_X64=true pytest -q tests/test_m5_rrtmg_*.py
```

Command outcomes:

- Harness build: PASS, binary SHA `313205b94e6528c614fde3fdf19e385fe45cda3f790311cefcce18721903815a`.
- `nm`: PASS; WRF symbols include `spcvmc_sw`, `rtrnmc`, `taumol_sw`, `taumol`, `setcoef_sw`, `setcoef`, and cloud prep symbols.
- Fixture generation: PASS.
- `m5_run_rrtmg.py`: FAIL as expected from strict correctness/intermediate parity; artifacts were still written.
- `m5_gate_rrtmg.py`: FALLBACK, correctness failed.
- Manifest validation: SW/LW/intermediate all PASS.
- AgentOS validation: PASS.
- Focused pytest: PASS, `16 passed`.

## proof objects produced

- `data/fixtures/rrtmg-intermediate-oracle-v1.npz`
- `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`
- `artifacts/m5/rrtmg_intermediate_validation.json`
- `artifacts/m5/rrtmg_per_band_status.json`
- `artifacts/m5/tier1_rrtmg_sw_parity.json`
- `artifacts/m5/tier1_rrtmg_lw_parity.json`
- `artifacts/m5/tier2_rrtmg_invariants.json`
- `artifacts/m5/rrtmg_profile.json`
- `artifacts/m5/rrtmg_gate_result.json`
- `artifacts/m5/hlo_dump/rrtmg_sw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_lw_production.txt`

## unresolved risks

- The harness per-band flux decomposition is low-level clear-sky per-band transfer, not full cloudy WRF wrapper per-band output. It is useful, but it does not yet fully close AC1 for cloudy McICA/WRF wrapper behavior.
- The WRF intermediate oracle is stored from WRF real kind through the local harness. Some `setcoef` fields fail the contract's float64-roundoff bar at roughly single-precision residual levels. The next worker should either rebuild an explicit double-precision extraction path if compatible with the WRF objects, or adjust the contract if the authoritative WRF build is single precision.
- SW `sfluxzen` source selection remains incorrect even though `taug` and `taur` pass. This affects flux-level SW parity.
- LW gas optical-depth and Planck fraction branches are not transcribed; all 16 LW `taug/fracs` gates fail.
- Launch count is still too high even after reducing SW HLO size. A real `lax.scan`/compact branch-table refactor is still required.
- ADR-009 remains NOT-PARITY and should not be changed until strict Tier-1 and intermediate per-band gates pass.

## next decision needed

Manager/reviewer should decide whether M5-S3.zz is:

1. SW-only closeout: fix `sfluxzen` and setcoef precision policy, then re-enable compact full SW branches with a real launch-fusion refactor.
2. LW-focused closeout: implement true LW `taumol` and `fracs` branches using the new oracle layer.
3. Harness-first closeout: upgrade AC1 to full cloudy wrapper per-band dumps before further branch work.

My recommendation is option 1 first. SW gas branches now have strong per-band oracle evidence; the quickest correctness/performance win is to fix `sfluxzen` and launch fusion without expanding scope into full LW `taumol`.
