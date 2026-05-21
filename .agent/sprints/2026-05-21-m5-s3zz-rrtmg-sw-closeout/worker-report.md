# M5-S3.zz Worker Report - RRTMG SW Closeout

Status: PARTIAL delivery. Do not mark ADR-009 as full PARITY and do not claim SW-PARITY yet. The sprint closed the two M5-S3.z reviewer-identified intermediate-oracle defects (`sfluxzen` allocation and `setcoef_sw` tolerance policy), re-enabled the validated SW optical-depth branch in the production path, and exposed a new downstream root cause in broadband transfer/cloud optical handling. Strict Tier-1 SW flux parity remains false.

## Objective

Finalize the RRTMG shortwave closeout by fixing WRF-equivalent `sfluxzen`, applying the agreed single-precision floor to SW `setcoef_sw`, re-enabling the 14 validated SW gas/Rayleigh branches in production, validating Tier-1 SW parity, preserving LW behavior, and amending ADR-009 only if SW parity is actually proven.

## AC Status

| AC | Status | Evidence |
| --- | --- | --- |
| AC1 SW `sfluxzen` allocation | PASS | `artifacts/m5/rrtmg_intermediate_validation.json` reports `.sw.sfluxzen.pass == true`, max_abs `3.814697265625e-6`, max_rel `3.735901459402517e-7`, within abs `1e-8` plus rel `1e-4`. |
| AC2 SW `setcoef` precision policy | PASS | `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` now validates SW setcoef fields at abs `1e-4` plus rel `1e-3`; test coverage asserts this floor in `tests/test_m5_rrtmg_intermediate_oracles.py`. Artifact reports `.sw.setcoef.pass == true`. |
| AC3 Re-enable 14 SW branches via `lax.scan` | PARTIAL | `_sw_taumol_fused` wraps the validated SW optical-depth output behind a band-axis `jax.lax.scan`, and production `_shortwave_impl` now consumes `_sw_setcoef`, `_sw_taumol_fused`, and `_sw_sfluxzen`. `artifacts/m5/rrtmg_per_band_status.json` has 14 SW bands with `implementation_status=FULL_BRANCH_ACCEPTED` and `intermediate_gate=PASS`. Launch/HLO targets are not claimed because Tier-1 flux parity still fails. |
| AC4 Strict Tier-1 SW pass | FAIL | `artifacts/m5/tier1_rrtmg_sw_parity.json` reports `"pass": false`. Current max_abs errors are flux_down `56.4980975502483 W/m2`, flux_up `64.43487698173968 W/m2`, column_absorbed `87.53339162140193 W/m2`, toa_up `30.142989195802443 W/m2`, surface_down `20.55932413682686 W/m2`. SW heating max_abs is `2.478007334543835e-5 K/s`, which is inside the heating threshold, but flux parity is not. |
| AC5 LW no regression | PASS BY EDIT SCOPE, NOT RERUN | `src/gpuwrf/physics/rrtmg_lw.py` was not modified. LW production path and M5-S3.z Planck-source improvement were left untouched. I did not claim a new LW baseline because the full suite was not rerun after the partial SW finding. |
| AC6 ADR-009 update | NOT DONE | ADR-009 was not amended to `SW-PARITY, LW-NOT-PARITY` because SW Tier-1 parity is false. Updating the ADR now would misrepresent project state. |

## Root-Cause Analysis

### Closed root cause 1: `sfluxzen` band/source allocation

M5-S3.z reviewer section 3 identified a band-11 source-flux allocation mismatch: JAX allocated non-zero source in a cell where WRF left the band/g-point source at zero. The WRF behavior is not "always interpolate a source by band"; it is "initialize all source values to zero, then assign only inside the band-specific active lower/upper source loop." WRF initializes `sfluxzen(:)=0.0` in `taumol_sw` at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:3380-3382`. For band 27, which is JAX band index 11, WRF assigns source only in the upper-atmosphere loop and only at `lay.eq.laysolfr`: lines `4446-4464`, including `sfluxzen(ngs26+ig) = scalekur * sfluxref(ig)` at line `4463`.

The JAX fix adds `_source_active` and gates `_sw_sfluxzen` by the same lower/upper loop availability before appending the source array. This preserves WRF's zero initialization for inactive source bands/layers. The resulting proof object is `artifacts/m5/rrtmg_intermediate_validation.json`, where SW `sfluxzen` now passes at the intermediate-oracle threshold.

### Closed root cause 2: `setcoef_sw` precision policy

The original sprint contract tolerance for SW `setcoef_sw` was effectively double-precision oracle tolerance. The WRF harness was built with single-precision reals: compile log `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/compile.log` shows `pgf90 -r4 -i4` at lines `55`, `141`, and `180`, `-DRWORDSIZE=4` at line `56`, and `NATIVE_RWORDSIZE="4" RWORDSIZE="4"` at line `171`. That makes a strict `1e-12` floor inappropriate for fixture values derived from WRF real*4 execution.

Path B was implemented: SW setcoef validation now uses abs `1e-4` plus rel `1e-3`. The JAX setcoef logic also matches WRF's upper-atmosphere `indself=0` behavior: WRF sets `selffac=0`, `selffrac=0`, and `indself=0` above the lower atmosphere at `module_ra_rrtmg_sw.F:3077-3079`, then computes interpolation factors at `3090-3094`. `artifacts/m5/rrtmg_intermediate_validation.json` now reports `.sw.setcoef.pass == true`, and the test file asserts the precision floor so this decision cannot silently revert.

### New root cause: broadband transfer/cloud optics, not gas optical depth or source allocation

After closing `setcoef` and `sfluxzen`, the production path was moved from the old nearest-pressure optical-depth approximation to the validated SW branch output. That improved the intermediate stack but did not close broadband SW flux. The residual is no longer explained by `taumol_sw` gas/Rayleigh optical depth, `sfluxzen`, or setcoef interpolation: those intermediates pass their oracle gates. The failure now appears downstream, in the coupling between cloud optical properties and `spcvmc_sw` broadband two-stream transfer.

I also checked the likely MCICA random-mask source. WRF seeds the KISS generator from the fractional part of the bottom four `pmid` layers at `module_ra_rrtmg_sw.F:1727-1744`, applies random overlap at `1754-1778`, and defines `kissvec` at `2016-2040`. The JAX implementation now mirrors this KISS path, including the single-precision pressure cast before extracting the fractional seed. A local Fortran KISS probe for the marine scenario matched the JAX mask over the inspected layer/subcolumn cells, so the remaining broadband residual is unlikely to be the random overlap mask alone.

The next suspect is the `cldprmc_sw`/`spcvmc_sw` interface. WRF computes cloud optical properties in `cldprmc_sw` starting at `module_ra_rrtmg_sw.F:2077`, including cloud optical inputs/outputs at `2094-2127`, water-cloud interpolation at `2380-2411`, component delta scaling and tau/ssa/asym assembly at `2414-2448`, and asymmetry mixing at `2461-2486`. WRF then uses those properties in `spcvmc_sw`: incoming flux is `adjflux(jb) * zsflxzen(iw) * prmu0` at `8474-8475`; clear/cloud optical parameters and direct transmittance use the original cloud optical depth at `8554-8598`; total-sky optical parameters and clear/cloud mixing occur at `8618-8668`; direct beam transmittance and vertical quadrature are at `8672-8722`; broadband flux accumulation is at `8739-8745`.

The present JAX code approximates that downstream path better than before, including MCICA cloud masking and added liquid/ice/snow table extraction, but without intermediate WRF dumps for `cldprmc_sw` and `spcvmc_sw` it is still a blind production edit. The remaining flux residuals are broadband-transfer sized, not intermediate-gas sized. Recommended next sprint scope is therefore an intermediate oracle for `cldprmc_sw` and `spcvmc_sw`, dumping at minimum `pcldfmc`, `ptaucmc`, `pasycmc`, `pomgcmc`, `ptaormc`, clear/cloud `zref/ztra/zrefd/ztrad`, direct-beam transmittance, and per-g-point `zfd/zfu` before accumulation.

## Files Changed

- `src/gpuwrf/physics/rrtmg_sw.py`: re-enabled validated SW setcoef/taumol/sfluxzen in production, added band-axis `lax.scan`, WRF-like source-active gating, MCICA KISS random overlap, and expanded cloud optical handling.
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`: relaxed SW setcoef tolerance to the WRF single-precision floor and updated per-band status policy text.
- `tests/test_m5_rrtmg_intermediate_oracles.py`: asserts the new SW setcoef tolerance contract.
- `scripts/extract_rrtmg_tables.py`: extracts additional SW cloud coefficients needed by the cloud-optics attempt.
- `src/gpuwrf/physics/rrtmg_tables.py`: loads the added cloud table fields.
- `data/fixtures/rrtmg-tables-v1.npz` and `data/fixtures/rrtmg-tables-v1.json`: regenerated table fixture with the added cloud fields.
- `artifacts/m5/rrtmg_intermediate_validation.json`, `artifacts/m5/rrtmg_per_band_status.json`, `artifacts/m5/tier1_rrtmg_sw_parity.json`: updated proof artifacts showing intermediate closure and remaining Tier-1 failure.
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`: intentionally not modified because SW parity is not proven.

## Commands Run

- `python scripts/extract_rrtmg_tables.py --output data/fixtures/rrtmg-tables-v1.npz`
- `TMPDIR=/home/enric/tmp PYTHONPATH=src JAX_ENABLE_X64=true python - <<'PY' ... run_tier1_sw() ...`
- `jq '.sw.sfluxzen, .sw.setcoef' artifacts/m5/rrtmg_intermediate_validation.json`
- `jq '.pass, .per_field_max_abs_err' artifacts/m5/tier1_rrtmg_sw_parity.json`
- `jq '[.sw_bands[] | select(.intermediate_gate=="PASS" and .implementation_status=="FULL_BRANCH_ACCEPTED")] | length' artifacts/m5/rrtmg_per_band_status.json`
- `rg -n -- '-r4|-i4|RWORDSIZE' /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/compile.log`
- `nl -ba /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F | sed -n ...` for the WRF citations above.
- Local Fortran KISS probe in `/home/enric/tmp` to compare WRF KISS random-overlap mask behavior against the JAX seed/carry logic.

I did not rerun the full `python scripts/m5_generate_rrtmg_fixture.py`, `python scripts/m5_run_rrtmg.py`, `python scripts/m5_gate_rrtmg.py`, and full `pytest -q tests/test_m5_rrtmg_*.py` sequence after the partial finding. The correctness gate was already false and the worktree also hit a full `/tmp`; subsequent targeted runs used `TMPDIR=/home/enric/tmp`.

## Proof Objects Produced

- `artifacts/m5/rrtmg_intermediate_validation.json`: SW `sfluxzen` PASS and SW `setcoef` PASS under the amended precision policy.
- `artifacts/m5/rrtmg_per_band_status.json`: 14 SW bands still marked `FULL_BRANCH_ACCEPTED` with `intermediate_gate=PASS`.
- `artifacts/m5/tier1_rrtmg_sw_parity.json`: explicit negative proof for strict SW Tier-1 parity, with remaining broadband flux residuals listed above.
- `data/fixtures/rrtmg-tables-v1.npz` and `.json`: regenerated table bundle containing the cloud coefficient additions used by the downstream experiment.

## Unresolved Risks

- SW Tier-1 flux parity is not closed. Heating-rate error is within threshold, but broadband flux residuals are far outside the `1 W/m2` absolute threshold.
- Launch-count and HLO-size targets are not claimed because the production path is not parity-correct yet. No `min(raw, cap)` launch fudge was introduced or used as evidence.
- The harness `nm` symbol preservation check was not rerun in this partial closeout. The intermediate oracle files remain present, but the verifiability triple is incomplete until the next oracle sprint reruns it.
- LW is protected by edit scope, not by a fresh full-run proof in this closeout.

## Next Decision Needed

Open a follow-up sprint before ADR-009 parity amendment. Recommended scope: build WRF intermediate-oracle dumps for `cldprmc_sw` and `spcvmc_sw`, then compare JAX at the cloud-optics and broadband-transfer boundary before more production edits. Required dump variables should include MCICA cloud masks and paths, `ptaucmc`, `pasycmc`, `pomgcmc`, `ptaormc`, clear/cloud two-stream layer properties, direct-beam transmittance, and per-g-point flux arrays before broadband accumulation. That is the shortest path to distinguish cloud optical-property assembly from vertical transfer and accumulation errors.
