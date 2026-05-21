# M5-S3.x Worker Report - RRTMG Transfer Solver Rewrite

Date: 2026-05-21
Branch: `worker/codex/m5-s3x-rrtmg-transfer-solver`
Worker: Codex GPT-5.5

## Objective

Replace the M5-S3 groundwork RRTMG shortwave and longwave transfer approximations with a real WRF-aligned transfer solver path, remove the fabricated SW gas optical-depth curve, preserve strict validation, and produce artifacts suitable for the mandatory second-AI reviewer pass.

## Verdict

This pass does not meet M5-S3.x acceptance. It is useful groundwork but remains a blocker for M6 coupled validation.

The implementation removes the explicit fabricated SW curve (`log1p(gas_coeff)` and `vapor_path`) and adds WRF-shaped transfer-solver pieces: SW delta scaling, Eddington layer reflectance/transmittance, WRF-style vertical quadrature, LW g-point recurrence, WRF molecular column construction, original SW cloud asymmetry extraction, and honest HLO/launch accounting. However, the implementation still does not port the full `setcoef` + per-band `taumol` interpolation or the full LW Planck-fraction/source machinery. Strict Tier-1 SW and LW parity both fail. Launch budget also fails honestly: the profile records 40 raw HLO launch markers and the gate reports `kernel_launches_per_step == raw_hlo_launch_marker_count == 40`.

This report is intentionally not an acceptance closeout.

## Files Changed

- `src/gpuwrf/physics/rrtmg_sw.py`
  - Added pressure-interface reconstruction, WRF-style molecular column estimates, Joseph-Wiscombe-Weinman delta scaling, Eddington two-stream layer coefficients, and WRF `vrtqdr_sw`-style adding/quadrature.
  - Removed the old fabricated `tau_gas = vapor_path * 0.01 * log1p(gas_coeff)` path. The new optical-depth path is still approximate and is not full `taumol_sw`.
- `src/gpuwrf/physics/rrtmg_lw.py`
  - Added pressure-interface reconstruction, WRF-style molecular columns, LW diffusivity secants, g-point transmittance, and upward/downward source recurrences.
  - This is still a reduced correlated-k transfer path, not full RRTMG `rtrnmc` + `taumol` parity.
- `src/gpuwrf/physics/rrtmg_constants.py`
  - Added molecular constants, default gas volume-mixing ratios, water-vapor molecular-weight ratio, and LW diffusivity coefficient arrays.
- `src/gpuwrf/physics/rrtmg_tables.py`
  - Added SW cloud liquid/ice asymmetry arrays to the table bundle.
- `scripts/extract_rrtmg_tables.py`
  - Extracts original SW cloud extinction, single-scattering albedo, and asymmetry separately instead of only pre-delta-scaled coefficients.
- `data/fixtures/rrtmg-tables-v1.npz`
  - Regenerated with SW cloud asymmetry arrays.
- `data/fixtures/rrtmg-tables-v1.json`
  - Regenerated manifest for the RRTMG tables bundle.
- `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml`
- `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml`
  - Regenerated fixture metadata after harness/table reruns.
- `artifacts/m5/*rrtmg*`
  - Regenerated strict Tier-1/Tier-2/profile/gate/HLO artifacts. These artifacts show strict fallback, not pass.
- `tests/test_m5_rrtmg_transfer_solver.py`
  - Added focused tests for the new delta-scaling helper, Eddington transparent-layer identity, removal of the fabricated SW gas curve, and LW diffusivity bounds.
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`
  - Amended to document the formulas, WRF source mapping, current partial implementation, and remaining blockers.

## Source Mapping and Formula Evidence

WRF source inspected:

- SW spectral solver: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:8196-8774` (`spcvmc_sw`).
- SW reference/transmittance solver: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:2530-2810` (`reftra_sw`).
- SW two-stream option comments and branch: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:2563`, `2632`, `2648-2656`. Important mismatch: the sprint asks for Eddington, but this local WRF source sets `kmodts=2`, which selects PIFM in the compiled WRF branch. The implementation follows the requested Eddington equations, while the oracle remains WRF-driver behavior.
- SW vertical quadrature: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:8035-8159` (`vrtqdr_sw`).
- SW delta scaling / optical-property combination: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:8603-8623`, `8638-8668`.
- SW coefficient interpolation: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:2843-3099` (`setcoef_sw`).
- SW gas optical depths: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:3190-4653` (`taumol_sw`, bands 16-29).
- SW cloud optical properties: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:2077-2498` (`cldprmc_sw`).
- LW transfer solver: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:3085-3522` (`rtrnmc`).
- LW coefficient interpolation: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:3556-3921` (`setcoef`).
- LW gas optical depths: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:4824-7942` (`taumol`).
- LW cloud optical properties: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:2764-3025` (`cldprmc`).

Implementation mapping:

- Joseph-Wiscombe-Weinman delta scaling is implemented in `src/gpuwrf/physics/rrtmg_sw.py:192-201`: `tau' = (1 - f omega) tau`, `omega' = (1 - f) omega / (1 - f omega)`, `g' = (g - f) / (1 - f)`, with `f = g**2`. Citation: Joseph, Wiscombe, and Weinman (1976), "The Delta-Eddington Approximation for Radiative Flux Transfer", Journal of the Atmospheric Sciences 33, 2452-2459.
- Eddington two-stream coefficients are implemented in `src/gpuwrf/physics/rrtmg_sw.py:204-274`: `gamma1 = (7 - omega(4 + 3g))/4`, `gamma2 = -(1 - omega(4 - 3g))/4`, `gamma3 = (2 - 3 mu0 g)/4`, `gamma4 = 1 - gamma3`. Citation: Meador and Weaver (1980), "Two-stream approximations to radiative transfer in planetary atmospheres", Journal of the Atmospheric Sciences 37, 630-643.
- WRF-style adding/quadrature is implemented in `src/gpuwrf/physics/rrtmg_sw.py:277-346` and mapped to WRF `vrtqdr_sw`.
- WRF-style molecular column construction is implemented in `src/gpuwrf/physics/rrtmg_sw.py:174-189` and `src/gpuwrf/physics/rrtmg_lw.py:169-186`. This is only molecular-column construction plus an approximate absorber mixture; it is not full WRF `setcoef`/`taumol`.
- LW g-point recurrence is implemented in `src/gpuwrf/physics/rrtmg_lw.py:200-221` and used in `src/gpuwrf/physics/rrtmg_lw.py:224-286`. Citation for correlated-k target behavior: Mlawer et al. (1997), "Radiative transfer for inhomogeneous atmospheres: RRTM, a validated correlated-k model for the longwave", Journal of Geophysical Research 102, 16663-16682.
- Original SW cloud asymmetry extraction is implemented in `scripts/extract_rrtmg_tables.py:403-434`, with fixed liquid effective radius 10 um and ice effective radius 30 um matching the existing harness defaults.

## Acceptance Criteria Status

AC1 - Real SW Eddington two-stream + delta scaling: partial, not pass.

Implemented delta scaling, Eddington coefficient algebra, layer reflectance/transmittance, WRF-style vertical quadrature, surface-albedo boundary, TOA solar source, and 14-band/g-point weighted flux summation. Strict SW parity still fails. The likely reasons are incomplete `setcoef_sw` + `taumol_sw`, approximate absorber mixing, and the Eddington-vs-WRF-PIFM oracle mismatch noted above.

AC2 - Real LW correlated-k integration: partial, not pass.

Implemented LW g-point weighting, molecular-column construction, diffusivity angles, surface emissivity, TOA cold-sky downward boundary, and upward/downward recurrences. This is not full `rtrnmc` parity because Planck-fraction interpolation and full band-by-band `taumol`/source coupling are still missing.

AC3 - Real gas absorption: partial, not pass.

The fabricated SW `log1p` saturation curve was removed and a test guards against its return. However, the current code still uses reduced absorber mixtures (`src/gpuwrf/physics/rrtmg_sw.py:188`, `src/gpuwrf/physics/rrtmg_lw.py:182`) and nearest reference-pressure coefficients. It does not yet port the per-band `setcoef` interpolation and `taumol` gas branches for H2O, CO2, O3, CH4, N2O, O2, CFC11, and CFC12.

AC4 - Cloud-radiation coupling: partial, not pass.

The table extractor now preserves original SW cloud asymmetry and the JAX solvers use cloud extinction, SSA, and asymmetry. Missing or fixed inputs:

- Liquid effective radius fixed at 10 um.
- Ice effective radius fixed at 30 um.
- Snow/graupel handling remains folded into simple path partitions.
- Cloud overlap/McICA remains deterministic and does not reproduce maximum-random or exponential-random overlap from WRF.

AC5 - Strict Tier-1 pass: fail.

Strict tolerances were not raised. The failed residuals are in `artifacts/m5/tier1_rrtmg_sw_parity.json` and `artifacts/m5/tier1_rrtmg_lw_parity.json`.

SW residuals:

| Field | Max abs error | Max rel error | Pass |
| --- | ---: | ---: | --- |
| heating_rate | 2.9023857620628224e-05 K/s | 1.1989202897643898 | true by absolute tolerance |
| flux_down | 107.68936518613896 W/m2 | 1.0002684001338669 | false |
| flux_up | 59.54565780869302 W/m2 | 1.1278234650078665 | false |
| toa_down | 67.04383583984372 W/m2 | 0.07765583457175178 | false |
| toa_up | 33.06389892320226 W/m2 | 0.11123142682827993 | false |
| surface_down | 58.95457983775114 W/m2 | 0.9863707057022061 | false |
| surface_up | 14.572475222160797 W/m2 | 0.9863707849382953 | false |
| column_absorbed | 111.53624306037793 W/m2 | 0.48076951453666855 | false |
| surface_absorbed | 51.880029494281544 W/m2 | 0.9863706768890842 | false |

LW residuals:

| Field | Max abs error | Max rel error | Pass |
| --- | ---: | ---: | --- |
| heating_rate | 6.148058425830156e-05 K/s | 19.939872022891567 | true by absolute tolerance |
| flux_down | 75.56380595798305 W/m2 | 0.7026938651515625 | false |
| flux_up | 45.5067191111325 W/m2 | 0.19808165842029532 | false |
| toa_down | 0.0 W/m2 | 0.0 | true |
| toa_up | 45.5067191111325 W/m2 | 0.19808165842029532 | false |
| surface_down | 9.851630433311186 W/m2 | 0.027442292070385416 | true by relative tolerance |
| surface_up | 0.5965461592547854 W/m2 | 0.0015365671203741102 | true |
| column_net_heating | 88.2496334872834 W/m2 | 0.6743828713267073 | false |
| surface_emission | 3.059611549360852e-05 W/m2 | 7.076241582851272e-08 | true |

The requested per-band residual table was not produced. The current Python validation artifacts summarize broadband fields and do not emit per-band flux residuals from the WRF harness. This is a deliverable miss and should be fixed before re-review.

AC6 - HLO + launch budget: fail on launches, pass on HLO size.

`artifacts/m5/rrtmg_gate_result.json` reports:

- `hlo_production_bytes_sw = 497598`, under the 500 KB limit but with little margin.
- `hlo_production_bytes_lw = 136941`, under the 500 KB limit.
- `kernel_launches_per_step = 40`.
- `raw_hlo_launch_marker_count = 40`.
- `gate_status = "FALLBACK"`.
- `rationale = "correctness failed"`.

No `min(raw, cap)` launch fudge is present. The launch count is honest and fails the <=10 budget.

AC7 - ADR-009 amended: done.

ADR-009 now documents the implemented formulas, WRF source lines, paper citations, HLO/launch evidence, and the remaining non-acceptance blockers.

## Commands Run

Read/inspection commands:

- `grep -n "subroutine spcvmc" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F`
- `grep -n "subroutine rtrnmc" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F`
- `grep -n "subroutine taumol" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F`
- `grep -n "subroutine setcoef" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F`
- `grep -n "subroutine cldprmc" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F`
- `grep -n "kmodts" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F`

Validation/build commands:

- `bash scripts/wrf_rrtmg_harness_build.sh` - passed.
- `python scripts/extract_rrtmg_tables.py --output data/fixtures/rrtmg-tables-v1.npz` - passed; regenerated table bundle with SW cloud asymmetry arrays.
- `python scripts/m5_generate_rrtmg_fixture.py` - passed.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/m5_run_rrtmg.py` - exited nonzero because strict Tier-1 failed; artifacts were produced.
- `python scripts/m5_gate_rrtmg.py` - exited nonzero; gate artifact reports strict fallback with raw launch count 40.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml` - passed.
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml` - passed.
- `python scripts/validate_agentos.py` - passed.
- `PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -q tests/test_m5_rrtmg_transfer_solver.py tests/test_m5_rrtmg_column_shapes.py tests/test_m5_rrtmg_tables.py tests/test_m5_rrtmg_tier2.py` - passed, 9 tests.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -q` - failed: 373 passed, 24 skipped, 27 failed.
- `git diff --check` - passed.

Full pytest failure categories:

- Sprint-relevant expected failures: `tests/test_m5_rrtmg_tier1.py` SW/LW strict parity false, and `tests/test_m5_rrtmg_gate.py` gate now reports `FALLBACK` instead of older gray-zone expectation.
- Environment/external artifact failures: missing Canary fixture payload/checksums, missing Thompson harness binary, M2 JAX/Triton venv/artifact failures. The M2 JAX install failed during pytest with `No space left on device`; JAX/Torch/Triton imports are absent in those scratch venvs.

## Proof Objects Produced

- `artifacts/m5/tier1_rrtmg_sw_parity.json` - strict SW parity record; `pass=false`.
- `artifacts/m5/tier1_rrtmg_lw_parity.json` - strict LW parity record; `pass=false`.
- `artifacts/m5/tier2_rrtmg_invariants.json` - Tier-2 invariant record; `pass=true`.
- `artifacts/m5/rrtmg_profile.json` - HLO size, launch marker count, zero post-init host/device transfer counters.
- `artifacts/m5/rrtmg_gate_result.json` - strict gate result; `gate_status=FALLBACK`, `kernel_launches_per_step=40`, `raw_hlo_launch_marker_count=40`.
- `artifacts/m5/hlo_dump/rrtmg_sw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_lw_production.txt`
- `data/fixtures/rrtmg-tables-v1.npz`
- `data/fixtures/rrtmg-tables-v1.json`
- `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml`
- `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml`

## Unresolved Risks and Blockers

1. Full SW `setcoef_sw` + `taumol_sw` is not ported. The present reduced absorber mixture is the main physics gap and should be treated as non-production.
2. Full LW `setcoef` + `taumol` + Planck fraction interpolation is not ported. The LW recurrence is structurally closer to RRTMG than the prior hand-rolled transfer but still not full correlated-k parity.
3. WRF local source selects PIFM (`kmodts=2`) while the sprint contract requires Eddington. The next worker/manager should decide whether Tier-1 should target compiled WRF PIFM exactly or a modified Eddington oracle. Without that decision, Eddington parity against the current harness is not a clean target.
4. Per-band residual artifacts are absent. The WRF harness or Python validation needs to emit per-band fluxes at TOA/surface before a reviewer can validate the per-band AC.
5. Cloud overlap and McICA are still deterministic approximations. Effective radii are fixed to harness defaults rather than full WRF inflow dimensions.
6. Launch budget fails: 40 raw HLO launch markers against a <=10 target. Correctness should be fixed first, then band/g-point scans need further fusion.
7. SW HLO size is barely under budget at 497,598 bytes, leaving little headroom for a full `taumol` port without additional structure changes.

## Next Decision Needed

Manager/reviewer should not close M5-S3.x as accepted. The next practical decision is whether to:

1. Scope the next sprint to exact WRF parity by targeting the compiled local WRF PIFM branch and full `setcoef`/`taumol` first, then revisit Eddington as an ADR-level change; or
2. Rebuild/patch the WRF oracle to force `kmodts=1` Eddington before continuing this Eddington-specific implementation.

Either path also needs a per-band fixture/harness output extension before another acceptance review.
