# M5-S3.y Worker Report - RRTMG setcoef/taumol/Planck Attempt

Date: 2026-05-21
Branch: `worker/codex/m5-s3y-rrtmg-setcoef-taumol-planck`
Worker: Codex GPT-5.5

## Objective

Implement the M5-S3.y contract items needed to unblock M6 RRTMG validation: first force the local WRF SW oracle to Eddington (`kmodts=1`), then move the JAX RRTMG path from reduced reference-pressure gas curves toward native WRF `setcoef_sw`/`taumol_sw` and LW Planck-source machinery, preserve strict tolerances, and produce honest proof objects.

## Verdict

This pass is **NOT ACCEPTANCE** and should not be merged as M6-unblocking RRTMG parity.

AC0 landed: the local WRF SW oracle was patched to Eddington and the harness was rebuilt. The rebuilt binary still links the real WRF RRTMG entry points.

The code also exposes native reduced SW `absa/absb/selfref/forref/sfluxref/Rayleigh` tables and LW `totplnk/totplk16` Planck tables as JAX table leaves, and the JAX kernels consume those leaves. That is real groundwork, not a synthetic table. However, strict Tier-1 still fails for SW and LW, the per-band WRF harness extension was not completed, and the SW HLO/launch profile regressed over budget (`1.31 MB`, `52` combined launches). I am filing this as a failed attempt with useful evidence, not as completed work.

## Files Changed

- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F`
  - Patched `kmodts=2` to `kmodts=1` at the `reftra_sw` branch cited by the contract (`module_ra_rrtmg_sw.F:2632`).
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/phys/module_ra_rrtmg_sw.F`
  - Same patch. This was required because the existing CMake build rules compile the `/home/enric/.../wrf_src` path even though the harness script names the `/mnt/data/.../wrf_gpu_src` tree.
- `scripts/extract_rrtmg_tables.py`
  - Added source parsing for WRF `tref(:)` and LW `totplnk/totplk16` assignments.
  - Added reduced-g extraction for native SW `absa`, `absb`, `selfref`, `forref`, `sfluxref`, Rayleigh and special absorber terms. Reduction follows the WRF `swcmbdat`/`cmbgb*` pattern: weighted k/continuum/Rayleigh reductions and unweighted solar-source reductions (`module_ra_rrtmg_sw.F:4763-4784`, `5135-5226`, `5733-5795`, `5801-5900`, `5957-6035`).
- `src/gpuwrf/physics/rrtmg_tables.py`
  - Extended `RRTMGTableBundle` so the new native SW and LW Planck tables are JAX leaves, not closed-over JIT constants.
- `src/gpuwrf/physics/rrtmg_sw.py`
  - Added a JAX `_SWSetCoefState` and a vectorized port of WRF `setcoef_sw` pressure/temperature/continuum interpolation factors (`module_ra_rrtmg_sw.F:2843-3099`).
  - Added a first-pass JAX `taumol_sw` branch implementation for all 14 SW bands using the extracted `absa/absb/selfref/forref` tables and WRF branch formulas (`module_ra_rrtmg_sw.F:3190-4653`).
  - Added WRF solar source-function interpolation over reduced `sfluxref` tables. This did not fix the third-scenario TOA-down residual, which remains a clue that wrapper/indexing/cloud or source-layer semantics still diverge.
- `src/gpuwrf/physics/rrtmg_lw.py`
  - Added LW interface-temperature reconstruction and WRF `totplnk` interpolation for `planklay`, `planklev`, and `plankbnd` (`module_ra_rrtmg_lw.F:3556-3921`).
  - Replaced the old grey `sigma*T^4*g_weight` source with a band-integrated Planck source scaled by `delwave*pi*1e4`, matching the WRF `rtrnmc` flux scaling shape (`module_ra_rrtmg_lw.F:3270-3340`, `3475-3496`).
- `data/fixtures/rrtmg-tables-v1.npz` and `.json`
  - Regenerated. New NPZ SHA: `9d8bedbfa93161b0d782d64a98c7e36d8f95e54b016d4fc570c0a7c5aa534013`; bytes: `4199742`.
- `fixtures/manifests/analytic-rrtmg-{sw,lw}-column-v1.yaml`, `fixtures/samples/analytic-rrtmg-sw-column-v1.npz`, `data/fixtures/analytic-rrtmg-sw-column-v1/full.npz`
  - Regenerated after Eddington oracle rebuild. LW sample/full did not materially change.
- `artifacts/m5/*rrtmg*`
  - Regenerated strict artifacts. They record failure.
- `artifacts/m5/tier1_rrtmg_per_band.json`
  - Added explicit failure artifact: per-band WRF harness output was not completed.
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`
  - Updated status to M5-S3.y still NOT PARITY with current evidence. I did not falsely set status to `PARITY`.

## AC Status

AC0 - Eddington oracle rebuild: **PASS**

- `/mnt/.../module_ra_rrtmg_sw.F` SHA before patch: `7f8af1da0ca1d25ce784a917bc68600300a7c569881c57e7de8501cd53496b59`.
- `/mnt/.../module_ra_rrtmg_sw.F` SHA after patch: `f6da816cd8ffa89e73397a2fe005e5e8a07716aa176eb0b5004f6a60efeda4e0`.
- Build-rule source `/home/enric/.../module_ra_rrtmg_sw.F` after patch: same SHA `f6da816cd8ffa89e73397a2fe005e5e8a07716aa176eb0b5004f6a60efeda4e0`.
- Rebuilt SW object SHA: `d3c13e0059d2db9bcde4432a21a2dfa575a8bfc7001f2924f91e3941fa17000b`.
- Rebuilt harness SHA: `25c88aa4f79e49533aa55dc557183e43052ce0a043da9d1ee4aa31e134fd2b33`.
- `nm data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_|cldprmc_" | head -n 20` shows the real WRF symbols, including `__rrtmg_sw_spcvmc_MOD_spcvmc_sw`, `__rrtmg_lw_rtrnmc_MOD_rtrnmc`, `__rrtmg_sw_taumol_MOD_taumol_sw`, `__rrtmg_lw_taumol_MOD_taumol`, `__rrtmg_sw_setcoef_MOD_setcoef_sw`, and `__rrtmg_lw_setcoef_MOD_setcoef`.

AC1 - SW `setcoef_sw` port: **PARTIAL**

The JAX `_sw_setcoef` implements `jp/jt/jt1`, `fac00/fac01/fac10/fac11`, `indself/indfor`, `selffac/forfac`, and molecular columns using WRF formulas from `module_ra_rrtmg_sw.F:2843-3099`. The state is JAX-resident. This has not been independently validated against a WRF intermediate-output fixture, so it is not a proof of exact parity.

AC2 - SW `taumol_sw` per-band port: **PARTIAL / FAILING VALIDATION**

All 14 band branches are represented in JAX and consume extracted native reduced-g tables. The branch formulas cite `module_ra_rrtmg_sw.F:3190-4653`. However, strict SW Tier-1 remains false and HLO/launch cost regressed. This is not acceptable as final `taumol_sw` parity.

AC3 - LW `setcoef` port: **PARTIAL**

Only the Planck part of LW `setcoef` is ported (`planklay`, `planklev`, `plankbnd`) from `module_ra_rrtmg_lw.F:3556-3921`. LW gas `jp/jt` ratio state and `minorfrac/scaleminor` for the full 16-band `taumol` path are not complete.

AC4 - LW `taumol` per-band + Planck fractions: **NOT DONE**

The old nearest-pressure LW optical-depth approximation remains. I did not complete full LW `taugb*` branches or `fracs(lev,igc)` interpolation from `module_ra_rrtmg_lw.F:4824-7942`.

AC5 - LW Planck-source machinery in `rtrnmc`: **PARTIAL**

The source now uses WRF integrated Planck bands and WRF flux scaling (`module_ra_rrtmg_lw.F:3270-3340`, `3475-3496`) instead of grey `sigma*T^4` global weights. It still does not implement the full WRF non-isothermal `dplankup/dplankdn` and `tfn_tbl` source correction inside the recurrence. LW residual improved modestly but remains far outside strict flux tolerance.

AC6 - Per-band fixture/harness extension: **NOT DONE**

The WRF harness still emits broadband TOA/surface and profile fluxes only. I added `artifacts/m5/tier1_rrtmg_per_band.json` with `produced=false` rather than fabricating per-band residuals.

AC7 - Launch fusion 40 -> <=10: **FAIL**

The SW native per-band branch implementation increased compiled complexity. Latest profile:

- `kernel_launches_per_step = raw_hlo_launch_marker_count = 52`
- `raw_hlo_launch_marker_count_sw = 36`
- `raw_hlo_launch_marker_count_lw = 16`
- `hlo_production_bytes_sw = 1312209`
- `hlo_production_bytes_lw = 154560`

This violates both the `<=10` launch target and the `<=500 KB` SW HLO budget. There is no launch fudge: the raw count is reported directly.

AC8 - Strict Tier-1 pass + ADR-009 finalize: **FAIL**

Strict tolerances remain unchanged (`abs <= 1 W/m2 + rel <= 0.05` for fluxes; `abs <= 1e-4 K/s + rel <= 0.05` for heating). The gate is still FALLBACK. ADR-009 was not finalized to `PARITY`; it was updated honestly to record M5-S3.y non-parity.

## Residual Table

Broadband SW residuals after this attempt:

| Field | Max abs error | Max rel error | Pass |
| --- | ---: | ---: | --- |
| heating_rate | `3.632664522070731e-05 K/s` | `1.2790599782112937` | true by abs |
| flux_down | `135.97104293374935 W/m2` | `1.0273788764621958` | false |
| flux_up | `79.21669171335645 W/m2` | `1.2128467380949062` | false |
| toa_down | `67.04542671926288 W/m2` | `0.07765767726269346` | false |
| toa_up | `23.579080632559453 W/m2` | `0.08020510057802743` | false |
| surface_down | `59.476898467889825 W/m2` | `0.9965700014200232` | false |
| surface_up | `15.039959431639133 W/m2` | `0.9965700014200233` | false |
| column_absorbed | `79.73584900803431 W/m2` | `0.3924687672148566` | false |
| surface_absorbed | `52.33967065174305 W/m2` | `0.9965700014200232` | false |

Broadband LW residuals after this attempt:

| Field | Max abs error | Max rel error | Pass |
| --- | ---: | ---: | --- |
| heating_rate | `6.091121542523097e-05 K/s` | `18.38772630807074` | true by abs |
| flux_down | `67.25107985479801 W/m2` | `0.6590888151425406` | false |
| flux_up | `44.33055076399509 W/m2` | `0.1929620325423348` | false |
| toa_down | `0.0 W/m2` | `0.0` | true |
| toa_up | `44.33055076399509 W/m2` | `0.1929620325423348` | false |
| surface_down | `10.743642163317077 W/m2` | `0.029927042852576902` | true by rel |
| surface_up | `0.643868833120905 W/m2` | `0.001658459556663873` | true |
| column_net_heating | `73.67198882864889 W/m2` | `0.5710347664979502` | false |
| surface_emission | `3.059611549360852e-05 W/m2` | `7.076241582851272e-08` | true |

Per-band residual table: **not produced**. This is an explicit deliverable miss.

## Verifiability Triple

1. `nm` symbol check: **PASS**. Real WRF RRTMG symbols remain linked after the harness rebuild.
2. Non-clipped coefficient ratio: **PASS for new native SW tables**. Inspection of `sw_absa`, `sw_absb`, `sw_selfref`, and `sw_forref` shows broad real distributions (`sw_absa` nonzero min `2.6169e-13`, max `1.263e7`; `sw_absb` nonzero min `3.772e-13`, max `67577`). Old A2 floor hits are not pinned patterns. A quick floor count over new native SW tables found zero hits at `0.0025`, `1e-5`, `0.25`, `0.16`; one raw data equality each at `0.003` and `0.2`, not a clipped table plateau.
3. Non-vacuous tolerance: **PASS**. I did not loosen manifests. The gate remains strict and false.

## Commands Run

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
- `sed -n '1,240p' AGENTS.md`
- `sed -n '1,260p' .agent/rules/sprint-lifecycle.md`
- `sed -n '1,280p' .agent/sprints/2026-05-21-m5-s3y-rrtmg-setcoef-taumol-planck/sprint-contract.md`
- `sed -n '1,260p' .agent/sprints/2026-05-21-m5-s3x-rrtmg-transfer-solver/reviewer-report.md`
- `sed -n '1,260p' .agent/sprints/2026-05-21-m5-s3x-rrtmg-transfer-solver/worker-report.md`
- `sed -n '1,260p' .agent/sprints/2026-05-21-m5-s3x-rrtmg-transfer-solver/manager-closeout.md`
- `sed -n '1,260p' .agent/decisions/ADR-009-rrtmg-jax-implementation.md`
- `sed -n '1,260p' .agent/sprints/2026-05-21-m5-s3-rrtmg-radiation-column/reviewer-a3-report.md`
- `sed -n '1,260p' .agent/sprints/2026-05-21-m5-s3-rrtmg-radiation-column/reviewer-a2-report.md`
- `sed -n '1,260p' /home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md`
- `sed -n '1,240p' .agent/skills/building-wrf-oracles/SKILL.md`
- `sed -n '1,240p' .agent/skills/validating-physics/SKILL.md`
- `sed -n '1,240p' .agent/skills/writing-gpu-kernels/SKILL.md`
- `sed -n '1,240p' .agent/skills/designing-gpu-state/SKILL.md`
- `sed -n '2620,2642p' /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F`
- `sha256sum /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F`
- `make phys/module_ra_rrtmg_sw.F.o`
- `bash scripts/wrf_rrtmg_harness_build.sh`
- `nm data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_|cldprmc_" | head -n 20`
- `python scripts/extract_rrtmg_tables.py --output data/fixtures/rrtmg-tables-v1.npz`
- `python scripts/m5_generate_rrtmg_fixture.py`
- `python scripts/m5_run_rrtmg.py` - failed as expected, produced proof objects
- `python scripts/m5_gate_rrtmg.py` - failed, gate FALLBACK
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml` - passed
- `python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml` - passed
- `python scripts/validate_agentos.py` - passed
- `PYTHONPATH=src pytest -q tests/test_m5_rrtmg_*.py` - failed: 10 passed, 3 failed (`test_m5_rrtmg_gate.py`, SW Tier-1, LW Tier-1)

## Proof Objects Produced

- `data/scratch/wrf_rrtmg_harness` SHA `25c88aa4f79e49533aa55dc557183e43052ce0a043da9d1ee4aa31e134fd2b33`
- `data/fixtures/rrtmg-tables-v1.npz` SHA `9d8bedbfa93161b0d782d64a98c7e36d8f95e54b016d4fc570c0a7c5aa534013`
- `artifacts/m5/tier1_rrtmg_sw_parity.json` - `pass=false`
- `artifacts/m5/tier1_rrtmg_lw_parity.json` - `pass=false`
- `artifacts/m5/tier2_rrtmg_invariants.json` - `pass=true`
- `artifacts/m5/rrtmg_profile.json` - raw launch/HLO evidence, `52` launches
- `artifacts/m5/rrtmg_gate_result.json` - `gate_status=FALLBACK`
- `artifacts/m5/tier1_rrtmg_per_band.json` - explicit `produced=false` per-band failure record
- `artifacts/m5/hlo_dump/rrtmg_sw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_lw_production.txt`

## Unresolved Risks And Blockers

1. Full LW `taumol` remains unimplemented. The native LW gas and Planck-fraction branch set (`module_ra_rrtmg_lw.F:4824-7942`) is still the main missing code surface.
2. The LW `rtrnmc` source recurrence still lacks the exact `dplankup/dplankdn`, `tfn_tbl`, and cloudy-layer source machinery from `module_ra_rrtmg_lw.F:3270-3340`; the partial Planck band source improved LW column-net-heating from `88.25` to `73.67 W/m2` but did not close parity.
3. SW exact gas table consumption did not close SW parity and increased HLO to `1.31 MB`. This suggests the remaining SW residual is not solely the prior reference-pressure gas approximation. The likely next suspects are WRF wrapper/source-layer semantics, cloud/McICA treatment, and vertical indexing around the wrapper top level.
4. Per-band WRF output is not implemented. Without it, a reviewer cannot localize which bands remain wrong.
5. Launch fusion is worse after unrolled branch expansion. A second implementation should use table-driven `lax.scan` or generated compact branch tables before attempting review.

## Next Decision Needed

Do not send this to the Opus reviewer as an acceptance candidate. The manager should decide whether to:

1. Scope M5-S3.z to first add WRF per-band harness output and intermediate `taug/taur/fracs/plank*` dumps, then validate each JAX branch against those intermediate oracles before touching transfer fusion; or
2. Roll back the SW native branch expansion and dispatch a narrower LW Planck-source sprint, because this attempt proves a naive all-band JAX branch expansion violates the HLO/launch budget before correctness is solved.

My recommendation is option 1. The project needs intermediate WRF optical-depth/source proof objects before more hand-transcribed branch code is useful.
