# v0.22 Feature Push Integration Verdict

Date: 2026-06-27

## Integration Branch

- Base: `7f219f94 release(v0.22): package base and corrected canary gate`
- Integration branch: `integration/v022-feature-push-gpt`
- Worktree: `/tmp/claude-1000/-home-user-src-wrf-gpu2/5a8a44d1-6d87-4194-9587-0d0c3c635b20/scratchpad/wt_v022_integration_gpt`
- Push/tag: not performed

## Cherry-Picked Feature Heads

| Piece | Source | Integrated as |
|---|---:|---:|
| G0 two-way feedback nesting | `ca02901f` | `b5424002` |
| G2 375-variable output stream | `dcf4899f` | `bf567230` |
| F1 3-D TKE / Smagorinsky | `9cb82c46` | `286cf723` |
| F2 missing schemes | `e2c5ac68` | `5da496b5` |
| F3 CAM-UW PBL | `b750aff4` | `7961a793` |
| G1 data assimilation | `ffa34a23` | `4431af6a` |
| G2 moving nests / adaptive dt | `76cde1fd` | `39ae124d` |
| G3 urban/lake | `cb8a59bc` | `062e2d71` |
| E evaluation harness | `45f97f27` | `f7997681` |

Note: the handoff said G0 and G2 output were already on the base branch, but ancestry check showed both were absent from `7f219f94`; both were cherry-picked explicitly.

## Merge / Default Audit

- Textual conflicts: none.
- Auto-merged shared files: `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/integration/daily_pipeline.py`, `src/gpuwrf/integration/nested_pipeline.py`, `src/gpuwrf/io/scheme_catalog.py`, `src/gpuwrf/io/namelist_check.py`.
- Default audit:
  - `OperationalNamelist.diff_opt=0`, `km_opt=0`, `bl_pbl_physics=5`, `sf_urban_physics=0`, `sf_lake_physics=0`, `data_assimilation=None`.
  - F1 TKE defaults: `c_k=0.15`, `mix_isotropic=0`, `mix_upper_bound=0.1`, `tke_upper_bound=1000.0`, `tke_mix2_off=False`.
  - `DailyPipelineConfig.auxhist=None`, `auxhist_streams=()`, `full_wrfout_variables=False`.
  - `NestedPipelineConfig.feedback=False`.

## Gate Verdict Table

| Feature | Compile / focused tests | Finite / bounded run | Oracle / skill | Verdict | Gate numbers / notes |
|---|---:|---:|---:|---|---|
| G0 two-way feedback nesting | PASS, `1 passed` | PASS | PASS | LANDED | conservation rel residual `0.0`; parent overlap error vs copy_fcn+sm121 `0.0`; feedback events `2`; E-harness PASS |
| G2 375-variable output stream | PASS, `21 passed, 1 skipped` | PASS | PASS | LANDED | full-wrfout stays opt-in; auxhist/main stream tests preserve default output; E-harness PASS |
| F1 3-D TKE / Smagorinsky | PASS, `75 passed` | PASS | PASS | LANDED | GPU oracle on `cuda:0`; km_opt=3 WRF formula max abs `0.0`; km_opt=2/5 QKE finite and bounded; known scaffold: exact deformation-stress/moist BN2 parity |
| F2 missing schemes | PASS, `4 passed` | N/A fail-closed | PASS fail-closed | SCAFFOLD-needs-iteration | `gate_pass=true`, `full_bundle_landed=false`; New Tiedtke/NSSL/Morrison-aero/RUC are recognized/reference/fail-closed, not operational ports; E-harness PASS |
| F3 CAM-UW PBL | PASS, `186 passed` | PASS | PARTIAL | SCAFFOLD-needs-iteration | GPU idealized/source-present gate PASS; no pristine-WRF numerical CAM-UW savepoint parity (`parity_claim=false`); E-harness PASS |
| G1 data assimilation | PASS, `2 passed` | PASS | PASS | LANDED | theta RMSE `10.0 -> 8.0`, qv RMSE `0.001 -> 0.0008`, DFI finite true; deferred raw obs ingest/full backward+forward DFI choreography; E-harness PASS |
| G2 moving nests / adaptive dt / global | PASS, `24 passed` | PASS | PARTIAL | SCAFFOLD-needs-iteration | moved start `[5,4]`; dt history `[5.5,6.6]`; global is metadata wrap only, polar/global runtime deferred; E-harness PASS |
| G3 urban/lake | PASS, `73 passed` | N/A fail-closed | PASS fail-closed | SCAFFOLD-needs-iteration | `gate_pass=true`, `full_physics_landed=false`; BEP/BEM/lake source recognized and fail-closed, no faithful physics run; E-harness PASS |
| E evaluation harness | PASS, `3 passed` | PASS | PASS | LANDED | smallest Canary synthetic operational-relaxed PASS at leads `24,72,120`; per-feature E-harness rollups PASS `9/9` |

## Commands Run

- `git cherry-pick -x ca02901f dcf4899f 9cb82c46 e2c5ac68 b750aff4 ffa34a23 76cde1fd cb8a59bc 45f97f27`
- `PYTHONPATH=src python -m compileall -q src/gpuwrf scripts ...`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-g0 -- pytest -q tests/test_v022_two_way_feedback_gate.py && python scripts/v022_two_way_feedback_gate.py ...`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-g2-output -- pytest -q tests/test_auxhist_stream.py tests/test_m7_netcdf_writer.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-f1 -- python proofs/v022/feature_push/f1_3d_tke_smag_oracle.py && pytest -q tests/dynamics/test_diffopt2_tke_smag3d.py tests/test_namelist_check.py tests/test_scheme_catalog_fail_closed.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-f2 -- python proofs/v022/f2_missing_scheme_bundle_oracle_check.py && pytest -q tests/test_v022_f2_missing_scheme_bundle.py tests/test_tiedtke_cumulus_oracle.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-f3-rerun -- GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF python proofs/v022/feature_push/f3_camuw_pbl_oracle_check.py && pytest -q tests/test_v022_camuw_pbl.py tests/contracts/test_v060_physics_interfaces.py tests/test_v013_operational_smoke.py tests/test_v013_sfclay_pbl_pairing.py tests/test_namelist_check.py tests/test_scheme_catalog_fail_closed.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-g1 -- python scripts/v022_data_assimilation_gate.py ... && pytest -q tests/test_v022_data_assimilation.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-g2nest -- python scripts/v022_moving_nest_gate.py ... && pytest -q tests/test_v022_moving_nest_adaptive.py tests/test_v0110_domain_tree.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-g3 -- python proofs/v022/g3city_urban_lake_gate.py ... && pytest -q tests/test_v022_g3city_urban_lake.py tests/test_namelist_check.py tests/test_scheme_catalog_fail_closed.py`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-eval -- pytest -q tests/test_v022_operational_relaxed_gate.py && scripts/v022_eval_smallest_canary_example.sh`
- `scripts/with_gpu_lock.sh --label gpt-v022-integration-eharness-by-feature -- python scripts/v022_operational_relaxed_gate.py ...` for all 9 feature lever IDs
- `git diff --check`

## Proof Objects

- `proofs/v022/feature_push/integration_gates/G0_TWO_WAY_FEEDBACK_GATE.json`
- `proofs/v022/feature_push/integration_gates/G1_DATA_ASSIMILATION_GATE.json`
- `proofs/v022/feature_push/integration_gates/G2_MOVING_NEST_ADAPTIVE_GATE.json`
- `proofs/v022/feature_push/integration_gates/G3_CITY_URBAN_LAKE_GATE.json`
- `proofs/v022/feature_push/integration_gates/E_EVAL_SMALLEST_CANARY/rollup.json`
- `proofs/v022/feature_push/integration_gates/eharness_by_feature/*/rollup.json`
- Refreshed integrated-branch proofs: `proofs/v022/feature_push/f1_3d_tke_smag_oracle.json`, `proofs/v022/feature_push/f3_camuw_pbl_oracle.json`, `proofs/v022/f2_missing_scheme_bundle_oracle_check.json`

## Environmental Note

The first broad F3 smoke attempt failed because the isolated worktree lacks `data/wrf_pristine/WRF`; the radiation smoke tests looked there. Rerunning with `GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF` passed `186 passed`. No code change was made for that environment issue.
