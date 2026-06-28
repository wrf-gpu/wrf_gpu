# v0.22.1 d02 History Cadence Fix

## Outcome

Fixed the max_dom=2 nested-output cadence bug in the domain-tree scheduler. The d02 leaf domain now emits at its own exact child history boundaries for `history_interval=20`, instead of skipping to hourly when the 20-minute alarm falls inside a 3-child-step parent subcycle.

No numerics clamps, masking, or writer schema changes were made.

## Root Cause

Root cause: `src/gpuwrf/runtime/domain_tree.py:501-518`.

The leaf-domain branch advanced a child by the full parent-ratio chunk, then checked output only once after the chunk. In the B200 two-domain shape, d02 is a leaf and advances in 3-step chunks. With `dt(d02)=6 s` and `history_interval=20 min`, d02 output cadence is 200 child steps. The old scheduler crossed 198->201 and 399->402 without seeing exact modulo step 200/400, so d02 did not output until step 600, i.e. hourly.

This was not a `daily_pipeline.py` path. B200 ran the same live-nested `nested_pipeline.py -> run_operational_domain_tree -> run_domain_tree_callbacks` path as the canary gate; the difference was scheduler topology:

- B200 max_dom=2: d02 is a leaf, advanced in 3-step chunks, so non-divisible cadence 200 was skipped.
- Canary max_dom=9: d02 is an internal parent, advanced one d02 step at a time, so d02 saw steps 200/400/600. d03-d09 are leaves, but their 20-minute cadence is 600 child steps, divisible by the 3-step chunk.

So prior 9-nest validation did not cover the failing max_dom=2 d02 leaf case.

## Fix

In the leaf-domain branch, split only the advance chunk that crosses an output alarm:

- advance to the exact child cadence boundary,
- call the output callback at that exact `own_step`,
- continue the remaining child substeps.

Exact/default hourly paths remain unchanged when the cadence boundary is already aligned with the chunk.

## Evidence

B200 paid-run log before fix, max_dom=2, 2 h:

- Source: `<DATA_ROOT>/wrf_downscale/runpod_s3_backups/b200_v0211_20260627T152712Z_full_s3_20260627T200143Z/logs/b200_v0211_20260627T152712Z/alps.log`
- d01: `wrfout_count=5`, expected 5, files at `12:20:06`, `12:40:12`, `13:00:18`, `13:20:24`, `13:40:30`.
- d02: `wrfout_count=2`, expected 6, files at `13:00:00`, `14:00:00`.

Static scheduler repro before/after:

- Proof log: `proofs/v022/d02_cadence/scheduler_repro_before_after.log`
- 70.2 min max_dom=2 window before: d01 3, d02 1.
- 70.2 min max_dom=2 window after: d01 3, d02 3 at d02 steps 200/400/600.
- 2 h max_dom=2 projection after: d02 6 at exact 20-minute child steps 200/400/600/800/1000/1200. d01 remains 5 over exactly 400 root steps because existing root cadence uses `ceil(1200/18)=67`, so the next d01 frame would be root step 402; that root rounding behavior was not changed.

Canary diagnostic answer:

- Existing canary proof: `<DATA_ROOT>/wrf_downscale/runs/20260512/gpu_vram_fix/live_census_enhanced_nofuse_h1/proofs/nested_pipeline_run.json`
- max_dom=9, 1 h, `history_interval=20`.
- d02-d09 each sustained three child frames: `18:20:00`, `18:40:00`, `19:00:00`.
- d01 had two frames at `18:20:06`, `18:40:12`, again due root-step rounding.

Attempted real GPU smoke:

- Command used proof-local `history_interval=20` AC1FIT input and `hours=1.17`.
- First attempt failed before integration because `GPUWRF_WRF_ROOT` was unset.
- Second attempt set `GPUWRF_WRF_ROOT=<DATA_ROOT>/src/wrf_pristine/WRF` but hit an AOT miss and cold compile; it was terminated after >4 minutes to obey the wall-clock-smart rule. This aborted smoke is not used as proof.

## Validation Commands

```bash
PYTHONPATH=src pytest -q tests/test_v0110_domain_tree.py
PYTHONPATH=src pytest -q tests/test_v014_noahmp_nested_pipeline.py
```

Results:

- `tests/test_v0110_domain_tree.py`: 22 passed.
- `tests/test_v014_noahmp_nested_pipeline.py`: 31 passed.

Default-path bit-identity/default behavior coverage:

- `test_leaf_child_hourly_exact_cadence_is_unchanged` verifies the exact hourly leaf-child path is unchanged.
- `test_nested_async_output_byte_identical_to_sync` verifies the nested wrfout writer async/sync split remains byte-identical.

## Risks

Release-gate NetCDF smoke completed after cold compile and is recorded below. No remaining release-blocking cadence evidence gap is known for max_dom=2 d02 output cadence.

## Release-Gate GPU NetCDF Smoke

Run root: `<DATA_ROOT>/wrf_gpu_validation/v0221_d02_cadence_release_20260627T214844Z`

Command shape:

```bash
scripts/with_gpu_lock.sh --timeout 7200 --label d02-cadence-release-gate -- bash -lc '... execute_nested_pipeline(... hours=1.2, max_dom=2)'
```

Environment:

- `GPUWRF_WRF_ROOT=<DATA_ROOT>/src/wrf_pristine/WRF`
- proof-local input copied from AC1FIT with only `history_interval = 20, 20, 20`
- `GPUWRF_TRAINING_OUTPUT_SUBSET` unset
- `GPUWRF_FULL_WRFOUT_VARIABLES` unset

Result:

- Verdict: `PIPELINE_GREEN`
- `all_domains_finite=true`
- `all_outputs_present=true`
- Proof JSON: `proofs/v022/d02_cadence/release_gate_summary.json`
- NetCDF finite scan: `proofs/v022/d02_cadence/release_gate_netcdf_finite_scan.json`

Counts and cadence:

| Domain | dt | history_interval | count | expected | Times |
| --- | ---: | ---: | ---: | ---: | --- |
| d01 | 18 s | 20 min | 3 | 3 | `18:20:06`, `18:40:12`, `19:00:18` |
| d02 | 6 s | 20 min | 3 | 3 | `18:20:00`, `18:40:00`, `19:00:00` |

The d02 times correspond to exact child output steps 200, 400, and 600. All six NetCDF files opened successfully. The post-run scan checked 106 floating variables in each file and found all values finite.

Default numerics / output-path scope:

- The code change only splits leaf-domain host advance chunks at output alarms before invoking the existing output callback.
- No masking/clamps or physics/dynamics arithmetic changes were made.
- `test_leaf_child_hourly_exact_cadence_is_unchanged` keeps the exact hourly/default cadence path unchanged.
- `test_nested_async_output_byte_identical_to_sync` keeps the nested writer async/sync output bytes identical.

## Fused-Path Critic Closure

Critic blocker: the original leaf split fixed the eager recursion, but the default
fused flat-leaf subtree could still advance a leaf child by a full
`parent_grid_ratio` and call output only at the end of that fused parent step.

Chosen fix: routing, not fused-kernel modification. The scheduler now keeps fused
for leaf children whose output cadence is parent-ratio aligned, and routes a
fused parent/subtree through the already-fixed eager path when any leaf child's
output cadence is not aligned with the parent-ratio endpoint.

Root-cause/fix anchors:

- `src/gpuwrf/runtime/domain_tree.py:385`: `fused_leaf_outputs_are_parent_ratio_aligned(...)`.
- `src/gpuwrf/runtime/domain_tree.py:395`: non-divisible cadence or misaligned resumed child clock returns `False`.
- `src/gpuwrf/runtime/domain_tree.py:540`: fused cascade is requested only when leaf output alarms are ratio-aligned.

CPU proof:

```bash
PYTHONPATH=src pytest -q tests/test_v0110_domain_tree.py tests/test_v022_colonfree_output.py
```

Result:

- `26 passed in 3.79s`.
- `test_fused_leaf_divisible_history_cadence_keeps_fused_fast_path`: divisible common case stays fused and remains eager-identical.
- `test_fused_leaf_nondivisible_history_cadence_falls_back_to_eager_alarms`: a fused-parent configuration with a non-divisible leaf cadence does not request/call the fused program, emits child outputs at exact steps `5, 10, 15`, and has no double-output.

GPU proof composition:

- No fresh cold-compile fused GPU run is required for this closure because the
  fix does not modify the fused cascade kernel. It only routes non-divisible leaf
  cadence cases to the eager leaf split path.
- That eager path is already GPU NetCDF release-gated above: `PIPELINE_GREEN`,
  d02 `3/3` expected frames at exact 20-minute child steps `200/400/600`, all
  checked NetCDF floating fields finite.
- A fresh locked GPU gate was started after the critic note but stopped once the
  wall-clock-smart steer clarified that routing to the eager path plus the
  existing eager GPU gate is the intended proof. The aborted cold run is not used
  as evidence.

Default bit-identity / behavior:

- Divisible leaf-cadence subtrees still take the fused fast path.
- Non-divisible leaf-cadence subtrees use the same eager split path already
  validated by NetCDF smoke.
- No dynamics/physics arithmetic, masks, clamps, NetCDF schema, or default
  colon timestamp naming changed.
