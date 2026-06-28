# wrf_gpu v0.22.1

v0.22.1 is a pod-data-quality point release on top of v0.22.0. It fixes two
B200-pod-run output-path defects. It does not change numerics, masking, clamps,
NetCDF schema, or default wrfout naming conventions.

## Default Behavior

Default behavior remains bit-identical / convention-preserving relative to
v0.22.0 for this release scope:

- WRF-standard wrfout timestamps remain the default:
  `wrfout_d02_2024-08-06_13:00:00`.
- `GPUWRF_COLONFREE_OUTPUT` is opt-in only and defaults off.
- The d02 cadence fix changes only missed child history-alarm output points; it
  does not alter model state updates or numeric fields.

## Fixes

### Nested d02 output cadence

The nested scheduler now preserves leaf-child history alarms on both scheduler
paths. The eager leaf advance splits at history alarms, and the default fused
flat-leaf path fails closed to that eager split when a child `history_interval`
cadence is not parent-ratio aligned. The fused path remains active for the
divisible common case.

Before v0.22.1, a B200 two-domain run with `history_interval=20,20` wrote d01 at
20-minute cadence but collapsed d02 to hourly output. That crushed the intended
3 km to 1 km training-pair yield.

After the fix, d02 emits at exact 20-minute child steps. The real GPU NetCDF
smoke gates wrote d01 and d02 at matching 20-minute cadence over the release
window, and all inspected floating-point fields were finite.

The 9-nest canary diagnostic remains sound for its covered path: it exercises
the nested cascade path, where d02 is an internal parent. The two-domain B200
failure localized to the leaf-domain cadence split, not to the internal cascade
writer.

### Opt-in colon-free wrfout names

`GPUWRF_COLONFREE_OUTPUT=1` writes the wrfout filename time portion as
`HH-MM-SS`, for example:

```text
wrfout_d02_2024-08-06_13-00-00
```

This is output-path only and is intended for RunPod-S3 / network-volume drains
where colon-containing object keys are unsafe for the drain path. The default
remains WRF-standard `HH:MM:SS`, for example:

```text
wrfout_d02_2024-08-06_13:00:00
```

The shared wrfout filename parser accepts both default colon and opt-in dash
timestamps so downstream inventory/postprocessing can read either mode.

## Validation

- d02 cadence GPU NetCDF gates: d01 and d02 both wrote the expected three
  frames at 20-minute cadence over the smoke window; all inspected
  floating-point fields were finite.
- Domain-tree CPU gate: fused divisible-cadence subtrees stay fused and
  eager-identical; fused non-divisible leaf cadence falls back to eager and
  emits exact child frames without double-output.
- Colon-free naming CPU unit: default colon mode and
  `GPUWRF_COLONFREE_OUTPUT=1` dash mode both passed.
- Full CPU release gate is run apples-to-apples with
  `--continue-on-collection-errors` because this environment has pre-existing
  collection debt: a third-party `tests` package shadows the repo's non-package
  `tests/` tree for `tests.init...` imports under the mandated `PYTHONPATH=src`
  invocation. This is not a v0.22.1 runtime regression.
- Default bit-identity check: default wrfout timestamp convention remains
  `HH:MM:SS`; cadence fix is output scheduling only, with no numeric masks,
  clamps, or schema changes.

## Carried Forward

Tracked but non-blocking for v0.22.x: cross-region nested-AOT prewarm gate
(#145), B200 runbook tooling (#146), and the F2/F3/G2-moving/G3 feature
scaffolds.
