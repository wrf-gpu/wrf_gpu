# V0.14 Canary 72h Field-Gate Drift Root Cause: wrfbdy LBC Cadence Bug

Verdict: `LBC_CADENCE_ROOT_CAUSE_PROVEN_FIX_GATE_PASS`. CPU-only. Companion
JSON: `lbc_cadence_root_cause.json`. Script: `lbc_cadence_root_cause.py`.

## Root cause

The standalone native-init root domain (d01) decodes one `*_bdy` leaf time
level per wrfbdy record (6-hourly, `interval_seconds=21600`), but
`interpolate_boundary_leaf` walked the leaf time axis at the hourly replay
default `boundary_config.update_cadence_s = 3600`. The d01 lateral boundary
therefore consumed the 72h forcing **6x too fast**: at hour `h` the spec zone
was forced with the wrfbdy value valid at `6*h` hours, and from h11 onward it
clamped **frozen** at the last record (t=66h) for the remaining 61 hours.

Evidence (Canary `20260501_18z_l2_72h_20260519T173026Z`, west outermost-column
mean MU, Pa):

- live GPU run h1..h10 == wrfbdy levels 1..10 exactly (`1867.0, 1663.8,
  1822.0, 1613.6, 1715.8, 1424.9, 1491.2, 1313.5, 1421.5, 1244.7`; max abs
  diff `0.0`); CPU-WRF truth == the correct linear ramp (`1718.3` at h1, ...).
- live GPU run h11..h20 frozen at `1390.5` == wrfbdy level 11 (t=66h), while
  CPU truth keeps ramping (`1697.7 ... 1752.5`).
- proof script bit-reproduces the broken run: old cadence emulation vs GPU
  emitted spec zone max abs `0.0e+00` Pa over h1..h20.
- domain-mean dry-mass mirror: CPU d01 mean MU h0->h6 `+181.9` Pa (smooth
  LBC-driven +-30 Pa/h, sign flip exactly at the 00:00 segment switch); GPU
  `-176.3` Pa with erratic per-hour tendencies tracking wrfbdy level-to-level
  jumps. d02 inherits the identical drift from the live d01 parent
  (dPSFC/dMU per lead match d01 within ~5%).

This is the driver of the v0.14 Canary h08/h10/h18 `FAIL` drift signal
(`PSFC` slope 73.6 Pa/h over h1-h8, bias sign consistency 1.0; `MU/P/PH`
tracking), and the same path/bug has been live since the v0.12.0 standalone
native-init (the TOST case-3 `PSFC` RMSE 525 Pa / KI-9 h10-14 wind-mass
divergence window is consistent with it). Switzerland uses
`interval_seconds=10800` on the same path -> 3x-fast forcing if launched
unfixed.

## Fix (2 files, narrow)

1. `src/gpuwrf/integration/nested_pipeline.py`: new
   `_root_boundary_cadence_override` applied to the ROOT domain only -- sets
   `boundary_config.update_cadence_s` to the loader's
   `boundary_meta["interval_seconds"]`. Children keep `update_cadence_s ==
   parent_dt` (live-nest cadence, unchanged). Replay paths (hourly wrfout
   history leaves, no `interval_seconds` meta) keep 3600.
2. `src/gpuwrf/integration/d02_replay.py::load_wrfbdy_boundary_leaves`: one
   synthesized terminal leaf level = last wrfbdy record advanced by its own
   `_BT*` tendency over the full interval, so leads 66-72h interpolate instead
   of clamping frozen; leaf time meta now reports `times=13`,
   `wrfbdy_records=12`, `terminal_level_synthesized=true`.

WRF-faithfulness: WRF forces `field_bdy + dtbc*field_bdy_tend` with the `_BT*`
tendency spanning `bdyfrq == interval_seconds`; linear interpolation between
consecutive record values at that cadence is the identical forcing (the
records are exact segment endpoints). CPU-WRF truth spec zone matches the
wrfbdy ramp to ~0.05 Pa, and the fixed interpolation matches CPU truth to
`0.000` Pa max over h1..h20 and `9.4e-06` Pa at h72.

## Proof gates (all PASS, `rc=0`)

| Gate | Result |
|---|---|
| bug reproduction vs live GPU spec zone h1..h20 | max abs `0.0` Pa |
| fixed cadence vs CPU-WRF truth spec zone h1..h20 | max abs `0.000` Pa |
| synthesized terminal level == base+tendency*interval | abs diff `0.0` Pa |
| h72 interpolated vs CPU truth | `9.4e-06` Pa |
| plumbing `update_cadence_s` | `3600 -> 21600` |
| `pytest test_m6_boundary_apply test_v013_tost_wrfbdy_fix` | 5 passed, 1 skipped |
| `pytest test_p0_1a_nesting test_gwd_operational_wiring` | 18 passed |

## Not fixed here (separate, pre-existing lanes)

- GPU near-surface full pressure is vapor-light: `PSFC - (p_top+MU+MUB)` is
  `~+210..220` Pa on CPU (== vapor column weight) vs `~0 +- tens` Pa on GPU at
  every lead; both runs' PSFC are exactly consistent with their own written
  `P/PH` extrapolation, so this is the runtime pressure state, not the writer.
  Quasi-static ~-210 Pa PSFC floor under the (now removed) growing drift; it
  is most of the h1 PSFC bias (-117 = +85 cadence-bug mass excess - ~202
  vapor gap).
- `PB/MUB` static 5-cell nest-frame spikes (known, boundary-frame only).
- Bounded MYNN/RRTMG step-1 residuals, radiation timing ~-20 min class.
