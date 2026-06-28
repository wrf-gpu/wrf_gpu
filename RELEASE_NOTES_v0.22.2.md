# wrf_gpu v0.22.2

v0.22.2 is a nested host-bound GPU-idle reduction point release on top of
v0.22.1. The default path keeps full wrfout semantics and CPU-proven
byte-identical output while cutting output-boundary host work. The higher-risk
overlap and radiation-source changes are opt-in and await the manager-scheduled
0:2 warm GPU / VRAM validation run.

## Headline

Nested wall-clock work moved from "less host work at every output boundary" to a
release-ready CPU-gated candidate:

- **Phase 1 host-work cuts are default-on and byte-identical.**
- **The per-output M9 RRTMG re-solve uses a scoped 512-column tile cap** so the
  output spike has a smaller VRAM transient without slowing main-forecast
  radiation.
- **`GPUWRF_NEST_OUTPUT_PIPELINE=1` is opt-in** for overlapping output
  materialization behind the next GPU segment.
- **`GPUWRF_NESTED_M9_RADIATION_FROM_CARRY=1` is opt-in** for skipping the
  nested Noah-MP per-output RRTMG diagnostic re-solve when carry-backed surface
  radiation is sufficient.
- **GPU idle / VRAM validation is pending 0:2.** No agent ran a GPU job while
  the 5090 was locked.

## Default-On Changes

- Removed the redundant nested output-boundary full-state `finite_summary`.
  The finite guard still runs; only the second host pull / min-max pass is gone.
- Batched wrfout payload device-to-host materialization into one
  `jax.device_get` over the prepared field map.
- Batched finite-guard checks into one device-side all-finite reduction over the
  floating leaves.
- Made training-subset output preparation subset-aware: with
  `GPUWRF_TRAINING_OUTPUT_SUBSET=1`, only requested variables are materialized
  before the host transfer.
- Added opt-in nested output-boundary timers with
  `GPUWRF_NEST_PERF_TIMERS=1`; default is off.
- Added a scoped 512-column cap for the M9 output diagnostic RRTMG re-solve.
  The shared LW/SW forecast default stays at 1024 columns; the M9 re-solve calls
  the same public tiled solvers with a per-call override, so this caps its
  radiation transient without changing the default output contract or the main
  forecast radiation tile count.
  `GPUWRF_RRTMG_COLUMN_TILE_COLS`, `GPUWRF_RRTMG_LW_COLUMN_TILE_COLS`, and
  `GPUWRF_RRTMG_SW_COLUMN_TILE_COLS` still override the non-M9/default cap.

These changes are intended to be numerics-free and full-output byte-identical.

## Opt-In Levers

### Structural Output Pipeline

`GPUWRF_NEST_OUTPUT_PIPELINE=1` enables the Phase-2/S1 structural output
pipeline. The step thread captures the post-step carry by immutable reference
and enqueues an output snapshot. A bounded materialize stage runs the finite
guard, M9 diagnostics, payload preparation, and handoff to the async NetCDF
writer while the next GPU segment can compute.

The flag is independent from the radiation-source flag. With
`GPUWRF_NEST_OUTPUT_PIPELINE=0` / unset, the legacy materialization path remains.

### Radiation From Carry

`GPUWRF_NESTED_M9_RADIATION_FROM_CARRY=1` enables the S2 nested Noah-MP output
shortcut. It avoids the per-output full RRTMG diagnostic re-solve and emits only
fields backed by the resident carry:

- `SWDOWN`, `GLW`, `SWNORM`, `LWDNB`;
- `SWDNB` only when `slope_rad != 1`;
- Noah-MP overlay `HFX`, `LH`, `TSK`, `T2`;
- normal surface-layer fields such as `U10`, `V10`, `Q2`, `PSFC`, and `PBLH`.

The default remains off because `carry.noahmp_rad` stores only held
`(SOLDN, LWDN, COSZ)` at WRF radiation-held time. It does not hold the
output-time `SWUPB`, `LWUPB`, `SWDNT`, `SWUPT`, `LWDNT`, `LWUPT`, or `OLR`
flux slices produced by the full diagnostic re-solve. The opt-in path omits
unavailable fields rather than fabricating them.

## Validation

CPU gates for the release candidate include:

- async wrfout default and byte-identity tests;
- nested wall-clock Phase 0/1 tests;
- Phase-2 output-pipeline byte-identity and fail-closed tests;
- training-subset env/default identity tests;
- root-sync orchestration identity;
- S2 nested Noah-MP radiation default/opt-in tests;
- explicit default wrfout `cmp` checks;
- `git diff --check`;
- `python -m compileall`.

GPU validation remains intentionally pending: the 5090 is locked by pane 0:2,
and the pre-release GPU gate is the next manager-scheduled run comparing v0.22.1
against v0.22.2 for GPU idle percentage, wall-clock seconds per forecast hour,
default full-output identity, and overlapped VRAM high-water behavior.
