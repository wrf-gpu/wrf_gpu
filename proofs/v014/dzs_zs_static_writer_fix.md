# v0.14 DZS/ZS static soil-geometry writer fix

## Problem

The GPU `wrfout` carried the `soil_layers_stag=4` dimension but emitted **no**
`ZS` (depths of soil-layer centers) or `DZS` (soil-layer thicknesses). The CPU
`wrfout` carries both, shape `(1,4)`. The Grid-Delta Atlas hard-gate marked both
null/unpaired -> `FAIL_TOLERANCE`. These are static Noah/Noah-MP soil-geometry
constants (WRF `init_soil_depth_2`, `module_soil_pre.F:1128-1151`).

## Root cause

The wrfout writer (`src/gpuwrf/io/wrfout_writer.py`) never registered `ZS`/`DZS`:
absent from `WRFOUT_VARIABLE_SPECS`, absent from the ordered write list
`OPERATIONAL_WRFOUT_VARIABLES`, and never populated in the field builder. The
authoritative WRF-faithful values existed only in the real-init lane
(`init_soil_depth_noahmp`) and were never routed to output.

## Fix (`src/gpuwrf/io/wrfout_writer.py`)

1. Added `ZS`, `DZS` to `GRID_COORDINATE_VARIABLES` (the static grid-geometry
   group, alongside `ZNU`/`ZNW`).
2. Added `WRFOUT_VARIABLE_SPECS["ZS"]` / `["DZS"]` using the existing `SOIL_1D`
   dims `("Time","soil_layers_stag")`, with the exact CPU attrs:
   `description`/`units="m"`/`stagger="Z"`/`MemoryOrder="Z  "`/`FieldType=104`,
   no `coordinates` attr (matches the reference wrfout for these 1-D fields).
3. Populated `fields["ZS"]`/`["DZS"]` in `_add_grid_coordinate_fields` from the
   soil config (`soil_layers_stag`). DZS = `[0.1,0.3,0.6,1.0]`; ZS computed by
   the WRF cumulative-center formula. Computed in `real(4)` (fp32) to bit-match
   the Fortran `init_soil_depth_2`: the fp32 cumulative sum rounds
   `ZS(3)=0.70000005` (`0x3f333334`), whereas an fp64 accumulation cast to fp32
   rounds the other way to `0x3f333333` (1-ULP low).

## Proof (real 1-h GPU forecast vs CPU)

GPU run: `gpuwrf.cli run --domain d01 --hours 1` (worktree `src`, GPU lock).
Compared `wrfout_d01_2023-01-15_01:00:00` GPU vs CPU:

| field | GPU present | dims | values | RAW-BIT-EXACT vs CPU | attrs match |
| --- | --- | --- | --- | --- | --- |
| ZS  | yes | (Time, soil_layers_stag) | [0.05, 0.25, 0.70000005, 1.5] | yes | yes |
| DZS | yes | (Time, soil_layers_stag) | [0.1, 0.3, 0.6, 1.0]          | yes | yes |

Grid comparator (`scripts/compare_wrfout_grid.py --vars ZS DZS`): both now PAIR,
classified `static`, RMSE=0 / bias=0 / max_abs=0 / p99=0. See
`dzs_zs_static_parity_compare.{json,md}`.

Writer + restart test suite: 43 passed, 2 skipped.
