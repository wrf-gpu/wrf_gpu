# V0.20 S4 1 km full-physics fp32 capability demo

Date: 2026-06-21

## Verdict

The completed proof establishes the fp32 capacity crossover for the v0.20 S4
full-physics path, but not a 1-hour 1 km run.

- Largest completed same-grid crossover: `700 x 700 x 45`, `dx=dy=1000 m`,
  `490,000` columns / `22.05M` 3-D cells.
- Same grid fp64 result: OOM on the first full-physics radiation step.
- Same grid fp32-aggressive result: fits one full-physics radiation step, all
  checked fields finite.
- Largest completed all-finite multi-step fp32 run: `650 x 650 x 45`, `dx=dy=1000 m`,
  `422,500` columns / `19.01M` 3-D cells, 2 steps, 2 RRTMG calls, `dt=1 s`.
- Full `1000 x 1000` 1 km full-physics grid remains over the capacity wall even
  with fp32-aggressive, hard RRTMG tile columns, and LW cloud-optics fp32.

This is a capacity proof for fp32 full physics at 1 km, not a validated real-WRF
science run. The grid/state are synthetic homogeneous proof inputs driven through
the production operational runtime with full physics enabled.

## Runtime Configuration

Common runtime:

- GPU lock: `scripts/with_gpu_lock.sh`, CPU pin `taskset -c 0-3`
- JAX allocator: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
- Full physics enabled:
  - Thompson microphysics: `mp_physics=8`
  - MYNN PBL: `bl_pbl_physics=5`
  - surface layer: `sf_sfclay_physics=5`
  - RRTMG shortwave/longwave: `ra_sw_physics=4`, `ra_lw_physics=4`
  - no cumulus: `cu_physics=0`
- RRTMG tiling: `GPUWRF_RRTMG_LW_COLUMN_TILE_COLS=64`,
  `GPUWRF_RRTMG_SW_COLUMN_TILE_COLS=64`
- fp32-aggressive mode:
  `GPUWRF_FORCE_FP64=0`,
  `GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020`
- LW cloud optics fp32 opt-in for final fp32 arms:
  `GPUWRF_RRTMG_LW_CLOUD_OPTICS_FP32=1`

## Headline Numbers

| Case | Grid | Steps / dt | Result | Peak sampled VRAM | Wall | Stability |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| fp64 crossover negative | `700x700x45` | 1 / 6 s | OOM, `13.66 GiB` allocation | `11,898 MiB` | `82.1 s` | n/a |
| fp32-aggressive crossover positive | `700x700x45` | 1 / 6 s | fits | `31,789 MiB` | `719.5 s` | all finite |
| fp32-aggressive stability positive | `650x650x45` | 2 / 1 s | fits | `27,984 MiB` | `1049.0 s` | all finite |
| fp64 at stability grid | `650x650x45` | 1 / 1 s | fits | `29,471 MiB` | `588.7 s` | all finite |
| fp32-aggressive long dt wall | `650x650x45` | 2 / 6 s | fits memory | `28,074 MiB` | `940.3 s` | non-finite |
| fp32-aggressive multi-step capacity wall | `700x700x45` | 2 / 1 s | OOM, `13.25 GiB` allocation | `29,598 MiB` | `667.8 s` | n/a |

Extension factor in this proof sequence:

- Largest verified fp64 fitting full-physics grid: `650x650x45`.
- Largest verified fp32 fitting grid where same-grid fp64 OOMs: `700x700x45`.
- Horizontal/3-D cell extension: `(700*700)/(650*650) = 1.16x`.

## 1M-column Wall

The requested full `1000 x 1000` horizontal grid does not fit on the RTX 5090 in
the current full-physics graph.

| Case | Grid | RRTMG tile cols | LW cloud fp32 | Result | Failing allocation | Peak sampled VRAM |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| fp64 | `1000x1000x50` | 256 | no | OOM | `381.47 MiB` after high memory pressure | `20,465 MiB` |
| fp32-aggressive | `1000x1000x50` | 256 | no | OOM | `30.47 GiB` | `15,653 MiB` |
| fp32-aggressive | `1000x1000x50` | 64 | no | OOM | `30.47 GiB` | `15,634 MiB` |
| fp32-aggressive | `1000x1000x45` | 64 | no | OOM | `27.17 GiB` | `14,495 MiB` |
| fp32-aggressive | `1000x1000x45` | 64 | yes | OOM | `27.17 GiB` | `14,493 MiB` |

Hard tiling from 256 to 64 columns did not move the `1000x1000x50` failure
allocation. The isolated completed LW-cloud-fp32 A/B at `1000x1000x45` also did
not move the hard allocation wall: `27.17 GiB` before and after, with only a
`2 MiB` difference in sampled peak VRAM. The final fitting fp32 arms used the
LW-cloud-fp32 opt-in, but the quantified isolated push at the 1M-column wall is
zero additional grid cells.

## Stability Verdict

- `700x700x45`, fp32-aggressive, `dt=6 s`, 1 radiation step: all finite.
- `650x650x45`, fp32-aggressive, `dt=1 s`, 2 radiation steps: all finite.
- `650x650x45`, fp32-aggressive, `dt=6 s`, 2 radiation steps: memory fits but
  dynamics go non-finite.
- `650x650x45`, fp32-aggressive, `dt=6 s`, 10 and 60 step runs also finish
  without OOM but end non-finite.
- `700x700x45`, fp32-aggressive, `dt=1 s`, 2 radiation steps OOMs on the second
  step.

Interpretation: the capacity crossover is real, but stable multi-step operation
at these near-limit grids needs the next phase of fp32 integration and speed/memory
A/Bs. The current best all-finite multi-step proof is `650x650x45`, `dt=1 s`,
2 RRTMG calls.

## Proof Objects

Artifact directory:
`proofs/v020/s4/1km_demo_20260621T200404Z/`

Key JSON/metric pairs:

- `fp64_700x700x45_step1_tile64.json`
- `fp64_700x700x45_step1_tile64.metric.json`
- `fp32_aggressive_700x700x45_step1_tile64_lwcloudfp32.json`
- `fp32_aggressive_700x700x45_step1_tile64_lwcloudfp32.metric.json`
- `fp32_aggressive_650x650x45_steps2_dt1_cadence1_segment1_tile64_lwcloudfp32.json`
- `fp32_aggressive_650x650x45_steps2_dt1_cadence1_segment1_tile64_lwcloudfp32.metric.json`
- `fp64_650x650x45_step1_dt1_tile64.json`
- `fp64_650x650x45_step1_dt1_tile64.metric.json`
- `fp32_aggressive_1000x1000x45_step1_tile64.json`
- `fp32_aggressive_1000x1000x45_step1_tile64_lwcloudfp32.json`
- `fp32_aggressive_1000x1000x50_step1_tile64.json`
- `fp32_aggressive_1000x1000x50_step1_tile256.json`
- `fp64_1000x1000x50_step1_tile256.json`

Each completed GPU run also has `.log` and `.vram.csv` files in the same
directory.

## Code / Harness Changes

- `src/gpuwrf/physics/rrtmg_lw.py`: added opt-in
  `GPUWRF_RRTMG_LW_CLOUD_OPTICS_FP32=1` path for LW cloud-optics transient arrays.
  Default behavior remains unchanged.
- `proofs/v020/s4/s4_1km_full_physics_demo.py`: proof-only synthetic 1 km
  full-physics harness using production operational runtime.
- `proofs/v020/s4/run_1km_demo_case.sh`: locked, pinned, reproducible runner with
  JSON, log, VRAM CSV, and metric output.

## Commands

Representative commands were run through:

```bash
scripts/with_gpu_lock.sh --label <case> -- \
  env GPUWRF_1KM_RUNNER_INNER=1 proofs/v020/s4/run_1km_demo_case.sh ...
```

Syntax checks:

```bash
bash -n proofs/v020/s4/run_1km_demo_case.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m py_compile \
  src/gpuwrf/physics/rrtmg_lw.py \
  proofs/v020/s4/s4_1km_full_physics_demo.py
```

## Unresolved Risks

- Synthetic homogeneous proof state, not a real WRF input fixture.
- No 1-hour stable 1 km run was achieved.
- `jax.memory_stats()["peak_bytes_in_use"]` is unavailable with the platform
  allocator on this setup, so peak VRAM comes from 1 Hz `nvidia-smi` sampling.
- The 1M-column wall is still the full-physics radiation/transient allocation
  wall; reducing it needs the next phase of targeted speed/memory A/Bs.
