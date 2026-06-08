# V014 Switzerland/Gotthard Validation Prep

Date: 2026-06-08
Worker: GPT-5.5 xhigh validation-prep sidecar
Scope: prepare only; no GPU run; no `src/` edits.

## Objective

Prepare the first post-grid/memory Switzerland/Gotthard GPU-vs-CPU-WRF validation
candidate, using the current on-disk case assets and current v0.14 priorities.

## Bottom Line

Recommendation: **NO-GO for an immediate equivalence run while grid-cell
divergence remains the active priority. CONDITIONAL GO after the grid-envelope
gate/root-cause decision and the RRTMG column-tiling memory lineage are accepted.**

If the manager wants a narrow **memory-fit smoke** after the RRTMG column-tiling
fix, run only the `128x128` case and label it as fit/run evidence until the
field comparator produces complete hard-field coverage. Do not use it to soften
the current Canary grid-divergence finding.

The run is worth preparing because Switzerland is non-Canary, winter/Alps, and
has retained CPU-WRF truth. It should remain secondary to v0.14 B4 grid parity:
current evidence has V10 grid RMSE above `1.5 m/s` in 3/3 durable Canary cases,
while station TOST can look better than the full grid.

## Current Data Availability

All paths below were inspected read-only on 2026-06-08.

| Root | Grid | CPU truth | GPU output now | Inputs | Timing / notes |
| --- | --- | --- | --- | --- | --- |
| `/mnt/data/wrf_gpu_switzerland_128` | `e_we=e_sn=129`, mass grid `128x128`, `e_vert=45`, `bottom_top=44`, `dx=dy=3000 m`, `time_step=18 s` | 25 hourly `wrfout_d01_*`, `2023-01-15_00:00:00` through `2023-01-16_00:00:00`; CPU run completed clean | one partial `run_gpu/wrfout_d01_2023-01-15_01:00:00`; `gpu_run.json` verdict `PIPELINE_BLOCKED`; `gpu_run.stderr` has RRTMG OOM | `run_cpu` has `wrfinput_d01`, `wrfbdy_d01`, `namelist.input`; `run_gpu_input` is a clean symlink dir with only those three files; WPS has 9 `met_em` files and GFS has 9 forcing links | CPU-WRF `28` rank dmpar gfortran; total wall `1056.9501616954803 s`; mainloop `1038.7748700000022 s`; final-frame finite checks passed |
| `/mnt/data/wrf_gpu_switzerland_big` | `e_we=e_sn=151`, mass grid `150x150`, `e_vert=45`, `bottom_top=44`, `dx=dy=3000 m`, `time_step=18 s` | 25 hourly `wrfout_d01_*`, same 24 h period; CPU run completed clean | no retained GPU `wrfout`; `run_gpu/proofs` contains generic proof stubs only | `run_cpu` has `wrfinput_d01`, `wrfbdy_d01`, `namelist.input`; WPS has 9 `met_em` files and GFS has 9 forcing links | CPU-WRF `28` rank dmpar gfortran; total wall `1483.3700413703918 s`; mainloop `1464.6753199999994 s`; final-frame finite checks passed |

Disk headroom under `/mnt/data` was `707G` available when inspected.

The `128` CPU truth has all ten frozen-tolerance comparator fields:
`T2`, `U10`, `V10`, `PSFC`, `RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`.
The old partial GPU file has only the first five of those ten, so any future
verdict must require `n_fields_compared == 10`; otherwise the comparator can
under-cover the documented hard fields.

## Why The Previous Attempt OOMed

`proofs/v0120/switzerland_128_gpu_result.json` characterizes the prior attempt
as an fp64 single-GPU grid ceiling, not a correctness failure.

- `150x150` status: OOM; reported dominant allocation was an RRTMG g-point
  radiation temporary, `25.58 GiB` as a single allocation.
- `128x128` status: OOM; reported peak VRAM `31193 MiB`; error included
  `CUDA_ERROR_OUT_OF_MEMORY`; dominant family was
  `f64[45,128,128,16]` RRTMG g-point arrays.
- The retained `128` stderr additionally shows failures to allocate small
  autotune requests after the arena was full and a final failed request of
  `12.09 GiB`.

Conclusion: the old run hit the full-column RRTMG LW/SW transient ceiling on a
32 GB RTX 5090 with desktop overhead. It did not reach a 24 h forecast or a
scientific comparison.

## RRTMG Column-Tiling Feasibility

RRTMG leading-column tiling is now present in the current tree:

- `src/gpuwrf/physics/rrtmg_sw.py` and `src/gpuwrf/physics/rrtmg_lw.py` default
  to column tiling enabled with `GPUWRF_RRTMG_*_COLUMN_TILE_COLS=16384`.
- CPU inertness proof: `proofs/v013/rrtmg_column_tile.json`, bit-identical for
  SW/LW all-sky and clear-sky/topography covered cases.
- GPU VRAM proof: `proofs/v013/rrtmg_column_tile_vram_suite.json`.
  At `ncol=65536`, untiled LW OOMed on `32.11 GiB`; tiled LW peaked at
  `5374.84 MiB`; tiled SW peaked at `1619.54 MiB`.

This likely changes Switzerland feasibility materially:

- The `128x128` case has `ncol=16384`, exactly the default tile size. The
  isolated RRTMG peak should be bounded by the proven tile-scale path rather
  than the old full-domain fused transient.
- The `150x150` case has `ncol=22500`, so it should run as two column tiles.
  It is plausible after `128x128` passes, but it should not be the first retry.
- The full forecast has not yet been measured post-tiling. Expected peak for
  `128x128` is low-to-mid teens GiB, not the prior `31.2 GiB` arena ceiling,
  but the acceptance artifact must record actual peak/log evidence if available.

## Command Recommendation

Use a durable validation root and a clean native-init input directory. Do **not**
point `--input-dir` at `run_cpu`: the CLI auto-detects CPU-WRF replay when it
sees two or more `wrfout_d01_*` files. The intended Switzerland experiment is
standalone native-init from `wrfinput_d01` plus `wrfbdy_d01`, then compare to
CPU-WRF.

Preferred command is the repo wrapper around the Switzerland script:

```bash
set -euo pipefail
OUT=/mnt/data/wrf_gpu_validation/v014_switzerland_gotthard_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"

scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  CASE_ROOT="$OUT" \
  CASE_INPUTS=/mnt/data/wrf_gpu_switzerland_128/run_cpu \
  CPU_REF=/mnt/data/wrf_gpu_switzerland_128/run_cpu \
  GPU_INPUT="$OUT/input" \
  GPU_OUT="$OUT/gpu" \
  SCRATCH="$OUT/scratch" \
  PROOF="$OUT/switzerland_equivalence.json" \
  CPU_WALL_BASIS=total \
  CPU_RANKS=28 \
  CPU_BUILD_LABEL="dmpar MPI gfortran, 28 ranks (HONEST denominator, NOT 1-core)" \
  GPUWRF_RRTMG_SW_COLUMN_TILING=true \
  GPUWRF_RRTMG_LW_COLUMN_TILING=true \
  GPUWRF_RRTMG_SW_COLUMN_TILE_COLS=16384 \
  GPUWRF_RRTMG_LW_COLUMN_TILE_COLS=16384 \
  PYTHON=/home/enric/miniconda3/bin/python \
  bash scripts/equivalence_switzerland.sh \
  | tee "$OUT/equivalence_switzerland.stdout.log"
```

Post-run guard before accepting the comparator result:

```bash
python - <<'PY' "$OUT/switzerland_equivalence.json"
import json, sys
p = json.load(open(sys.argv[1]))
c = p["comparison"]
fields = {f["field"]: f for f in c["fields"]}
missing = [name for name, row in fields.items() if row["verdict"] == "NO_DATA"]
if len(c["compared_files"]) != 24:
    raise SystemExit(f"expected 24 GPU/CPU timestamp pairs, got {len(c['compared_files'])}")
if c["n_fields_compared"] != 10 or missing:
    raise SystemExit(f"incomplete hard-field coverage: n={c['n_fields_compared']} missing={missing}")
if c["overall_verdict"] not in ("EQUIVALENT", "NOT_EQUIVALENT"):
    raise SystemExit(f"invalid scientific verdict: {c['overall_verdict']}")
print("SWITZERLAND_COMPARE_COVERAGE_OK", c["overall_verdict"], c["exceeding_fields"])
PY
```

## Fields And Gates

Hard frozen-tolerance gate from `docs/equivalence-switzerland.md` /
`scripts/equivalence_demo.py`:

- `T2 <= 1.5 K` pooled RMSE
- `U10 <= 1.5 m s-1` pooled RMSE
- `V10 <= 1.5 m s-1` pooled RMSE
- `PSFC <= 120 Pa` pooled RMSE
- `RAINNC <= 1.0 mm` pooled RMSE
- `T <= 1.5 K` pooled RMSE
- `U <= 1.8 m s-1` pooled RMSE
- `V <= 1.8 m s-1` pooled RMSE
- `W <= 0.30 m s-1` pooled RMSE
- `QVAPOR <= 1.0e-3 kg kg-1` pooled RMSE

Required run gates:

- GPU emits 24 hourly `wrfout_d01_*` frames under `$OUT/gpu`, paired by timestamp
  to CPU truth at hours 1-24.
- All hard fields above are present in both GPU and CPU outputs; `NO_DATA` is a
  blocker, not a pass.
- All compared arrays are finite; shape mismatches are blockers.
- `EQUIVALENT` and `NOT_EQUIVALENT` are both valid scientific outcomes once data
  coverage is complete. `NO_DATA`, missing pairs, OOM, or partial output are
  failures of the validation run.
- Report the result as non-Canary support only. It does not replace the v0.14
  all-comparable-field Canary grid-envelope gate.

Report-only inventory for a stronger v0.14 artifact should include every common
GPU/CPU `wrfout` variable with compatible dimensions, including surface/radiation
fields such as `Q2`, `RAINC`, `RAINSH`, `SWDOWN`, `GLW`, `PBLH`, `UST`, `HFX`,
`LH`, `TSK`, `CLDFRA`, `QCLOUD`, `QICE`, `QRAIN`, and current writer 3D/static
fields (`P`, `PB`, `PH`, `PHB`, `MU`, `MUB`, coordinates/statics) where emitted.
Fields without frozen tolerances are report-only.

## Must Be True Before Running

- The manager has either fixed/accepted the current grid-divergence root cause,
  or explicitly wants a memory-fit Switzerland smoke that will not be marketed
  as equivalence.
- The run branch includes RRTMG column tiling and the proof artifacts above.
- No active GPU validation owns `/tmp/wrf_gpu_validation_gpu.lock`; use
  `scripts/run_gpu_lowprio.sh`.
- The run uses a clean native-init input dir (`$OUT/input`), not `run_cpu`.
- Existing partial `/mnt/data/wrf_gpu_switzerland_128/run_gpu` output is not
  reused for scoring.
- `$OUT` is under `/mnt/data/wrf_gpu_validation/...` and contains logs, scratch,
  generated GPU `wrfout`, and comparator JSON.
- The comparator result is accepted only if the post-run hard-field coverage
  guard passes.

## Files Changed

- `.agent/reviews/2026-06-08-v014-switzerland-validation-prep.md`
- `proofs/v014/switzerland_validation_plan.md`

## Commands Run

Read-only inspections only; no GPU commands were run.

- `sed -n` on repo instructions, current V014 validation plan, local validation
  skills, Switzerland docs, GPU runbook, and wrapper script.
- `find`, `du`, `df`, `rg`, `ncdump`, and NetCDF metadata inspections under
  `/mnt/data/wrf_gpu_switzerland_128` and `/mnt/data/wrf_gpu_switzerland_big`.
- `python -m json.tool` / JSON reads for prior Switzerland result, RRTMG tiling,
  V10 diagnostics, and memory proof artifacts.
- `sed -n` / `rg` inspections of CLI init-mode selection and wrfout writer field
  coverage; no `src/` changes.

## Proof Objects Produced

- `proofs/v014/switzerland_validation_plan.md`

This is a plan/proof-prep object, not evidence of a completed GPU run.

## Unresolved Risks

- Full 24 h Switzerland post-column-tiling VRAM has not been measured.
- Existing comparator can under-cover hard fields unless the post-run guard is
  applied.
- Current grid-cell divergence means a `NOT_EQUIVALENT` Switzerland outcome is
  likely and scientifically valid; it should not trigger tolerance tuning.
- The V014 B7 draft command uses `run_cpu` as `--input-dir`; that would take
  CPU-WRF replay mode under current CLI detection and should be corrected before
  execution.

## Next Decision Needed

Decide whether Switzerland runs only after grid-parity remediation, or whether
to authorize a clearly labeled memory-fit smoke of the `128x128` case first.
