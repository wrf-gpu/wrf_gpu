# v0.17 fp32 Physics Bench

## Verdict

`GPUWRF_FP32_PHYSICS=1` is implemented as an opt-in physics compute island for
RRTMG SW, RRTMG LW, and MYNN.  The default path remains off/fp64.

Status:

| Gate | Result |
| --- | --- |
| RRTMG SW fp32 oracle | PASS |
| RRTMG LW fp32 oracle | PASS |
| MYNN fp32 oracle | PASS |
| Coupled short-run sanity | PASS |
| WS-128 timing | PASS, measured 1.915x warm speedup |
| Canary 9/3/1 timing | RED, not measured here |

No Canary speedup is claimed.  The available Canary validation directory
`/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_allgreen/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
contains wrfout outputs but no `namelist.input`, so `build_replay_case()` cannot
reconstruct a timed operational case.

## Oracle Gates

Artifacts:

- `proofs/perf/v017/oracles/tier1_rrtmg_sw_fp32.json`
- `proofs/perf/v017/oracles/tier1_rrtmg_lw_fp32.json`
- `proofs/perf/v017/oracles/tier1_mynn_fp32.json`

All three use `compute_dtype=float32`, return fp32 scheme outputs, and meet the
existing fixture tolerances.

Selected max absolute errors:

| Scheme | Field | Max abs error |
| --- | ---: | ---: |
| RRTMG SW | `flux_down` | 0.103271484375 |
| RRTMG SW | `flux_up` | 0.08819580078125 |
| RRTMG SW | `heating_rate` | 3.164313966408372e-08 |
| RRTMG LW | `flux_down` | 0.0001220703125 |
| RRTMG LW | `flux_up` | 0.000152587890625 |
| RRTMG LW | `heating_rate` | 3.5666744224727154e-08 |
| MYNN | `theta` | 0.000244140625 |
| MYNN | `qv` | 2.1513551473617554e-07 |
| MYNN | `u` | 0.002731800079345703 |

## Coupled Sanity

Artifact: `proofs/perf/v017/fp32_physics_bench.json`

The short coupled sanity used the existing dummy coupled scan with
`GPUWRF_FP32_PHYSICS=1` for 10 steps on `8x8x8`.

Result:

- finite: `true`
- floating dtypes in final carry: `float32`, `float64`
- `mu` relative drift: `0.0`
- `qv` changed from `2.549692392349243` to `2.5118961334228516`

## Performance

Harness: `proofs/perf/v017/fp32_physics_bench.py`

The requested `proofs/perf/v017/bench_harness.py` was not present in this
v0.16.0 worktree, so this harness mirrors the requested schema:
`ms/step` cold and warm, `peak_vram_gib`, CPU/oracle compare, and
`measured_vs_projected`.  fp64 and fp32-physics variants run in separate child
processes so the env flag cannot be hidden by JIT cache reuse.

WS-128 settings:

- case: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/run_h36`
- grid: `128x128x44` (`ncol=16384`)
- `dt_s`: 10.0
- boundary off, GWD off, NoahMP off
- radiation cadence forced to 1 so radiation/PBL are a real per-step fraction

| Mode | Cold ms/step | Warm ms/step | Peak VRAM GiB | Finite |
| --- | ---: | ---: | ---: | --- |
| fp64 default | 10518.145379028283 | 386.0816579932968 | 12.772934675216675 | true |
| fp32 physics | 10456.960820010863 | 201.6272500040941 | 7.9162867069244385 | true |

Measured warm speedup: `1.914828764392994x`.

Peak VRAM reduction: `4.856647968292236 GiB` (`38.02%`).

Projection: none used.  The JSON reports measured-only; no projected speedup is
claimed.

## Canary Status

Canary timing is RED/not measured.  Both fp64 and fp32-physics workers failed
before timing with:

```text
FileNotFoundError: [Errno 2] No such file or directory:
/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_allgreen/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z/namelist.input
```

No Canary 9/3/1 speedup is reported from this run.
