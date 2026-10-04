# V0.22 K2 Nest dt-ladder Synthesis

Date: 2026-06-26

## Verdict

**FAIL / stop after N1.** The requested 3-domain nest baseline rung
(`d01/d02/d03 = 54/18/6 s`, `n_sound=10`) did not complete the 3 h gate.
It wrote valid 18/36/54 min frames, then failed at 72 min with a non-finite
d03 `Ni` state:

`NonFiniteStateError: domain=d03 field=Ni level=0 step=720 sim_time_s=4320 first_index=(0, 12, 11)`

Because the baseline N1 rung failed, N2-N4 were not launched. Higher root dt
rungs would not be meaningful on top of an unstable baseline comparison.

## Configuration

- Input: `<DATA_ROOT>/wrf_downscale/runs/20250121/cpu`
- Max domains: 3
- Requested N1 dt: d01 `54 s`, d02 `18 s`, d03 `6 s`
- `n_sound`: 10
- Gate length: 3 h
- History cadence: 18 min
- GPU execution: `scripts/with_gpu_lock.sh --timeout 21600 --label k2-nest-ladder -- ...`
- Ladder config: `proofs/v022/k2_nest_ladder/ladder_config.json`

The 3 h length was used because a 1 h run does not align exactly to the
requested root dt of 54 s.

## N1 Diagnostics

Proof object: `proofs/v022/k2_nest_ladder/runs/N1/result.json`

| Domain | dt | Cx | Cy | Cz | Ctotal |
|---|---:|---:|---:|---:|---:|
| d01 | 54 s | 0.522 | 0.213 | 0.656 | 1.391 |
| d02 | 18 s | 0.355 | 0.186 | 1.398 | 1.932 |
| d03 | 6 s | 0.337 | 0.179 | 1.051 | 1.565 |

Gate summary:

- `run_ok`: false
- finite/bounded through last written frame: true
- horizontal CFL target: true
- total CFL target: false
- worst vertical Courant: `1.398`
- worst total Courant: `1.932`

The key signal is the vertical CFL stress: d03, the inner 1 km nest targeted by
the brief, already crosses `Cz > 1`; d02 is worse and pushes total Courant well
past the diagnostic target before the d03 non-finite failure.

## Commands Run

```bash
scripts/with_gpu_lock.sh --timeout 21600 --label k2-nest-ladder -- \
  env K2_NEST_INPUT_DIR=<DATA_ROOT>/wrf_downscale/runs/20250121/cpu \
      K2_NEST_MAXDOM=3 \
      K2_NEST_RUNG_TIMEOUT=5400 \
      K2_NEST_MANAGER_PANE=0:1.0 \
      proofs/v022/k2_nest_ladder/run_nest_ladder.sh
```

```bash
env PYTHONPATH=src JAX_ENABLE_X64=true \
  python proofs/v022/k2_nest_ladder/summarize_nest_rung.py \
    --tag N1 \
    --root-dt 54 \
    --n-sound 10 \
    --run-dir proofs/v022/k2_nest_ladder/runs/N1 \
    --hours 3 \
    --maxdom 3 \
    --rc 1 \
    --log proofs/v022/k2_nest_ladder/logs/N1.log \
    --out proofs/v022/k2_nest_ladder/runs/N1/result.json
```

## Proof Objects

- Run log: `proofs/v022/k2_nest_ladder/logs/N1.log`
- Corrected summary log: `proofs/v022/k2_nest_ladder/logs/N1_summary.log`
- Rung result JSON: `proofs/v022/k2_nest_ladder/runs/N1/result.json`
- Harness: `proofs/v022/k2_nest_ladder/run_nest_ladder.sh`
- Summarizer: `proofs/v022/k2_nest_ladder/summarize_nest_rung.py`

## Decision

K2 should remain **opt-in** for the nest path until there is a stable nest
baseline and a fresh dt ladder. The single-domain K2 result remains valuable,
but this 3-domain gate found the intended steep-terrain/nest CFL risk before
any higher-dt speedup rung could be evaluated.
