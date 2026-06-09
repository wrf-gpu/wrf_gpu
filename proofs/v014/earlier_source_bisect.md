# V0.14 Earlier-Source Bisection

Verdict: `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.

## Summary

- Classification: `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.
- GPU used: `True`.
- CPU preflight: `CPU_LOAD_REQUIRES_GPU`.

## Snapshot Results

| surface | d01 | d02 | native PB/MUB match wrfinput | CPU-WRF PB/MUB match | worst primary field | max abs |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `initial_native_load_carry` | 0 | 0 | `True` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_600` | 200 | 600 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_1200` | 400 | 1200 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_1800` | 600 | 1800 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_2400` | 800 | 2400 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_3000` | 1000 | 3000 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_3600` | 1200 | 3600 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_4200` | 1400 | 4200 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_4800` | 1600 | 4800 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_5400` | 1800 | 5400 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_replay_segment_d02_step_5997` | 1999 | 5997 | `False` | `False` | `MUB` | 1050.3046875 |

## Decision

Open a source-changing fix sprint for src/gpuwrf/integration/d02_replay.py::build_replay_case native child base-state split construction; reproduce WRF's post-initialization PB/MUB split or load an accepted h0 base-state oracle before replay.
