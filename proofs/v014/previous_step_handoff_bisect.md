# V0.14 Previous-Step Handoff Bisection

Verdict: `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.

## Summary

- Classification: `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.
- Final replay matches checkpoint: `True`.
- GPU used: `True`.
- CPU preflight: `CPU_LOAD_REQUIRES_GPU`.

## Snapshot Results

| surface | d01 | d02 | all target match WRF | static PB/MUB match | worst field | max abs |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `after_segment_replay_d02_step5997_before_final_partial_parent` | 1999 | 5997 | `False` | `False` | `MUB` | 1050.3046875 |
| `before_parent_d01_step2000_child_d02_step5997` | 1999 | 5997 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_parent_d01_step2000_before_child_force` | 2000 | 5997 | `False` | `False` | `MUB` | 1050.3046875 |
| `before_operational_force_d02_step5997` | 2000 | 5997 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_operational_force_before_child_step5998` | 2000 | 5997 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_child_advance_step5998_midscan_capture` | 2000 | 5998 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_child_advance_step5999_midscan_capture` | 2000 | 5999 | `False` | `False` | `MUB` | 1050.3046875 |
| `after_child_advance_step5999_before_checkpoint_write` | 2000 | 5999 | `False` | `False` | `MUB` | 1050.3046875 |

## Decision

Open a narrower earlier-handoff/source sprint before d02 step 5997; do not target _operational_force or final child _advance_chunk first.
