# Manager Closeout

## Outcome

Verdict: `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.

The initial d02 JAX `OperationalCarry` matches native `wrfinput_d02` for
`PB/MUB`, but not CPU-WRF h0/h1/h10 or h10 pre-RK truth. CPU-WRF `PB/MUB` are
stable across those surfaces on the target patch, so replay-time drift is not
needed to explain the bad h10 base carry.

## Proof Objects

- `proofs/v014/earlier_source_bisect.py`
- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`
- External compact replay:
  `/mnt/data/wrf_gpu2/v014_earlier_source_bisect/earlier_source_bisect.live_replay_compact.json`

## Merge Decision

Merge Decision:

Accept and land. This closes the earlier-source bisection sprint as an evidence
proof.

## Scope Changes

Next scope is source-changing and narrow:
`src/gpuwrf/integration/d02_replay.py::build_replay_case` native child
base-state split construction plus proof scripts. TOST, Switzerland, FP32, and
broad memory work remain paused until this grid-parity bug is fixed or
explicitly bounded.

## Lessons

The bad base carry is not caused by the final partial subcycle, checkpoint
serialization, or replay-time drift after initialization. The root issue is
that the live-nested child starts from a `wrfinput_d02` base-state split while
CPU-WRF's forecast/history state uses a post-initialization `PB/MUB` split.

## Next Sprint

Open `.agent/sprints/2026-06-09-v014-base-state-split-fix/` and dispatch a
source-changing worker. The worker must reproduce WRF's post-initialization
`PB/MUB` split or prove the exact WRF/source hook needed.
