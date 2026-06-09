# Manager Closeout

## Outcome

Verdict: `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.

The final producer-shaped replay is exact against the existing bad checkpoint
for all target leaves. The bad d02 carry is already wrong at completed step
5997, before parent step 2000, `_operational_force`, and child steps 5998-5999.
At the earliest captured surface, `MUB` is the worst target field with max_abs
`1050.3046875`; `PB` is also already wrong with max_abs `1047.015625`.

## Proof Objects

- `proofs/v014/previous_step_handoff_bisect.py`
- `proofs/v014/previous_step_handoff_bisect.json`
- `proofs/v014/previous_step_handoff_bisect.md`
- `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`
- External compact replay:
  `/mnt/data/wrf_gpu2/v014_previous_step_handoff_bisect/previous_step_handoff_bisect.live_replay_compact.json`

## Merge Decision

Merge Decision:

Accept and land. This closes the previous-step handoff sprint as an evidence
proof, not a source fix.

## Scope Changes

No production code changes. The next sprint scope moves earlier than d02 step
5997 and should test native load / initial carry / earlier replay segments
before touching runtime dycore, acoustic, pressure-gradient, force, or final
child advance code.

## Lessons

Checkpoint serialization is already ruled out by the prior source trace, and
the final partial subcycle is now ruled out by this bisection. The current
debug target is an earlier source of bad base/static or perturbation state,
especially the `PB/MUB/MU/T/P` split and native-domain load/replay handoff.

## Next Sprint

Open `.agent/sprints/2026-06-09-v014-earlier-source-bisect/` to decide whether
the mismatch is present at native load/initial carry, appears in the first or a
later replay segment before d02 step 5997, or requires a narrower source hook.
