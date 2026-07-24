# GPU A/B Scheduling Decision

## Decision

`DEFER_TO_NEXT_CONFLICT_FREE_WINDOW`

No GPU A/B was started on 2026-07-09. Deployed v0.23.3 and all live source,
services, queue behavior, and production settings remain unchanged.

## Owner gates at 2026-07-09 22:45 WEST

- Active `20250228_18z` GPU filler: not terminal; latest complete d03-d09 group
  was forecast valid `2025-03-01_14:00:00`, with an 18-hour target.
- Product `20260709_00z`: not terminal or publish+verified; latest d02 output was
  valid `2026-07-15_00:00:00`.
- The next 18z GPU window begins around the 00:30 cycle choreography.
- Two isolated three-hour arms, launch-time Nsight captures, exact wrfout/subset
  QA, memory/transfer analysis, and verdict cannot safely fit after both active
  prerequisites finish and before that window.

Starting an abbreviated or overlapping A/B would violate the owner directive
and invalidate the primary comparison. The sidecar remains CPU-accepted only.

## Next trigger

Run the contract A/B only after both prerequisites are terminal and a quiet
window can contain both full arms plus QA without production CPU WRF or nightly
GPU overlap. Recheck the GPU lock, available window, fixture, cache, cpuset,
flags, and deployed/candidate hashes immediately before launch.
