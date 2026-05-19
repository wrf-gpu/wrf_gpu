# Memory Patch Proposal

## Scope

Two operational facts worth capturing for future M2-S3..S8 sprints and longer-term reference.

## Evidence

- Reviewer required the worker branch to not contain `scripts/dispatch_role.sh` changes. Manager contamination came from shared filesystem between manager process and codex worker. Documented as a real workflow gap.
- ALL M2 candidates will hit `ERR_NVGPUCTRPERM` because the local user lacks NVIDIA driver perfmon access. The contract's fallback pattern (cuobjdump + occupancy API + bench-output timing) is the project-standard workaround.

## Proposed Destination

In-sprint capture in the manager closeout (already done). For the broader workflow point, a manager-procedure note worth capturing in auto-memory.

## Patch

Append to `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_manager_autonomy.md`:

```markdown
## Operational addition 2026-05-19 (post M2-S2)

**Manager must not commit while a worker is in flight.** The codex worker runs in a separate tmux window but shares the same git working tree. Any `git checkout` the worker does (e.g. switching to its `worker/gpt/<sprint>` branch) silently moves the manager's HEAD too. A manager commit during this window lands on the worker branch by accident — a sprint contamination that reviewers correctly Reject.

Options to avoid:
1. Manager only commits when no worker is active (between sprint roles, not during).
2. Use `git worktree add` per role for isolation (heavier, deferred).

If contamination happens, the recovery is: cherry-pick the manager commit onto main, then `git rebase --onto <parent-of-contamination> <contamination-sha> <worker-branch>` to drop it from the worker branch.
```

Also append to project_target_hardware.md:

```markdown
## Profiler-counter restriction (added 2026-05-19 post M2-S2)

`ncu` (NVIDIA Nsight Compute) on this workstation fails with `ERR_NVGPUCTRPERM` because the local user account lacks driver perfmon permission. This is a system-level constraint, not a per-sprint fixable issue. Workaround for all M2+ profiling sprints: register/occupancy/local-memory metrics from `cuobjdump --dump-sass | grep` + `cuOccupancyMaxActiveBlocksPerMultiprocessor`; wall_time and transfer bytes from bench output; record a `profiler_limitation` field in the profile JSON. ADR-001 must label fallback-derived metrics (achieved_bandwidth_gbps) accordingly.

To enable real ncu counters, the system administrator must set `nvidia-driver-perfmon-allow=1` and reboot. Currently out of scope for the v0 bakeoff.
```

## Reviewer Status

Reviewer Status: not yet applied to auto-memory files. Manager will apply both edits as part of S2 hygiene before opening S3.
