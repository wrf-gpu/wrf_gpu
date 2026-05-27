# Worker Report - M7 Restart Continuity

Summary: Implemented the M7 checkpoint/restart path and produced a PASS restart-continuity proof for the default 20260521 V3 IC with N=10. The checkpoint stores all 47 `State` fields explicitly, namelist/grid pytrees, step index, and optional runtime reproducibility state. The probe uses two fresh subprocesses for B1/B2 and restores the operational carry so the restarted run matches the unbroken 20-step reference with max delta 0.0 for every State field.

## Objective

Implement `.npz`/pickle-class restart continuity for M7 acceptance gate #3: run N steps, checkpoint, restart in a fresh process, run N more steps, and compare against a 2N-step reference within Tier-1 tolerance.

## Files Changed

- `src/gpuwrf/runtime/checkpoint.py`
- `src/gpuwrf/runtime/__init__.py`
- `scripts/m7_restart_continuity_probe.py`
- `tests/test_m7_restart_checkpoint_roundtrip.py`
- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`
- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_overhead.json`
- `.agent/sprints/2026-05-27-m7-restart-continuity/command_outputs/restart_b1.stdout`
- `.agent/sprints/2026-05-27-m7-restart-continuity/command_outputs/restart_b1.stderr`
- `.agent/sprints/2026-05-27-m7-restart-continuity/command_outputs/restart_b2.stdout`
- `.agent/sprints/2026-05-27-m7-restart-continuity/command_outputs/restart_b2.stderr`

## Commands Run + Output

- `taskset -c 0-3 python -m pytest tests/test_m7_restart_checkpoint_roundtrip.py -q`
  Output: `.. [100%] 2 passed in 1.95s`

- `taskset -c 0-3 python scripts/m7_restart_continuity_probe.py --n-steps 10`
  Output: `verdict=PASS`, `n_steps=10`, device `cuda:0`, all 47 field comparisons passed, maximum reported field delta `0.0`. B1 stdout/stderr and B2 stdout/stderr are captured under `command_outputs/`; both stderr files are empty.

- `git diff --check`
  Output: no output, exit 0.

- `taskset -c 0-3 python scripts/validate_agentos.py`
  Output: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`

Smoke note: an initial N=1 state-only smoke exposed nonzero deltas in operational surface/velocity fields because operational carry scratch was not checkpointed. After adding optional `runtime_state`, the N=1 smoke passed with max delta 0.0. Smoke artifacts and binary checkpoints were removed before commit.

## Proof Objects

- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`: PASS, 20260521 IC, 20-step reference vs 10+restart+10, every State field max delta 0.0.
- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_overhead.json`: checkpoint write 0.066475849 s, read 0.055848051 s, total overhead 0.1223239 s, 0.1262% of one N-step forecast wall-time basis.
- `tests/test_m7_restart_checkpoint_roundtrip.py`: unit proof for bitwise synthetic State round-trip and optional runtime-state preservation.

## Risks

- Checkpoints use pickle and are trusted-local artifacts, not a public interchange format.
- Exact restart continuity depends on saving the operational carry scratch as reproducibility state. `read_checkpoint()` still returns the contracted `(state, namelist, grid, step_index)` tuple, while the probe uses `read_checkpoint_with_runtime_state()` for the exact restart path.
- Binary checkpoint payloads are generated during the probe but intentionally not committed.

## Handoff

Objective complete. Files are within the sprint contract ownership. No model code or governance files were modified. Next decision needed: reviewer/tester should confirm that treating operational carry scratch as restart reproducibility state satisfies the M7 restart gate for the current operational mode.
