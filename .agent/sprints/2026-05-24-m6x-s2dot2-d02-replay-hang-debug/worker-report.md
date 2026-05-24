# Worker Report

Summary: Fixed the d02 replay hang by narrowing it to the integration-layer replay scan compile path, then removing the dynamic radiation branch from the monolithic timestep scan. The observed stop point was `run_replay_proof()` after `run_replay_scan dispatch start`, before the later `block_until_ready` log; instrumentation proved imports, Gen2 data access, boundary packing, GPU allocation, and initial device sync all completed. The causal line range was the pre-fix `_candidate_timestep_adr023()` dynamic `jax.lax.cond` radiation predicate inside `src/gpuwrf/integration/d02_replay.py` (old scan body; current static replacement is lines 524-550 and segmented scan orchestration is lines 584-849). For `--duration-s 1`, `final_radiation=True` made that RRTMG branch part of the one-step scan compile, yielding the previous opaque zero-output timeout.

## Files Changed

- `scripts/m6_d02_boundary_replay_1h.py`: added flushed timing logs around imports, args, proof call, output write; added effective `dt_s` handling so `0 < duration_s < dt_s` runs one short step instead of zero steps.
- `src/gpuwrf/integration/d02_replay.py`: added debug timing probes; replaced dynamic radiation `lax.cond` with static no-radiation, one-step, and radiation-block helpers; kept top-level `run_replay_scan` jitted with static `grid`/`ReplayConfig`; added zero-step diagnostics handling; tightened replay trace transfer counting to H2D/D2H JSON trace events only.
- `tests/test_m6x_d02_replay_hang_debug.py`: new synthetic minimal replay smoke test calling `run_replay_scan()` and asserting device completion under 60s.
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/worker-report.md` and proof outputs.

## Commands Run

- `python -m py_compile scripts/m6_d02_boundary_replay_1h.py src/gpuwrf/integration/d02_replay.py tests/test_m6x_d02_replay_hang_debug.py`
  Output: exit 0, no stderr.
- Progressive duration sweep exactly per contract:
  `for D in 0 0.25 1 60 300; do timeout 1800 python scripts/m6_d02_boundary_replay_1h.py --duration-s "$D" --output .../proof_run_d${D}.json 2>&1 | tee -a .../proof_progressive.txt; done`
  Output: all PASS. JSON replay wall times: d0=0.019s, d0.25=54.254s, d1=90.762s, d60=230.710s in the sweep, d300=209.455s. CLI completion logs: d0 +14.628s, d0.25 +73.345s, d1 +120.933s, d60 +301.902s, d300 +313.108s.
- Explicit d=1 timing proof:
  `/usr/bin/time -p timeout 300 python scripts/m6_d02_boundary_replay_1h.py --duration-s 1 --output .../proof_replay_run_d1.json`
  Output: PASS, replay `wall_time_s=141.330`, CLI done at +193.534s, `real 195.05`.
- `pytest tests/test_m6x_d02_replay_hang_debug.py -v | tee .../proof_smoke.txt`
  Output: `1 passed in 30.78s`.
- No-regression command exactly per contract:
  `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m3_transfer_audit.py -v | tee .../proof_no_regression.txt`
  Output: `46 passed in 312.68s (0:05:12)`.
- `git diff --check`
  Output: exit 0, no whitespace errors.

## Proof Objects

- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_progressive.txt`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_run_d0.json`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_run_d0.25.json`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_run_d1.json`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_run_d60.json`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_run_d300.json`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_replay_runs.txt`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_replay_run_d1.json`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_smoke.txt`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_no_regression.txt`
- `.agent/sprints/2026-05-24-m6x-s2dot2-d02-replay-hang-debug/proof_d60_rerun.txt`

## Risks

- The d60 CLI end-to-end run is borderline against the 5 minute short-duration target: the exact sweep logged +301.902s and the rerun logged +300.777s, while replay scan wall time was under 231s and status PASS. The extra time is debug/proof/audit overhead, not a replay hang.
- CLI debug logging defaults on for this sprint so future failures are visible; callers can set `GPUWRF_D02_REPLAY_DEBUG=0` if they need quiet output.
- This sprint did not change `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, or `src/gpuwrf/physics/`.

## Handoff

The hang root cause is in the d02 replay integration scan/radiation scheduling, not data loading, imports, GPU init, or operator code. The worker branch is `worker/gpt/m6x-s2dot2-d02-replay-hang-debug`. Per sprint contract, I did not remote push.
