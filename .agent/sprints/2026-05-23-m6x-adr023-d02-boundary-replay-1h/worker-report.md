# Worker Report — M6.x ADR-023 d02 Boundary Replay 1h

Summary: HALT-BY-MANAGER-PATH-SPLIT. I stopped before producing the required 1h replay proof because the manager reported that the production-grade ADR-023 sprint was rejected and that the public nonhydrostatic scan path currently routes through prototype-grade stabilization, not the MPAS-recurrence path that produced the 1.69% RMSE evidence. Continuing would validate the wrong branch. I committed only incremental scaffolding: replay orchestration, integration helper, smoke test, and a failed debug proof from the now-superseded path.

## Files Changed

- `scripts/m6_d02_boundary_replay_1h.py`
- `src/gpuwrf/integration/d02_replay.py`
- `tests/test_m6x_d02_boundary_replay.py`
- `.agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_debug_10step_theta_base.json`
- `.agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/worker-report.md`

## Commands Run

- `python -m py_compile scripts/m6_d02_boundary_replay_1h.py src/gpuwrf/integration/d02_replay.py tests/test_m6x_d02_boundary_replay.py`
  - Output: no stdout/stderr, exit 0.
- `python scripts/m6_d02_boundary_replay_1h.py --duration-s 10 --skip-trace-audit --skip-static-audit --output .../proof_debug_1step.json ...`
  - Output: failed before proof write with JAX scan carry dtype mismatch: previous pressure was `float32` while solver output was `float64`.
- `python scripts/m6_d02_boundary_replay_1h.py --duration-s 1 --dt-s 1 --skip-trace-audit --skip-static-audit --output .../proof_debug_1step_dt1.json ...`
  - Output: `status=PASS`, `steps=1`, `first_nonfinite_step=null`, peak GPU memory about 9.31 GB.
- `pytest tests/test_m6x_d02_boundary_replay.py -v`
  - First output: failed; 10-step smoke found `first_nonfinite_step=2`.
  - Second output after replay-local alias reconstruction: failed; finite check passed but trace audit reported `post_init_total_bytes=3693908`.
- `python scripts/m6_d02_boundary_replay_1h.py --duration-s 10 --dt-s 1 --skip-trace-audit --skip-static-audit --output .../proof_debug_10step_theta_base.json ...`
  - Output proof status: `FAIL`, `steps=10`, `first_nonfinite_step=2`, RMSE informational values `T2=139.613 K`, `U10=122.419 m s-1`, `V10=123.480 m s-1`, `theta_k20=30.650 K`, `w_k20=49.653 m s-1`.

The contract validation commands were not run after the manager halt. No `proof_d02_replay.json`, `proof_d02_smoke_and_audit.txt`, or `proof_d02_no_regression.txt` acceptance artifact was produced.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_debug_10step_theta_base.json` — failed debug artifact only; not acceptance evidence.
- This `worker-report.md` — records the manager halt and path-split rationale.

## Risks

- Current scaffolding intentionally points at the existing public ADR-023 scan path, which the manager now says is the wrong branch to validate.
- The debug test also exposed a transfer-audit trace count in the 10-step smoke after finite-state repair; this was not investigated because the sprint was halted.
- The 1h full-domain replay was not run and no RMSE evidence should be used for M6 close from this branch.

## Handoff

- Objective: halted d02 replay integration after manager path-split rejection.
- Files changed: listed above.
- Commands run: listed above with outputs.
- Proof objects produced: failed debug proof and this halt report.
- Unresolved risks: ADR-023 public path must be unified with the MPAS-recurrence branch before replay evidence is meaningful.
- Next decision needed: redispatch this sprint only after the corrected ADR-023 path is merged and reviewer-accepted.
