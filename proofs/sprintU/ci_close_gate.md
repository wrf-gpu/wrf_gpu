# Sprint U / P0-5 — CI close-gate (idealized tests assert PASS, not false-green)

Date: 2026-05-29
Branch: `worker/opus/f7d-pressure-mass-fix`

## Finding being closed (GPT pre-close P0-4)

> `tests/idealized/test_warm_bubble.py` and `test_density_current.py` assert only
> `result.verdict in {"PASS", "FAIL"}`.  A regression to FAIL would still pass
> pytest, so a future "all tests passed" can be false-green for the dycore.

## Fix

1. New `close_gate` pytest marker (`pyproject.toml`) for the dycore close gate.
2. New `tests/idealized/test_dycore_close_gate.py` — runs both cases through the
   unified operational dycore and **asserts `verdict == "PASS"`** (with the failed
   sub-checks in the assertion message), then **archives the proof JSON** to
   `proofs/sprintU/close_gate/`.
3. The two existing idealized tests changed from `verdict in {PASS, FAIL}` to
   `verdict == "PASS"` and tagged `@pytest.mark.close_gate`.

A regression that drops a sub-check now turns the close gate RED instead of green.

## Evidence

```
$ pytest tests/idealized/test_dycore_close_gate.py -m close_gate -v
tests/idealized/test_dycore_close_gate.py::test_warm_bubble_close_gate_passes PASSED
tests/idealized/test_dycore_close_gate.py::test_density_current_close_gate_passes PASSED
======================== 2 passed in 411.59s (0:06:51) =========================
```

Archived verdicts (`proofs/sprintU/close_gate/`):
* `warm_bubble_verdict.json` → `verdict: PASS`
* `density_current_verdict.json` → `verdict: PASS`

## Notes

* The close gate requires a visible JAX GPU backend (CUDA RTX 5090). On a CPU-only
  box it skips with a clear reason rather than silently passing; in the GPU CI
  environment it is a hard PASS assertion.
* Run the close gate with `pytest -m close_gate` (the idealized cases are slow:
  ~7 min for both, dominated by Straka's 9000 steps).
