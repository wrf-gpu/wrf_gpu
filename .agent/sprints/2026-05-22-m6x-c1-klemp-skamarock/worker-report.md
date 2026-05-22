# M6.x-c1 Worker Report — Klemp-Skamarock Clean-Room

## Objective

Implement the c1 contingency acoustic core: Klemp 2007 §3a-c horizontal-explicit and vertical-implicit acoustic update, per-column tridiagonal solve, and small-step μ-continuity flux accumulator, without heuristic damping factors or physics-kernel changes.

## Files Changed

- `src/gpuwrf/dynamics/acoustic.py`
- `src/gpuwrf/dynamics/tridiag.py`
- `src/gpuwrf/dynamics/rk3.py`
- `src/gpuwrf/dynamics/step_debug_stripped.py`
- `src/gpuwrf/dynamics/advection.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/coupling/driver.py`
- `scripts/m6_full_domain_batching.py`
- `tests/test_m6x_fallback_c1_acoustic.py`
- `tests/test_m6x_fallback_c1_tridiag.py`
- `.agent/decisions/ADR-018-m6x-fallback-c1-tridiag-backend.md`
- `.agent/decisions/ADR-019-m6x-fallback-c1-klemp-skamarock-clean-room.md`

No files under `src/gpuwrf/physics/` were modified.

## Proof Objects Produced

- `artifacts/m6x-fallback-c1/tridiag_benchmark.json`
- `artifacts/m6x-fallback-c1/cfl_diagnostic.json`
- `artifacts/m6x-fallback-c1/full_domain_batching_0p05h_probe.json`
- `artifacts/m6x-fallback-c1/full_domain_batching_1h_bound_probe.json`
- `artifacts/m6x-fallback-c1/full_domain_batching_1h_bound_no_radiation_probe.json`

## Commands Run

```bash
pytest -q tests/test_m6x_fallback_c1_tridiag.py tests/test_m6x_fallback_c1_acoustic.py
pytest -q tests/test_m4_acoustic.py tests/test_m4_dycore_step.py tests/test_m4_rk3.py
pytest -q tests/test_m4_dycore_step.py tests/test_m6x_*.py
git diff -- src/gpuwrf/physics/
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 0.05 --tier2-hours 0.05 --output artifacts/m6x-fallback-c1/full_domain_batching_0p05h_probe.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/m6x_c1_0p05h_probe --skip-nsys --skip-legacy-baseline-sanitize-audit --audit-steps 2 --audit-block-steps 18
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/full_domain_batching_1h_bound_probe.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/m6x_c1_1h_bound_probe --skip-nsys --skip-legacy-baseline-sanitize-audit --audit-steps 2 --audit-block-steps 120
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/full_domain_batching_1h_bound_no_radiation_probe.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/m6x_c1_1h_bound_no_rad_probe --skip-nsys --skip-legacy-baseline-sanitize-audit --audit-steps 2 --audit-block-steps 120 --radiation-cadence-steps 100000 --skip-final-radiation
```

## Result

Status: **implementation landed, coupled acceptance still FAIL**.

The acoustic-only Gen2 d02 scan remains finite for 60 large steps with state-bound `n_acoustic=86`. The 18-step coupled probe also has zero sanitize firing. The 1h coupled probes still fail, including the no-radiation-cadence run, so the remaining red gate is not the old acoustic CFL or RRTMG cadence alone.

The strongest current hypothesis is interaction between the real-domain advection/coupling path and the c1 pressure/μ fields. A direct dycore-only scan with advection included becomes nonfinite around step 30, while acoustic-only does not.

## Unresolved Risks

- AC6/AC7 are not met: 1h coupled probes still hit finite-guard bounds, so 24h was not run to PASS.
- AC9 is not met: transfer audit remains at the existing `167904` post-init bytes in the harness trace.
- ADR-007 remains FAIL; it was not amended.
- `src/gpuwrf/dynamics/advection.py` now advects perturbation pressure instead of total pressure to avoid transporting the static hydrostatic base state. Reviewer should explicitly accept or reject this as within c1 dycore scope.

## Next Decision Needed

Decide whether c1 continues into a focused advection/boundary-consistency follow-up, or whether the manager escalates to the c2 semi-implicit path. The immediate technical fork is whether real-domain dycore advection must stop using periodic halo assumptions before the c1 acoustic core can be fairly judged on the d02 coupled forecast.
