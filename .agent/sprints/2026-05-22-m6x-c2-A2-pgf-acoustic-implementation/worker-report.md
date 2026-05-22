# c2-A2 Worker Report

## Objective

Implement the ADR-020 WRF-shaped horizontal PGF, diagnostic acoustic scan carry, and in-loop mu continuity. Run the sequential gates through the first failing proof object without bundling downstream fixes.

## Files Changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `tests/test_m6x_c2_pgf.py`
- `tests/test_m6x_c2_acoustic.py`
- `scripts/m6_warm_bubble_test.py`
- `.agent/sprints/2026-05-22-m6x-c2-A2-pgf-acoustic-implementation/proofs/phase1_phase2_unit_tests.json`
- `.agent/sprints/2026-05-22-m6x-c2-A2-pgf-acoustic-implementation/proofs/warm_bubble_600s.json`
- `.agent/sprints/2026-05-22-m6x-c2-A2-pgf-acoustic-implementation/worker-report.md`

## Commands Run

- `python -m py_compile src/gpuwrf/dynamics/acoustic_wrf.py` -> PASS
- `PYTHONPATH=src pytest -q tests/test_m6x_c2_scan.py tests/test_m6x_c2_stabilizers.py` -> `7 passed in 8.89s`
- `PYTHONPATH=src pytest -q tests/test_m6x_c2_pgf.py` -> `7 passed in 6.48s`
- `PYTHONPATH=src pytest -q tests/test_m6x_c2_acoustic.py` -> `8 passed in 7.68s`
- `PYTHONPATH=src pytest -q tests/test_m6x_c2_*.py tests/test_m6_state_extension.py` -> `30 passed in 19.08s`
- `python -m json.tool .agent/sprints/2026-05-22-m6x-c2-A2-pgf-acoustic-implementation/proofs/phase1_phase2_unit_tests.json >/dev/null` -> PASS
- `python -m py_compile scripts/m6_warm_bubble_test.py` -> PASS
- `PYTHONPATH=src python scripts/m6_warm_bubble_test.py --nx 8 --ny 8 --nz 6 --duration-s 4 --dt-s 2 --n-acoustic 2 --output /tmp/warm_bubble_smoke.json` -> expected nonzero, `FAIL_MISSING_SAMPLE`
- `PYTHONPATH=src python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-22-m6x-c2-A2-pgf-acoustic-implementation/proofs/warm_bubble_600s.json` -> expected nonzero, `FAIL_NONFINITE`

## Proof Objects Produced

- `proofs/phase1_phase2_unit_tests.json`: PASS for AC1-AC4 unit gate.
- `proofs/warm_bubble_600s.json`: FAIL for AC6. First nonfinite step is 1; 300s/600s `w_max_m_s` remains `0.0`; theta centroid remains near `2000.0001 m`.

## Implementation Notes

- PGF first three terms follow WRF `module_small_step_em.F:828-831` and `:902-905`.
- Non-hydrostatic fourth term follows WRF `module_small_step_em.F:836-862` and `:910-936`.
- M1 ambiguity is resolved in code as the literal WRF factor `-0.5*c1h*(mu_left + mu_right)`.
- M2 is documented in `acoustic_wrf.py`: `php` and `dpn` are substep-local intermediates.
- `al/alt/cqu/cqv` are explicit `AcousticScanCarry` leaves.
- `mu_perturbation` is updated inside the acoustic substep scan, with `mu_total` kept aligned to `BaseState.mub + mu_perturbation`.

## Unresolved Risks

- AC6 is not green. The restored warm-bubble harness shows the current c2 acoustic path becomes nonfinite at the first large step and has no vertical warm-bubble rise.
- The current implementation covers horizontal PGF and mu continuity, but does not yet implement WRF vertical acoustic momentum/geopotential/theta transport needed for warm-bubble parity.
- Because AC6 failed, AC7 Schar mountain, AC8 1h coupled, AC9 24h, AC10 speedup, and AC11 Tier-4 RMSE were not run.

## Next Decision Needed

Dispatch a focused c2-A2.x fix for the first AC6 failure, starting with vertical acoustic momentum/geopotential/theta transport and a bisection flag set in `scripts/m6_warm_bubble_test.py`. Do not proceed to sanitize-based coupled probes until warm-bubble passes.
