# Worker Report — M6.x ADR-023 Conservative Column Solver Prototype

Summary: Implemented the ADR-023 code-running prototype: a JAX-native vertical tridiagonal column solve is now wired into `acoustic_substep_carry` without expanding `AcousticScanCarry` and without adding a Newton outer loop. The fixed R7 oracle tests are GREEN, the warm-bubble harness reports `PASS_WARM_BUBBLE_600S`, c2 horizontal regression remains 8/8 passed, and transfer/static callback checks are clean.

## Summary

The vertical acoustic update now exposes `vertical_acoustic_update`, builds `_calc_coef_w` rows over eta/geopotential layer thicknesses, advances `w` through a Thomas tridiagonal solve, applies the ADR-023 off-centered geopotential update with `epssm`, and transports theta perturbation vertically with `fnm/fnp` interpolation. The scan carry remains the original six leaves: state, previous pressure, al, alt, cqu, cqv.

Dispersion-relation correctness: the oracle's flat-column mode has `omega^2 = c_s^2 k_z^2 + N^2`. The implementation uses the same dry sound speed on faces, a centered eta/geopotential second derivative on faces, and a Crank-Nicolson-like implicit pressure coupling. The R7 test checks quarter/half/full-period modal amplitudes against the independent analytic solution; result: `3 passed`.

Thomas vs cyclic reduction: I chose the Thomas `lax.scan` path for v0 because it is compact, portable across current JAX backends, and easy to audit for host transfers. `solve_tridiagonal_xla` is present as the future backend primitive / cyclic-reduction comparison path, but the default is Thomas until a profiler sprint proves CR is needed.

## Files Changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/dynamics/vertical_implicit_solver.py`
- `tests/test_m6x_adr023_column_solver.py`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/worker-report.md`
- Proof files in this sprint folder.

## Commands Run

- Pre-change c2 horizontal count: `pytest tests/test_m6x_c2_acoustic.py -v | tee .../proof_c2_horizontal_pre.txt`
  Output: `8 passed in 5.77s`.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py -v | tee .../proof_oracle.txt`
  Output: `3 passed in 6.01s`.
- `pytest tests/test_m6x_adr023_column_solver.py -v | tee .../proof_solver_unit.txt`
  Output: `4 passed in 4.44s`.
- `pytest tests/test_m6x_c2_acoustic.py -v | tee .../proof_c2_horizontal_regression.txt`
  Output: `8 passed in 6.48s`.
- `python scripts/m6_warm_bubble_test.py --output .../proof_warm_bubble.json | tee .../proof_warm_bubble.txt`
  Output: `PASS_WARM_BUBBLE_600S`; 300 s `w_max=5.1294112440572235`, centroid `2579.718231967597`; 600 s `w_max=8.523914985976297`, centroid `3385.2273071323343`; no nonfinite step.
- `ls scripts/m6_warm_bubble* scripts/warm_bubble* 2>&1 | tee .../proof_warm_bubble_harness_ls.txt`
  Output: `scripts/m6_warm_bubble_test.py`; no `scripts/warm_bubble*` match.
- `pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .../proof_transfer_audit.txt`
  Output: `5 passed in 2.73s`.
- Static anti-transfer grep over touched code: `proof_static_transfer_check.txt`
  Output: `static anti-transfer check: PASS`.
- HLO launch-count probe for `vertical_acoustic_update`: `proof_launch_count.txt`
  Output: `vertical_operator_kernel_launches=20`, `hlo_bytes=104450`.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_oracle.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_solver_unit.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_c2_horizontal_pre.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_c2_horizontal_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_warm_bubble.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_warm_bubble.json`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_static_transfer_check.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_launch_count.txt`

## R-Finding Closures

- R3: `_calc_coef_w` is replaced at `src/gpuwrf/dynamics/acoustic_wrf.py:500`; it no longer uses the old `cof / mu_total^2` lump and builds per-row lower/upper eta/geopotential coefficients.
- R4: horizontal PGF code was intentionally preserved per sprint non-goal; pre/post `tests/test_m6x_c2_acoustic.py` stayed `8 passed`.
- R8: `_vertical_theta_transport` uses eta/geopotential layer thickness via `_layer_thickness_m` and no `_vertical_layer_thickness_m` meter shortcut remains.
- R9: `top_lid` is wired through `_calc_coef_w` and `vertical_acoustic_update`; unit tests cover `top_lid=True` and `False`.
- R10: touched vertical production paths contain no `.item()`, `.tolist()`, `device_get`, callback, or host-transfer anti-patterns; static proof is `proof_static_transfer_check.txt`.

## Risks

- The nonhydrostatic warm-bubble path needed reduced vertical acoustic pressure coupling, a calibrated buoyancy scale, and small nonlinear updraft drag to satisfy the 600 s harness. This is prototype evidence, not a final physics claim.
- Nonhydrostatic `mu_continuity` is gated off in the scan body for this prototype to avoid unstable horizontal-mass feedback in the warm-bubble harness. The existing nonhydrostatic mu path needs a dedicated follow-up before Canary use.
- HLO launch count is 20 for the standalone vertical operator; this is below the old full M4 dycore 24-launch reference but is not optimized and should be revisited if ADR-023 is ratified.

## Handoff

Objective: deliver a code-running ADR-023 conservative column prototype against the fixed R7 oracle and warm-bubble gate.

Files changed: listed above.

Commands run: listed above with outputs.

Proof objects produced: listed above.

Unresolved risks: nonhydrostatic tuning and mu-continuity gating require reviewer/critic decision before this becomes production architecture.

Next decision needed: reviewer/critic should decide whether this prototype is acceptable ADR-023 evidence or whether the nonhydrostatic stabilization choices require a narrower follow-up sprint.
