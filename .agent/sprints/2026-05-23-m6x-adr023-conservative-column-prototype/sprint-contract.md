# Sprint Contract — M6.x ADR-023 Conservative Column Solver Prototype (single large sprint)

## Objective

User standing order: "Send out an agent to try a re-write with a different method that can be tested after one large sprint or so. Don't get stuck." This is that sprint.

Implement the ADR-023 conservative tridiagonal column solver for the c2 vertical-acoustic operator. The R7 analytic oracle (`tests/test_m6x_vertical_acoustic_oracle.py`) is already on main (commit `f6965be`) with 3 RED tests. **Acceptance is the 3 RED tests turning GREEN, plus the warm-bubble harness producing 600 s w_max in [5, 10] m/s.**

This runs **in parallel with the 3-way critic** so that when the critic ratifies (or rejects) ADR-023, the manager has code-running evidence rather than paper analysis.

## Non-Goals

- Do not change the c2-A2 horizontal PGF (`acoustic_wrf.py:309-408`) or mu continuity (`:508-540`). Those are ACCEPTed.
- Do not expand `AcousticScanCarry` beyond the c2-A2 5-tuple. The architectural payoff of ADR-023 is the small carry.
- Do not implement a Newton outer loop. ADR-023 v0 specifies linear single-pass tridiagonal.
- Do not modify the analytic oracle in `src/gpuwrf/validation/analytic_oracles/`. The oracle is fixed; the operator must come to it.
- Do not modify governance files.
- Do not introduce host/device transfers inside the timestep loop (transfer-audit gate remains binding).

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-adr023-conservative-column-prototype`:

- `src/gpuwrf/dynamics/acoustic_wrf.py` — replace `_calc_coef_w`, `vertical_acoustic_update`, `_vertical_theta_transport`. Preserve the 5-tuple carry signature.
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` (new) — Thomas / cyclic-reduction tridiagonal solver, JAX-native, no host transfer. May reuse `src/gpuwrf/numerics/tridiagonal_solver.py` if it exists from M5-S2; otherwise add the module here.
- `tests/test_m6x_adr023_column_solver.py` (new) — additional unit tests beyond the R7 oracle (e.g., tridiagonal solver correctness on a synthetic SPD system).
- This sprint folder.

Read-only everywhere else, including `src/gpuwrf/contracts/` and `src/gpuwrf/validation/`.

## Inputs

- `.agent/decisions/ADR-023-conservative-column-solver-DRAFT.md` — your spec
- `.agent/decisions/ADR-020-c2-dycore-architecture.md` — architecture constraints
- `.agent/decisions/ADR-007-precision-policy.md` — fp64 for pressure/mass/φ; fp32 OK for θ'
- `.agent/sprints/2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` — R1-R10 specific fixes (R3 hybrid-eta denominator, R4 msf-factor, R8/R9/R10 cleanup)
- `src/gpuwrf/dynamics/acoustic_wrf.py` — what you are replacing (vertical portion only)
- `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py` — your spec for what the oracle expects
- `tests/test_m6x_vertical_acoustic_oracle.py` — your 3 RED tests that must turn GREEN
- WRF source `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` lines 619-651 (calc_coef_w canonical form)
- MPAS source `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F` lines 1461-1656 (coefficient assembly), 2141-2208 (forward-sweep + back-substitution)
- ICON4Py (public, fetch via curl if needed) `model/atmosphere/dycore/stencils/vertically_implicit_dycore_solver.py:139-206, 283-356` and `solve_tridiagonal_matrix_for_w_forward_sweep.py:18-68` for the Thomas / forward-sweep pattern under GT4Py
- SCREAM (public) `components/eamxx/.../homme/src/share/cxx/DirkFunctorImpl.hpp:344-356, 707-778` for cyclic-reduction tridiagonal pattern (algorithmic only; we're JAX not Kokkos)

## Acceptance Criteria

1. **All 3 R7 oracle tests GREEN.** `pytest tests/test_m6x_vertical_acoustic_oracle.py -v` returns `3 passed`.
2. **Warm-bubble integration target.** Run the warm-bubble harness (`scripts/m6_warm_bubble_test.py` if it exists, else cite the closest M4/M6 harness). At 600 s simulated time, `w_max` ∈ [5, 10] m/s. Capture output to `proof_warm_bubble.txt`.
3. **Tridiagonal solver unit tests PASS.** `pytest tests/test_m6x_adr023_column_solver.py -v` returns all passed; include at least: solver correctness on a synthetic SPD system, boundary handling (`top_lid=True` and `top_lid=False`), zero-rhs invariance.
4. **No regression in c2-A2 horizontal tests.** `pytest tests/test_m6x_c2_acoustic.py -v` returns same pass count as before this sprint (capture the pre/post counts in the worker report).
5. **Transfer audit clean.** Run the existing transfer audit (`pytest tests/test_transfer_audit*` or equivalent) and confirm zero host/device transfers inside the timestep loop. The XLA static-kernel-check from the c2-A2 sprint family applies here.
6. **R3, R4, R8, R9, R10 closed**. Cite the resolution location (file:line) for each in the worker report.
7. **Worker report** at `worker-report.md` in this sprint folder. Must include: summary, dispersion-relation correctness argument vs the analytic oracle, the §5 cyclic-reduction-vs-Thomas trade-off decision and rationale, files changed, test outputs, transfer audit results, R-finding closures, risks, handoff. Token `Summary:` required.
8. **Branch + commits.** Work on branch `worker/gpt/m6x-adr023-conservative-column-prototype`. Multiple commits OK; final commit must leave the branch ready to merge.

## Validation Commands

```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_oracle.txt
pytest tests/test_m6x_adr023_column_solver.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_solver_unit.txt
pytest tests/test_m6x_c2_acoustic.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/proof_c2_horizontal_regression.txt
# Warm bubble (use whichever harness exists; check scripts/ first):
ls scripts/m6_warm_bubble* scripts/warm_bubble*
```

## Performance Metrics

- Tridiagonal solve must not produce host transfers (XLA static-kernel-check). Cite the launch count for the full vertical operator after the rewrite.
- ADR-007 4× target gate: report whether the operator's HLO launch count stays within the project's per-operator budget. This is informational only at this stage; not a sprint blocker.

## Proof Object

- `proof_oracle.txt`, `proof_solver_unit.txt`, `proof_c2_horizontal_regression.txt`, `proof_warm_bubble.txt` all in this sprint folder
- `worker-report.md` in this sprint folder
- Source files committed on branch `worker/gpt/m6x-adr023-conservative-column-prototype`

Time budget: **6-10 hours**. This is the user-mandated "single large sprint" — do not split into sub-sprints.

## Risks

- **Specification ambiguity** in ADR-023 §1. If the spec is unclear on a specific term, choose the MPAS Klemp 2007 form (the closest precedent) and document the choice in the worker report.
- **Tridiagonal solver primitive choice**. `lax.scan` Thomas is portable + simple; cyclic-reduction is faster on GPU but harder to get right. ADR-023 v0 suggested CR but the worker may choose Thomas if the oracle tests pass at acceptable cost. Document the choice and the launch count.
- **Warm-bubble harness path**. If `scripts/m6_warm_bubble_test.py` doesn't exist by that name, locate the closest equivalent and use it; do not block on missing harness — adapt or note in the report.
- **Transfer audit**. The c2-A2 family used XLA static-kernel-check; reuse it. Any new host transfer is a sprint blocker.
- **R7 oracle "operator-level exposure" issue** noted by the oracle worker in `worker-report.md` Risks section: "the hydrostatic-rest test fails on missing `vertical_acoustic_update` exposure after the zero-drift assertions pass." This is the operator-hook the prototype must wire correctly; do not change the test, change the operator hook.

## Handoff Requirements

When all four `proof_*.txt` files are written, `worker-report.md` is on disk, and the branch is pushed (locally — no remote push), type `/exit` as a slash command in the codex CLI. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr023-conservative-column-prototype / codex] exit=<ec>` to the manager pane.

## Failure modes the manager will reject

- **Tautological oracle bypass**. Do not modify the R7 analytic oracle or the test file to make the tests pass. Modify the operator.
- **Hidden host transfers**. Spurious `.tolist()`, `.item()`, `device_get`, `host_callback`, `pure_callback`, `io_callback`, `jax.lax.pure_callback` — all rejected by the transfer audit.
- **Carry expansion**. Do not add WRF small-step scratch fields to `AcousticScanCarry`. The whole point of ADR-023 is the small carry.
- **Newton outer loop**. ADR-023 v0 is linear single-pass; if you find Newton is needed, document it and report — do not silently add it.
- **Spec-gaming patterns** from the M5 cycle (worker-authored "real X" labels with clipped polynomial fits, vacuous tolerances, etc.). The "verifiability triple" rule applies: `nm`-symbol equivalent + non-clipped coefficient ratio + non-vacuous tolerance bound.
