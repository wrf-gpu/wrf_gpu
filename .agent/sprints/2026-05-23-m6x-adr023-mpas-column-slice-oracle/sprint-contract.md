# Sprint Contract — M6.x ADR-023 MPAS Column-Slice Oracle (F1 closure)

## Objective

ADR-023 is now PROPOSED. The critic's MAJOR finding F1 (`reviewer-report.md §3.1`) flagged that the "mathematical equivalence to MPAS Klemp 2007" claim was unproven and requires a discrete derivation artifact + one MPAS column-slice comparison. The prototype already closed F3/F4/F10; this sprint closes F1.

**Deliverable**: a non-tautological 1-D column-slice oracle extracted from the local MPAS source, exposing per-substep `(w(z, t), theta(z, t), mu_perturbation(z, t), rho_perturbation(z, t), ph_perturbation(z, t))` as a reference trajectory for the ADR-023 conservative column solver to be compared against.

This unblocks the next production-grade sprint (`m6x-adr023-production-grade`) by providing the column-slice rung of the F6 acceptance ladder.

## Non-Goals

- **No edits to** `src/gpuwrf/dynamics/acoustic_wrf.py`, the analytic oracle, the test files from the prototype sprint, or any production code outside this sprint's file ownership. The slice oracle is a NEW validation artifact, not a modification.
- No re-implementation of MPAS. We extract a reference column from MPAS's existing Fortran implementation, not write a Python clone.
- No 24h forecast. Single column, single substep cadence is sufficient.
- No claim of "MPAS equivalence" in this sprint's outputs — only "MPAS-derived reference for comparison."
- No governance file edits.
- No host/device transfer inside the timestep loop (transfer-audit gate remains binding, even though this is a validation-only artifact).

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-adr023-mpas-column-slice-oracle`:

- `src/gpuwrf/validation/mpas_oracles/__init__.py` (new package)
- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` (new module — Python loader/runner for the slice)
- `tests/test_m6x_mpas_column_slice_oracle.py` (new)
- `data/fixtures/mpas_column_slice/<scenario>.npz` (new fixture — committed to `data/` if size permits, otherwise referenced via `fixtures/manifests/`)
- This sprint folder.

Read-only everywhere else, including `src/gpuwrf/dynamics/`.

## Inputs

Required reading:

- `.agent/decisions/ADR-023-conservative-column-solver.md` — your spec; especially §"Critic required fixes" F1 and §"Open questions" 3
- `.agent/sprints/2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md` — F1 + §3.1 detailed
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/worker-report.md` — what the prototype operator looks like (your slice must match those interfaces)
- `src/gpuwrf/dynamics/acoustic_wrf.py` — `vertical_acoustic_update` signature and post-solve outputs (your oracle exposes the same fields)
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — solver interface
- `src/gpuwrf/contracts/state.py` and `src/gpuwrf/contracts/grid.py` — `State` / `DycoreMetrics.flat()` shapes
- `tests/test_m6x_vertical_acoustic_oracle.py` — your test file follows the same conventions

MPAS source (local):

- `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`
  - Lines 437-475: vertical implicit coefficient assembly
  - Lines 1461-1656: `atm_compute_dyn_tend` + coefficient builder
  - Lines 1824-1832: explicit citation of Klemp et al. 2007 forward-backward integration
  - Lines 1889-1906: prognostic perturbation variables `ru_p`, `rw_p`, `rtheta_pp`, `rho_pp`
  - Lines 2038-2041: `resm = (1 - epssm) / (1 + epssm)`
  - Lines 2172-2208: upward sweep, downward sweep, back-substitution

Look for an existing MPAS idealized test case in the source tree: warm bubble, hydrostatic rest, or constant-N stratified column. Use the case that most closely matches the project's R7 analytic oracle setup (`tests/test_m6x_vertical_acoustic_oracle.py` plus `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py`).

If MPAS itself is not buildable on this workstation (likely — it needs nvfortran/parallel netCDF), extract the column-state intermediates from the **MPAS Fortran source code symbolically**: read the `atm_advance_acoustic_step` subroutine and reproduce a single-column version of its trajectory in Python by porting the equations literally (not algorithmically derived) for one column. The reference values come from the Fortran equations themselves, not from running the Fortran binary.

## Acceptance Criteria

1. **Slice oracle Python module** `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` exposing:
   ```python
   def mpas_column_slice(
       scenario: str,     # e.g., "warm_bubble_2km" or "stratified_rest"
       n_levels: int,
       column_height_m: float,
       dt_acoustic_s: float,
       n_substeps: int,
       epssm: float = 0.1,
   ) -> dict:
       """
       Returns dict with keys:
         - t: shape (n_substeps + 1,)
         - w: shape (n_substeps + 1, n_levels + 1)
         - theta_perturbation: shape (n_substeps + 1, n_levels)
         - ph_perturbation: shape (n_substeps + 1, n_levels + 1)
         - mu_perturbation: shape (n_substeps + 1,)
         - rho_perturbation: shape (n_substeps + 1, n_levels)
       """
   ```
   Implementation must be a **literal port of MPAS Fortran lines 2172-2208** (forward-sweep + back-substitution + perturbation reconstruction), with MPAS line citations in the docstring for every equation block. Pure NumPy — no JAX. Single column. No PDE solver substitution.

2. **Pytest oracle** `tests/test_m6x_mpas_column_slice_oracle.py` with at least three tests:
   - `test_slice_runs_warm_bubble_scenario` — invokes `mpas_column_slice(scenario="warm_bubble_2km", ...)`, verifies output shapes + finite values + warm-bubble rise (`w_max > 1 m/s` somewhere in the trajectory).
   - `test_slice_runs_stratified_rest_scenario` — invokes a hydrostatic-rest scenario, verifies `|w| < 1e-10` over all substeps.
   - `test_adr023_operator_matches_slice_within_tolerance` — runs the c2 `vertical_acoustic_update` for the same column setup as the slice, compares trajectories. **Initial tolerance**: 20% peak amplitude error tolerated for warm bubble. Document the actual measured error in the report.

3. **Fixture** `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` (or under `fixtures/manifests/` if size is binding): captured slice trajectory for fast pytest replay.

4. **No regression in existing tests**. Run `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v` and confirm 3+4+8 = 15 still passing.

5. **Worker report** at `worker-report.md` documenting:
   - The exact MPAS source lines ported (with `file:line` citations for every equation block).
   - The discretization choices and where they map to ADR-023 spec.
   - The measured trajectory-deviation of the prototype operator vs the slice (test 3 above).
   - Open questions for the production-grade sprint (e.g., "operator deviation is X% — is this acceptable for Tier-4 RMSE on U10/V10/T2?").

6. **Branch commits** on `worker/gpt/m6x-adr023-mpas-column-slice-oracle`. Multiple commits OK; the slice fixture commit must NOT be amended (kept reproducible).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_slice
pytest tests/test_m6x_mpas_column_slice_oracle.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_slice.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_no_regression.txt
```

## Performance Metrics

None — validation infrastructure sprint.

## Proof Object

- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` (Python module)
- `tests/test_m6x_mpas_column_slice_oracle.py` (3+ tests)
- `data/fixtures/mpas_column_slice/<scenario>.npz` (captured trajectory)
- `.agent/sprints/.../proof_slice.txt` (pytest output)
- `.agent/sprints/.../proof_no_regression.txt`
- `.agent/sprints/.../worker-report.md`

Time budget: **3-5 hours**.

## Risks

- **MPAS source readability**: the Fortran is dense; cite lines you actually ported. If a needed term is in a different MPAS file, find and cite it.
- **Tautology trap**: if your "MPAS slice" is just your own JAX integrator with a different name, the comparison test is meaningless. The Fortran-line-by-line port enforces this — every numerical step matches a cited MPAS line.
- **Fixture size**: a captured trajectory at typical resolution is ~kB to MB. Commit if <5 MB; otherwise generate on demand from the seed parameters and document.
- **Operator-vs-slice tolerance**: if the prototype operator deviates >50% from MPAS slice, the warm-bubble PASS was a coincidence (lucky stabilization heuristics) rather than physics. Flag this loudly in the report — that's information the production-grade sprint needs.
- **Spec-gaming**: M5 pattern. Verifiability triple still applies. The MPAS line citations are the anchor.

## Handoff Requirements

When all four `proof_*` files and the worker-report are written and committed, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr023-mpas-column-slice-oracle / codex] exit=<ec>` to the manager pane.

## Failure modes the manager will reject

- "MPAS-equivalent" claim without line-by-line citation.
- Slice that's a JAX integrator wearing an MPAS label.
- No comparison test against the prototype operator (test 3 above is mandatory).
- Tolerance set so loose that the comparison is vacuous (e.g., 1000% allowed).
- Editing `src/gpuwrf/dynamics/acoustic_wrf.py` or the analytic oracle.
