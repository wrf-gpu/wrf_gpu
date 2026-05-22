# Sprint Contract — M6.x Vertical-Acoustic Analytic Oracle (R7 closure prerequisite)

## Objective

The c2-A2 + c2-A2.x reviewer-report (`9bca47c`) R7 finding: **there is no vertical-acoustic analytic or savepoint oracle.** Every test added in c2-A2.x was structural/qualitative only ("w_max > 0", "shapes match"). The warm-bubble target [5, 10] m/s at 600 s is only checked by an integration harness, not at the operator level.

R7 is a pivot-blocker for ADR-021 AND ADR-022 — neither implementation can verify without an analytic oracle. The oracle is **shared between both pivot directions** (the operator may differ, but the linear-acoustic dispersion test does not), so building it now is unblocked work regardless of which ADR ratifies.

Worker job: implement a 1-D vertical linear-acoustic / gravity-wave column analytic oracle test, hook it into pytest, and prove it FAILS on the current `vertical_acoustic_update` so that whichever pivot lands will turn it green by closing R1/R2/R3.

## Non-Goals

- Do **not** modify `acoustic_wrf.py` or any other production code in `src/gpuwrf/dynamics/`. The current operator is expected to FAIL the oracle; that failure is the proof object.
- Do not implement the operator replacement. That is the pivot sprint, dispatched after the ADR ratifies.
- Do not extend the test to non-hydrostatic warm-bubble at this stage; that is integration testing.

## File Ownership

Write-only:
- `tests/test_m6x_vertical_acoustic_oracle.py` (new file)
- `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py` (new module — analytic solution generator)
- This sprint folder.

Read-only everywhere else.

## Inputs

- WRF source for the canonical operator form: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F` lines 620-651 (`calc_coef_w`), 1340-1489 (`advance_w` body), 1533-1584 (φ update).
- Current implementation under test: `src/gpuwrf/dynamics/acoustic_wrf.py`.
- Dispersion-relation derivation reference: WRF technical note (Skamarock et al. 2008) §3.2 (split-explicit vertical-acoustic). The Lipps-Hemler 1982 anelastic-approximation paper for the linearization geometry, if accessible; if not, derive from first principles.
- Project state contracts: `src/gpuwrf/contracts/state.py`, `src/gpuwrf/contracts/grid.py` — for the `State` / `BaseState` / `DycoreMetrics` shapes you can pass to the operator.

## Acceptance Criteria

1. **Analytic solution module** `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py` exposing a pure-NumPy function:
   ```python
   def vertical_acoustic_mode(
       n_levels: int,
       column_height_m: float,
       theta_base_K: float,
       brunt_vaisala_N_inv_s: float,
       wavelength_m: float,
       initial_amplitude_w_m_s: float,
       times_s: np.ndarray,
   ) -> dict:  # returns dict with keys: t, w(t, k), ph_perturbation(t, k), theta_perturbation(t, k), period_s, decay_rate_inv_s
   ```
   Implements the linear-mode solution for an isothermal-or-uniformly-stratified column under gravity, hydrostatic base, sound-speed `c_s = sqrt(gamma * R_d * T_base)`. The dispersion relation must be cited from a paper section number or derived in a comment block at the top of the file.

2. **Pytest oracle** `tests/test_m6x_vertical_acoustic_oracle.py` with at least three tests:
   - `test_linear_acoustic_period_matches_dispersion_relation` — initialize the operator with a single vertical mode (k_z = 2π / wavelength), advance 1 acoustic period, assert period match within <2%. Currently expected to FAIL on `vertical_acoustic_update`.
   - `test_no_drift_in_hydrostatic_rest_state` — initialize with zero wind, no perturbation, advance 1000 acoustic substeps, assert `|w| < 1e-12` and `|ph_perturbation - ph_perturbation_initial| < 1e-12`. Hydrostatic-rest invariance.
   - `test_amplitude_decay_within_2pct_of_analytic` — initialize with a damped mode, assert the e-folding time matches the analytic decay rate within 2%. (For schemes with zero numerical damping at the chosen `epssm`, the analytic decay is zero — the test asserts `|amplitude(T) - amplitude(0)| < tolerance`.)

3. **FAIL proof on current operator**. Run `pytest tests/test_m6x_vertical_acoustic_oracle.py -v` and capture the output to `proof.txt` in this sprint folder. The output must show all three tests FAILING (this is the proof that the oracle is non-tautological — the current `vertical_acoustic_update` is structurally non-WRF and is expected to miss). If any test PASSES on the current operator, investigate; either the test is tautological or the operator is closer to correct than the reviewer-report indicated.

4. **No regression in existing tests.** Run `pytest -q` and capture exit code; non-zero exit only if pre-existing failures change in count.

5. **Worker report** at `worker-report.md` documenting:
   - The dispersion relation derivation (or paper citation with section).
   - For each test: setup, what it measures, current FAIL evidence, what a passing implementation must satisfy.
   - List of tests pre-existing and unchanged.
   - Any divergence from the spec above.

## Validation Commands

```bash
pytest tests/test_m6x_vertical_acoustic_oracle.py -v | tee .agent/sprints/2026-05-23-m6x-vertical-acoustic-analytic-oracle/proof.txt
pytest -q
```

## Performance Metrics

None — test infrastructure.

## Proof Object

- `tests/test_m6x_vertical_acoustic_oracle.py`
- `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py`
- `.agent/sprints/.../proof.txt` (pytest -v output showing FAILs)
- `.agent/sprints/.../worker-report.md`

Time budget: 4-6 hours.

## Risks

- **Tautological oracle.** If the "analytic" function is just the implementation's own integrator at higher resolution, the test passes regardless of correctness. Mitigation: the analytic must be a closed-form expression derived from the dispersion relation, not a numerical integration of the same PDE.
- **Wrong dispersion relation.** The relevant one is for compressible-Boussinesq or full-Euler vertical-acoustic on the eta grid, not the shallow-water gravity-wave one. The comment block must show the derivation source so reviewer can verify.
- **Operator passes by accident.** If the current operator passes the oracle, the reviewer-report's R1/R2 findings are wrong — investigate and report. This is acceptable evidence either way; the sprint succeeds.
- **Test framework collision.** Existing M4 tests cover the reduced dycore. Use a distinct fixture set, no shared state. Filename `test_m6x_vertical_acoustic_oracle.py` is unique.

## Handoff Requirements

When `worker-report.md`, both code files, and `proof.txt` are written, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-vertical-acoustic-analytic-oracle / codex] exit=<ec> report=...` into the manager pane.
