# Sprint Contract — M6.x ADR-023 Production-Grade Implementation

## Objective

Prototype landed: 3/3 R7 oracle GREEN, warm-bubble 600s PASS, MPAS slice agreement at 1.92% peak / 38.7% trajectory RMSE. ADR-023 ratified PROPOSED.

This sprint promotes the prototype to **production-grade**: replace prototype-grade stabilization heuristics with derivation-driven equivalents, close critic findings F2/F5/F7/F9, un-gate nonhydrostatic `mu_continuity`, sweep `epssm`, and drive MPAS-slice trajectory RMSE down toward operationally acceptable (target <15% per critic §3.6 expectations).

## Non-Goals

- No expansion of `AcousticScanCarry` beyond the 6-leaf form. ADR-023 §1 binding.
- No Newton outer loop without explicit ADR-023 amendment. v0 stays linearized.
- No modification of R7 analytic oracle or its test file.
- No modification of MPAS column-slice oracle or its test file.
- No changes to c2-A2 horizontal PGF (`acoustic_wrf.py:309-408`) or mu_continuity_tendency.
- No 24h forecast yet — that's the next-after sprint.
- No governance file edits.
- No host/device transfer inside the timestep loop.

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-adr023-production-grade`:

- `src/gpuwrf/dynamics/acoustic_wrf.py` — replace prototype-grade `vertical_acoustic_update`, `_calc_coef_w`, `_vertical_theta_transport`, and `_vertical_buoyancy_acceleration` with production-grade equivalents. Un-gate nonhydrostatic `mu_continuity` properly. Document the post-solve replacement order for `(w, theta, ph_perturbation, mu_perturbation, p_perturbation, al, alt)` (F9).
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — extend with `epssm`-aware coefficient builder. May add Thomas-vs-CR profiler comparison output.
- `.agent/decisions/ADR-023-conservative-column-solver.md` — fold F2 (explicit linearized framing), F7 (carry vs locals vs scratch), and F9 (post-solve order) into the ADR text. Mark F1/F3/F4/F10 CLOSED with citations. Refresh status to PROPOSED → ACCEPTED once this sprint's reviewer concurs.
- `tests/test_m6x_adr023_production_grade.py` (new) — acceptance-ladder gate tests (F6): mpas-slice trajectory RMSE under target; epssm sweep; mu_continuity coupled in-scan; post-solve order assertion.
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/` — proofs + worker-report.

Read-only everywhere else, including the analytic and MPAS slice oracles, `src/gpuwrf/contracts/`, `tests/test_m6x_vertical_acoustic_oracle.py`, `tests/test_m6x_adr023_column_solver.py`, `tests/test_m6x_c2_acoustic.py`, `tests/test_m6x_mpas_column_slice_oracle.py`, `src/gpuwrf/validation/`.

## Inputs

Required reading:
- `.agent/decisions/ADR-023-conservative-column-solver.md` — §"Critic required fixes" (your blockers F2/F5/F7/F9), §"Prototype caveats", §"Fallback trigger"
- `.agent/sprints/2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md` — §3.1 F1 (closed), §3.2 F2 (linearized framing — yours to fold), §3.5-3.10 (F5-F10)
- `.agent/sprints/2026-05-23-m6x-adr023-conservative-column-prototype/worker-report.md` — §Risks (prototype caveats to replace)
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/worker-report.md` — measured deviation; trajectory shape mismatch to drive down
- `src/gpuwrf/dynamics/acoustic_wrf.py` — current operator (prototype-grade)
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — solver primitive
- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` — your reference; do not modify
- `tests/test_m6x_mpas_column_slice_oracle.py` — your reference test pattern

MPAS source (read-only reference):
- `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F` lines 1589-2208 (already cited by slice)

## Acceptance Criteria

1. **MPAS slice trajectory RMSE driven down**. New gate test asserts `trajectory_rmse_fraction < 0.15` (target 15%; prototype is 38.7%). If this proves unattainable without Newton or carry expansion, document why in the worker report and propose a precise next step (e.g., propose ADR-023 amendment) — do NOT silently downgrade the threshold without manager review.

2. **`epssm` sweep**. Run R7 + warm-bubble + slice-trajectory at `epssm ∈ {0.0, 0.1, 0.3}`. Capture `proof_epssm_sweep.txt`. Report the chosen production default in the worker report and update ADR-023 accordingly.

3. **Nonhydrostatic `mu_continuity` un-gated in-scan**. The prototype gated mu off in the warm-bubble path. The production-grade operator must include the coupled `(w, mu, theta, phi)` mu update inside the acoustic scan body without destabilizing warm-bubble. If this is structurally impossible without carry expansion, document the obstacle and propose a path — but the default expectation is that ADR-023's claim of "no carry expansion" survives.

4. **Prototype-grade stabilization heuristics removed**. The prototype used "reduced vertical acoustic pressure coupling, calibrated buoyancy scale, and small nonlinear updraft drag". The production-grade operator must derive equivalent stabilization from MPAS off-centering (`epssm`) + WRF-compatible discretization, OR document why each heuristic survives in production with a citation.

5. **Post-solve replacement order documented (F9)**. Add a docstring + comment block in `acoustic_wrf.py` specifying the order of `(w, theta, ph_perturbation, mu_perturbation, p_perturbation, al, alt)` updates per substep. Add a new test that asserts this order is honored by `vertical_acoustic_update`.

6. **Carry/locals/scratch documentation (F7)**. Add a docstring block to `AcousticScanCarry` distinguishing:
   - Public carry (the 6-leaf form — unchanged)
   - Per-substep locals (named transient state in the scan body)
   - Solver scratch (tridiagonal coefficients consumed within `_calc_coef_w`)
   Add a test verifying no scan-leaked state.

7. **All prior tests still GREEN**:
   - `pytest tests/test_m6x_vertical_acoustic_oracle.py -v` → 3 passed
   - `pytest tests/test_m6x_adr023_column_solver.py -v` → 4 passed
   - `pytest tests/test_m6x_c2_acoustic.py -v` → 8 passed
   - `pytest tests/test_m6x_mpas_column_slice_oracle.py -v` → 4 passed (note: the `test_adr023_operator_matches_slice_within_tolerance` test currently allows 50% — once production drives RMSE <15%, tighten the tolerance in the new gate test, not the existing one)

8. **Warm bubble preserved**. Run `scripts/m6_warm_bubble_test.py`; output to `proof_warm_bubble_production.txt`. Verdict must remain `PASS_WARM_BUBBLE_600S` with `w_max ∈ [5, 10]` m/s at 600 s. If the prototype's heuristics were load-bearing for this gate, the production-grade derivation must restore the same behavior.

9. **Transfer audit clean**. `pytest tests/test_m3_transfer_audit.py` PASS. `proof_transfer_audit.txt`.

10. **Profiler artifact**. Capture HLO + launch count for the production `vertical_acoustic_update`. `proof_launch_count_production.txt`. If launch count grew significantly vs prototype 20, document why.

11. **ADR-023 status update**. Fold F2/F5/F7/F9 closures inline; mark F1/F3/F4/F10 with closure citations. The reviewer (next sprint) will move PROPOSED → ACCEPTED based on this sprint's evidence.

12. **Worker report** at `worker-report.md`. Must include: summary, derivation of replaced heuristics (with WRF/MPAS line citations), epssm sweep results table, trajectory RMSE evolution, files changed, commands run, proof objects, risks, handoff.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_prod
pytest tests/test_m6x_adr023_production_grade.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_production_gate.txt
pytest tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_no_regression.txt
python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_warm_bubble_production.json | tee .agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_warm_bubble_production.txt
pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_transfer_audit.txt
# Then: HLO/launch-count probe (replicate prototype's approach in src/gpuwrf/dynamics/vertical_implicit_solver.py)
```

## Performance Metrics

- Vertical operator launch count: target ≤ 20 (prototype baseline). If grows, document.
- No host/device transfer in the timestep loop. Binding.

## Proof Object

- `.agent/sprints/.../proof_production_gate.txt` — new acceptance-ladder gate tests
- `.agent/sprints/.../proof_no_regression.txt` — 19 tests (3+4+8+4) pass
- `.agent/sprints/.../proof_warm_bubble_production.txt` + `.json` — warm-bubble verdict
- `.agent/sprints/.../proof_transfer_audit.txt`
- `.agent/sprints/.../proof_epssm_sweep.txt`
- `.agent/sprints/.../proof_launch_count_production.txt`
- `.agent/sprints/.../worker-report.md`
- Updated `.agent/decisions/ADR-023-conservative-column-solver.md`
- Updated `src/gpuwrf/dynamics/acoustic_wrf.py` + `vertical_implicit_solver.py`
- New `tests/test_m6x_adr023_production_grade.py`

Time budget: **6-10 hours**. Per user "one large sprint" pattern.

## Risks

- **Trajectory RMSE < 15% may not be achievable** with the linearized v0 alone. If the operator-grade limit is the linearization, the worker should NOT silently add a Newton outer (forbidden); instead document and propose ADR-023 amendment for next sprint. Manager will decide.
- **`mu_continuity` un-gated may destabilize warm-bubble** if the prototype's gating was load-bearing. Trace the destabilization cause; if mu coupling is fundamentally incompatible with the small carry, this is critical evidence (likely triggers fallback to ADR-021 per ADR-023 fallback trigger).
- **`epssm` sweep result may flip the default**. If `epssm = 0.3` clearly outperforms 0.1 on slice RMSE, the production default should update.
- **MPAS slice oracle was a symbolic port** (not a built MPAS executable). Trajectory RMSE compared to the symbolic slice may differ from a true MPAS-binary comparison. Note this caveat; do not extrapolate.
- **Spec-gaming**: M5 pattern. Verifiability triple — actual nm-equivalent symbols, non-clipped coefficients, non-vacuous tolerances. Every numerical claim cites a line.

## Handoff Requirements

When all proof_*.txt are on disk and worker-report.md is committed on branch `worker/gpt/m6x-adr023-production-grade`, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr023-production-grade / codex] exit=<ec>`.

## Failure modes the manager will reject

- Silently downgrading the 15% trajectory RMSE threshold to make a test pass.
- Modifying the analytic oracle, MPAS slice oracle, or their test files.
- Adding fields to `AcousticScanCarry`.
- Adding a Newton outer loop without ADR amendment.
- Host transfer regression.
- Heuristic stabilization without WRF/MPAS-cited derivation.
- Updating ADR-023 status from PROPOSED to ACCEPTED yourself — that's the reviewer's call in the follow-up.
