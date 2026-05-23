# Reviewer Report - M6.x ADR-023 Production-Grade Reviewer

Role: reviewer
Branch: `reviewer/opus/m6x-adr023-production-grade-reviewer`
Reviewed worker branch/diff: `worker/gpt/m6x-adr023-production-grade` (`0a05159`) vs `9f19960`

## §1 Re-run Spot Checks

I re-ran the four required proof commands in `/tmp/wrf_gpu2_review_prod` and saved the raw outputs in this sprint folder.

- `pytest tests/test_m6x_adr023_production_grade.py -v` passed 4/4. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_production_gate.txt:9-14`.
- `pytest tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v` did **not** reproduce the worker's 19/19 result. It passed 18/19 and failed `test_warm_bubble_fixture_replays_generated_slice` because `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` is absent. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_no_regression.txt:12`, `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_no_regression.txt:32-41`, `tests/test_m6x_mpas_column_slice_oracle.py:163-177`.
- `python scripts/m6_warm_bubble_test.py --output /tmp/reviewer_wb.json` passed `PASS_WARM_BUBBLE_600S`, with `w_max=8.523914985976297` m/s at 600 s. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_warm_bubble.txt:9-15`.
- `pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v` passed 5/5. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_transfer_audit.txt:9-15`.

The mandatory reviewer confirmation in the contract therefore cannot be made as written: 4/4 production gate PASS, warm-bubble PASS, and 5/5 transfer audit PASS are confirmed, but 19/19 no-regression PASS is not confirmed.

## §2 Critic Findings Closure Audit

F1 - MPAS equivalence claim: **partially closed**. ADR-023 demotes equation-level MPAS equivalence to "family-level same conservative off-centered tridiagonal structure" and says the current slice is a validation rung, not binary equivalence. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:105`. The NumPy-only slice is present and independent (`src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py:1-4`, `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py:26`). However, the regression test that replays the generated fixture fails in this reviewer checkout because the fixture path is missing, so the slice proof is not reproducible here.

F2 - Newton-outer justification: **closed for v0 framing**. ADR-023 now frames v0 as a linearized MPAS/WRF-style acoustic-gravity solve and explicitly says nonlinear HEVI would need Newton machinery. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:31`, `.agent/decisions/ADR-023-conservative-column-solver.md:60`. I found no Newton outer loop in the touched production code.

F3 - R7 oracle red: **closed**. The no-regression rerun passed the three R7 tests. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_no_regression.txt:13-15`.

F4 - tridiagonal solver module path: **closed**. The ADR now points to `src/gpuwrf/dynamics/vertical_implicit_solver.py` and the implementation contains the Thomas default plus XLA alternative. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:153`, `src/gpuwrf/dynamics/vertical_implicit_solver.py:85-160`.

F5 - `epssm` default: **partially closed**. The sweep artifact selects `epssm=0.1` because `0.3` fails R7 even though it improves the slice RMSE. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_epssm_sweep.txt:1-38`. The caveat is important: the warm-bubble results are identical for all three `epssm` values in that proof, and the public nonhydrostatic scan path routes to `_wrf_buoyancy_column_update`, which deletes `epssm`. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:707-728`, `src/gpuwrf/dynamics/acoustic_wrf.py:907-915`.

F6 - staged Tier-4 acceptance ladder: **partially closed**. The new production-gate test covers the MPAS-slice trajectory target and the worker produced warm-bubble and transfer proof files. Evidence: `tests/test_m6x_adr023_production_grade.py:101-116`, `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_warm_bubble_production.txt:9-15`. It remains incomplete because the mandatory no-regression fixture replay fails here, and the next rungs (`1h d02`, `24h/72h Gen2`) remain future work as allowed by this sprint.

F7 - public carry vs locals vs scratch: **closed**. `AcousticScanCarry` remains the six-leaf form and the docstring now distinguishes public carry, per-substep locals, and solver scratch. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:61-75`, `tests/test_m6x_adr023_production_grade.py:148-160`.

F8 - cost re-estimation: **not closed in ADR text**. The ADR still contains the 3-5 day warm-bubble estimate in the trade-off table. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:116`. This is not a code blocker by itself, but it leaves the critic's schedule correction unresolved.

F9 - post-solve replacement order: **closed as documentation/test surface**. The ADR and code name the order, and the new test asserts the exported constant and docstrings. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:86-98`, `src/gpuwrf/dynamics/acoustic_wrf.py:35-43`, `tests/test_m6x_adr023_production_grade.py:148-160`.

F10 - performance/residency claims: **partially closed**. The transfer spot check passes and the worker supplied a launch-count proof. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_transfer_audit.txt:9-15`, `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_launch_count_production.txt:1-13`. The proof shows 67 kernel launches and 48 device-to-device memcpy calls, so optimization evidence is not acceptable for a speed claim.

## §3 Anti-Tautology Audit

The MPAS slice oracle is genuinely independent of the production JAX operator. It is NumPy-only, states that it does not call the local ADR-023 operator, and imports NumPy rather than JAX. Evidence: `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py:1-4`, `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py:24-27`. The production module does not import `gpuwrf.validation.mpas_oracles`; its dynamics imports are limited to contracts, damping, and the vertical solver. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:12-16`.

The R7 analytic oracle and MPAS slice oracle are file-disjoint from `acoustic_wrf.py`. The worker diff did not modify `src/gpuwrf/validation/`, `tests/test_m6x_vertical_acoustic_oracle.py`, `tests/test_m6x_mpas_column_slice_oracle.py`, `tests/test_m6x_adr023_column_solver.py`, or `tests/test_m6x_c2_acoustic.py`; `git diff --name-status 9f19960..0a05159 -- src/gpuwrf/validation tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py` returned no changed files.

The 1.69% trajectory RMSE is computed against the MPAS slice, not directly against itself: the production test calls `mpas_column_slice`, advances the local operator, and computes `_trajectory_rmse_fraction(c2_w, slice_result)`. Evidence: `tests/test_m6x_adr023_production_grade.py:68-99`, `tests/test_m6x_adr023_production_grade.py:101-104`.

The anti-tautology concern is the branch split, not direct oracle self-use. The improved MPAS-slice path is exercised by calling `vertical_acoustic_update(..., pressure_scale=0.0)` directly in the tests. Evidence: `tests/test_m6x_adr023_production_grade.py:82-91`. The public scan path used by the coupled warm-bubble test maps `non_hydrostatic=True` to `pressure_scale=-1.0`, which enters `_wrf_buoyancy_column_update`, not `_mpas_recurrence_vertical_update`. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:641-660`, `src/gpuwrf/dynamics/acoustic_wrf.py:707-744`, `src/gpuwrf/dynamics/acoustic_wrf.py:907-915`. That means the strongest MPAS-slice number is not yet proof that the production nonhydrostatic scan uses the same conservative recurrence that achieved it.

I do not see the exact M5-style "verifiability triple" pattern of invented symbols or vacuous tolerances. I do see a related risk: tests and profiling are passing on a code path that is not the default nonhydrostatic scan path.

## §4 Forbidden-Move Audit

No `AcousticScanCarry` expansion found. The carry still has exactly `("state", "previous_pressure", "al", "alt", "cqu", "cqv")`. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:74`, `tests/test_m6x_adr023_production_grade.py:158`.

No Newton outer loop found in the implementation. The default tridiagonal solver is the Thomas scan, and `solve_tridiagonal_xla` is left as an alternative. Evidence: `src/gpuwrf/dynamics/vertical_implicit_solver.py:85-160`.

No oracle module or oracle test modifications found in the worker diff. This satisfies the read-only oracle requirement.

I did not find worker self-promotion of ADR-023 from PROPOSED to ACCEPTED. The ADR remains PROPOSED and explicitly waits for reviewer concurrence. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:1-3`, `.agent/decisions/ADR-023-conservative-column-solver.md:141-143`.

The transfer-audit spot check passed and the launch proof reports zero HtoD and DtoH async copies for the profiled vertical operator. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_transfer_audit.txt:9-15`, `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_launch_count_production.txt:4-8`.

The c2-A2 horizontal PGF and `mu_continuity_tendency` were not directly rewritten in the diff. However, the scan now applies a separate `_mu_continuity_increment` limiter before replacing `mu`, so the coupled mass update behavior has changed even though `mu_continuity_tendency` itself remains intact. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:403-435`, `src/gpuwrf/dynamics/acoustic_wrf.py:457-475`, `src/gpuwrf/dynamics/acoustic_wrf.py:918-922`.

## §5 Open Risks, Findings, and Required Fixes

Severity-ranked findings:

1. **Blocker - mandatory no-regression spot check fails in reviewer checkout.** The contract requires confirmation of 19/19 no-regression PASS, but the independent rerun failed 1 test because `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` is missing. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/spot_no_regression.txt:32-41`, `tests/test_m6x_mpas_column_slice_oracle.py:163-177`. Required fix: restore the fixture in the repository's external-data contract path or make the test generate/locate it through a tracked manifest so a fresh reviewer checkout reproduces the worker proof without manual state from `/tmp/wrf_gpu2_prod`.

2. **Major - the MPAS-slice proof path is not the public nonhydrostatic scan path.** The production gate and profiler call `vertical_acoustic_update` directly with `pressure_scale=0.0`; the actual scan with default `non_hydrostatic=True` routes to `pressure_scale=-1.0` and a different `_wrf_buoyancy_column_update` path. Evidence: `tests/test_m6x_adr023_production_grade.py:82-91`, `src/gpuwrf/dynamics/acoustic_wrf.py:641-660`, `src/gpuwrf/dynamics/acoustic_wrf.py:707-744`, `src/gpuwrf/dynamics/acoustic_wrf.py:907-915`, `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_launch_count_production.txt:1-3`. Required fix: either wire the conservative MPAS/Thomas recurrence into the public nonhydrostatic scan path and test/profile that path, or narrow ADR-023's status so it is not called production-grade for the coupled scan.

3. **Major - prototype-grade stabilization still survives in the coupled warm-bubble path.** `_wrf_buoyancy_column_update` ignores `epssm`, applies a fixed `NONHYDROSTATIC_BUOYANCY_SCALE`, and uses a positive-updraft nonlinear drag. `_mu_continuity_increment` also applies a tanh CFL limiter to mass updates. Evidence: `src/gpuwrf/dynamics/acoustic_wrf.py:31-33`, `src/gpuwrf/dynamics/acoustic_wrf.py:707-744`, `src/gpuwrf/dynamics/acoustic_wrf.py:457-475`. The contract permits documenting surviving heuristics only with real WRF/MPAS-cited derivation; the current MPAS Rayleigh-block analogy is not enough because MPAS applies a `dss` damping block after the tridiagonal solve, not this positive-only drag. MPAS reference: `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2184-2193`. Required fix: remove these from the production path or explicitly classify them as temporary validation stabilizers with a separate acceptance gate before any ADR acceptance.

4. **Minor - F8 cost correction is still stale in ADR text.** The ADR says F8 is revised but still lists "3-5 days" in the trade-off table. Evidence: `.agent/decisions/ADR-023-conservative-column-solver.md:116`, `.agent/decisions/ADR-023-conservative-column-solver.md:159`. Required fix: update the ADR text to reflect the critic's revised estimate or mark F8 intentionally deferred.

5. **Note - production-grade tester report is an empty template.** Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade/tester-report.md:1-11`. I did not rely on it for judgment; I relied on worker proof files plus independent spot checks.

Performance risks:

- Launch count grew from prototype 20 to 67, exceeding the production sprint target of `<=20` while still being documented as required. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade/sprint-contract.md:92-95`, `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_launch_count_production.txt:4-13`. I do not treat this alone as a reject reason because the project hard rule is "no GPU optimization claim without profiler artifact," not "reject if launch count >20." It does block any speed claim and should be optimized before longer forecast rungs.
- The 48 device-to-device memcpy calls are not a host-transfer violation, but they are a real memory-pressure signal in the Thomas-scan path. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_launch_count_production.txt:6-10`.
- The selected default `epssm=0.1` is reasonable if R7 is the priority: `epssm=0.3` gives lower slice RMSE but fails the R7 tolerance. Evidence: `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_epssm_sweep.txt:17-33`. It is not yet proven for the public nonhydrostatic warm-bubble/d02 path because that path currently ignores `epssm`.

## §6 Verdict

REJECT

Decision: Reject

This is not a transfer-audit rejection: host-transfer evidence is clean. It is a reproducibility and correctness-path rejection. The mandatory no-regression spot check failed in the reviewer checkout, and the most important production-grade claim is split across different branches of `vertical_acoustic_update`: the MPAS-slice proof exercises `pressure_scale=0.0`, while the public nonhydrostatic scan uses a separate warm-bubble limiter path. Until the fixture reproducibility and path-split issues are fixed, ADR-023 should remain PROPOSED and should not be promoted to ACCEPTED on main.

## §7 Open Questions for the Manager

1. Should the next fix sprint be scoped narrowly to the missing MPAS slice fixture plus public-path test wiring, or should it also remove the positive-updraft drag and mass CFL limiter before ADR-023 can be reconsidered?

2. Is ADR-023 intended to accept a temporary two-path implementation, where the MPAS recurrence proves a column-slice rung but the coupled warm-bubble path remains stabilized by a simpler buoyancy update? If yes, the ADR needs to say that explicitly and stop calling the current coupled path production-grade.

3. Should the `1h d02` replay start before the public nonhydrostatic scan path is unified with the MPAS-slice path? My recommendation is no; otherwise the next rung may validate the limiter path rather than the conservative column recurrence.

4. Should a profiler-bot sprint optimize the Thomas path now, or wait until the correctness path is unified? My recommendation is wait; launch optimization before path unification may optimize the wrong branch.
