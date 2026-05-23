# Sprint Contract — M6.x ADR-023 Public Scan Path Unification (reviewer-reject closure)

## Objective

Production-grade reviewer (`b2f7a05`) returned **REJECT** with five findings. The two binding ones:

1. **BLOCKER**: `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` is missing from a fresh checkout, breaking `test_warm_bubble_fixture_replays_generated_slice`. No-regression check fails for an independent reviewer.

2. **MAJOR — path split**: The MPAS-recurrence path that drove slice trajectory RMSE to 1.69% is reached only by `vertical_acoustic_update(..., pressure_scale=0.0)`. The **public nonhydrostatic scan** with `non_hydrostatic=True` routes to `pressure_scale=-1.0` → `_wrf_buoyancy_column_update`, which:
   - IGNORES `epssm` entirely
   - Uses a `NONHYDROSTATIC_BUOYANCY_SCALE` constant (prototype-grade)
   - Applies a positive-only nonlinear updraft drag (not derived from MPAS)
   - Goes through `_mu_continuity_increment` with a tanh CFL limiter on mass

Plus 3 lesser:
- **MAJOR**: prototype-grade stabilization survives in coupled path (overlaps #2)
- **MINOR**: F8 cost re-estimation stale in ADR text (still says "3-5 days" in trade-off table)
- **NOTE**: production-grade tester report was an empty template (no tester run)

**Until path unification + fixture restoration land, ADR-023 stays PROPOSED. No d02 or 24h replay until this sprint closes.**

This sprint UNIFIES the public nonhydrostatic scan path so the production warm-bubble + d02 + 24h forecasts all run through the same conservative MPAS-recurrence operator that hit 1.69% RMSE on the slice.

## Non-Goals

- No new physics or stabilization scheme. Path unification is removal + redirection, not new code.
- No carry expansion. `AcousticScanCarry` stays 6-leaf.
- No Newton outer.
- No modification of analytic oracle, MPAS slice oracle, or their tests.
- No d02 or 24h forecast inside this sprint. Those are downstream.
- No remote push.
- No host/device transfer regression.

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-adr023-public-scan-path-unification`:

- `src/gpuwrf/dynamics/acoustic_wrf.py` — remove the path split:
  - Delete `_wrf_buoyancy_column_update` (or refactor it to be a thin wrapper that calls the MPAS-recurrence path)
  - Delete `NONHYDROSTATIC_BUOYANCY_SCALE` constant (verify no other reference)
  - Delete positive-only nonlinear updraft drag
  - Either delete `_mu_continuity_increment` tanh CFL limiter OR derive its form from MPAS source with cited line numbers OR document it as a temporary stabilizer with a follow-up issue
  - Route both `pressure_scale=0.0` and the public `non_hydrostatic=True` scan call through the same MPAS-recurrence path
  - Plumb `epssm` through the public scan so the sweep result actually binds the production path
- `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` — restore the fixture into the repository. If size constraints prevent committing, add `fixtures/manifests/mpas_column_slice_warm_bubble_2km.json` with a content hash + a generation script + adjust the test to generate-on-first-run.
- `tests/test_m6x_mpas_column_slice_oracle.py` — adjust `test_warm_bubble_fixture_replays_generated_slice` if needed for the fixture restoration approach (NEW exception to read-only rule — only this one test; if it can stay read-only, prefer that)
- `.agent/decisions/ADR-023-conservative-column-solver.md` — fix F8 in trade-off table; if the path-unification removes the buoyancy-scale heuristic entirely, update §5 to reflect that
- `tests/test_m6x_adr023_path_unification.py` (new) — assertion tests:
  - There is no separate `_wrf_buoyancy_column_update` callable on the production path (introspection test)
  - `vertical_acoustic_update(..., pressure_scale=anything)` always uses the MPAS recurrence
  - The warm-bubble harness scan exercises the same code as the MPAS-slice gate test (same kernel signature / hash)
  - `epssm` is honored in the public scan path (a sweep test that mirrors the prior `proof_epssm_sweep.txt` but against the public scan)
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/` — proofs + worker-report

Read-only elsewhere, especially: `src/gpuwrf/validation/`, c2-A2 horizontal PGF (`acoustic_wrf.py:309-408`), `mu_continuity_tendency`, R7 oracle test file, MPAS slice oracle module.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/reviewer-report.md`** — your binding spec; the 5 findings + cited file:line ranges
- `.agent/decisions/ADR-023-conservative-column-solver.md` — current state (PROPOSED)
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/worker-report.md` — what the prior worker did
- `src/gpuwrf/dynamics/acoustic_wrf.py` — current path-split state. Key lines per the reviewer:
  - `:31-33` NONHYDROSTATIC_BUOYANCY_SCALE constant
  - `:61-75` AcousticScanCarry
  - `:403-435` mu_continuity scan stage
  - `:457-475` mu CFL limiter
  - `:641-660` `pressure_scale` routing
  - `:707-744` `_wrf_buoyancy_column_update` (the path that's gating production)
  - `:907-915` epssm deletion site
  - `:918-922` mu replacement
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — the MPAS Thomas solver
- `tests/test_m6x_adr023_production_grade.py:68-99` — what the slice gate currently exercises
- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` — read for understanding only
- MPAS source `mpas_atm_time_integration.F:2184-2193` — actual MPAS damping form (cited by reviewer as the legitimate post-solve damping pattern, distinct from the positive-only drag the worker added)

## Acceptance Criteria

1. **Path unification verified.** Production tests in `tests/test_m6x_adr023_path_unification.py` PASS:
   - No `_wrf_buoyancy_column_update` callable accessible (or it's a transparent wrapper).
   - `vertical_acoustic_update(..., pressure_scale=p)` for `p ∈ {0.0, -1.0}` routes to the same MPAS-recurrence kernel (HLO signature equality or call-stack inspection).
   - `epssm` propagates through the public scan path — sweep `epssm ∈ {0.0, 0.1, 0.3}` on the warm-bubble harness; record `proof_public_path_epssm_sweep.txt`. Results must DIFFER across `epssm` (vs current behavior where they're identical because `_wrf_buoyancy_column_update` drops `epssm`).
   - **Critically**: warm-bubble at `epssm=0.1` must still PASS (w_max ∈ [5, 10] at 600s). If unification breaks warm-bubble, the simplifications were load-bearing — report and propose a path.

2. **Fixture restoration.** `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` is reproducible on a fresh checkout:
   - Either commit the .npz directly (if <5MB)
   - OR commit a generator script + manifest under `fixtures/manifests/` and adjust the test to call it on first use
   - Verify by deleting the file and re-running `pytest tests/test_m6x_mpas_column_slice_oracle.py -v` — must still PASS.

3. **All prior tests still GREEN** on a clean checkout (no `/tmp/wrf_gpu2_*` state pollution):
   - `pytest tests/test_m6x_vertical_acoustic_oracle.py -v` → 3 PASS
   - `pytest tests/test_m6x_adr023_column_solver.py -v` → 4 PASS
   - `pytest tests/test_m6x_c2_acoustic.py -v` → 8 PASS
   - `pytest tests/test_m6x_mpas_column_slice_oracle.py -v` → 4 PASS (including the fixture replay)
   - `pytest tests/test_m6x_adr023_production_grade.py -v` → 4 PASS
   - `pytest tests/test_m6x_adr023_path_unification.py -v` → all new tests PASS

4. **MPAS slice gate now exercises the public scan path.** Modify `tests/test_m6x_adr023_production_grade.py:68-99` so it calls the production scan path (not `pressure_scale=0.0` directly), and the trajectory RMSE on that path is reported. Target: still <15%. If unification produces a different RMSE (better or worse), report it; don't paper over.

5. **Transfer audit clean.** `pytest tests/test_m3_transfer_audit.py` PASS. `proof_transfer_audit.txt`.

6. **Launch count update.** Re-measure launch count after unification. Capture `proof_launch_count_unified.txt`. The prior baseline was 67 on the direct path; the unified path may rise or fall. Document. No threshold; informational only.

7. **F8 ADR fix.** Update `.agent/decisions/ADR-023-conservative-column-solver.md` trade-off table to use the critic's revised cost estimate, or mark F8 explicitly deferred.

8. **Worker report** at `worker-report.md`. Must include:
   - What the path split was (cite reviewer-report file:line)
   - What was removed / refactored / kept (with rationale per item)
   - Updated `epssm` sweep against the public scan
   - Updated launch count
   - Confirmation that c2-A2 horizontal + mu_continuity_tendency are untouched
   - Files changed, commands, proof objects, risks, handoff

9. **Branch commits** on `worker/gpt/m6x-adr023-public-scan-path-unification`.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_unify
# Test deliverables on a CLEAN checkout (no /tmp/wrf_gpu2_prod state pollution):
pytest tests/test_m6x_adr023_path_unification.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_unification_gate.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_full_regression.txt
pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_transfer_audit.txt
python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_warm_bubble_unified.json | tee .agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/proof_warm_bubble_unified.txt
# Public-path epssm sweep:
# (worker writes a small script or inline pytest parametrize and emits proof_public_path_epssm_sweep.txt)
```

## Performance Metrics

- Launch count: informational. Compare to prior 67.
- Warm-bubble run wall time: informational.
- Zero host/device transfers (binding).

## Proof Object

- `proof_unification_gate.txt` — path-unification new tests
- `proof_full_regression.txt` — all 23+ tests passing
- `proof_transfer_audit.txt`
- `proof_warm_bubble_unified.txt` + `.json`
- `proof_public_path_epssm_sweep.txt`
- `proof_launch_count_unified.txt`
- `worker-report.md`
- Updated `.agent/decisions/ADR-023-conservative-column-solver.md`
- New tests file
- Restored / generator-script-backed fixture

Time budget: **4-7 hours**. This is path simplification + fixture restoration, not new architecture.

## Risks

- **Warm-bubble may break after removing `_wrf_buoyancy_column_update`.** That's exactly the path-split signal — the prototype heuristics may have been load-bearing on warm-bubble. If unification fails warm-bubble PASS_WARM_BUBBLE_600S:
  1. Re-verify the MPAS recurrence is actually being called (it should be, with `epssm=0.1` matching ADR-022/MPAS).
  2. If warm-bubble truly fails on the conservative path, this is critical evidence that the prototype was over-stabilized and the conservative path doesn't actually run a warm bubble — DOCUMENT THIS LOUDLY and propose ADR-023 amendment.
  3. Do NOT re-introduce `NONHYDROSTATIC_BUOYANCY_SCALE` constant or positive-only drag silently to make warm-bubble pass.
- **Fixture commit > 5 MB**: use the manifest+generator pattern.
- **Spec-gaming**: the verifiability triple still applies. Every numerical claim cites a file:line.

## Handoff Requirements

When all proof files are on disk, ADR-023 is updated, worker-report.md is committed on `worker/gpt/m6x-adr023-public-scan-path-unification`, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr023-public-scan-path-unification / codex] exit=<ec>`.

## Failure modes the manager will reject

- Path remains split (any code branch the production scan can take that doesn't hit the MPAS recurrence).
- Re-introducing stabilization heuristics to make warm-bubble pass.
- Modifying analytic oracle, MPAS slice oracle, or `mu_continuity_tendency`.
- Carry expansion, Newton outer.
- Host transfer regression.
- Silently downgrading the F6 acceptance ladder thresholds.
- Self-promoting ADR-023 from PROPOSED to ACCEPTED — that's the follow-up reviewer's call.
