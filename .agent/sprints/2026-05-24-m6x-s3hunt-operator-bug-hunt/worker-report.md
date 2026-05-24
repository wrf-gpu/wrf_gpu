# Worker Report

## Summary

Summary: Implemented the M6.x S3-hunt diagnostic-only sanitizer-bypass replay and A/B harness. No production operator fix was made. Verdict is `NO-BUG-LOCALIZED`: the first sanitizer-off nonfinite reproduces at step 2, localizes to post-recurrence dycore output, and none of the seven one-suspect A/B buckets moves the first nonfinite beyond step 2 or reaches the 10-step sanitizer-off acceptance bar.

## Files Changed

- `scripts/diagnostic_first_bad_step_tracer.py` — new shared diagnostic helper for sanitizer-off replay, stage localization, A/B patch contexts, and coefficient sanity dump.
- `scripts/m6_d02_short_replay_sanitizer_off.py` — new Stage 1 CLI proof writer.
- `scripts/m6_bughunt_ab_toggle.py` — new Stage 2/3 A/B and coefficient proof writer.
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_trace.json`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_log.txt`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_ab_toggles.json`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_ab_log.txt`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_column_coefficients.json`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_no_regression.txt`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md`
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/worker-report.md`

No edits were made to `src/gpuwrf/dynamics/acoustic_wrf.py`, `src/gpuwrf/dynamics/vertical_implicit_solver.py`, `src/gpuwrf/integration/d02_replay.py`, governance files, reviewer/tester/manager reports, or goal files.

## Commands Run

- `python -m py_compile scripts/diagnostic_first_bad_step_tracer.py scripts/m6_d02_short_replay_sanitizer_off.py scripts/m6_bughunt_ab_toggle.py`
  Output: exit 0.
- `python scripts/m6_d02_short_replay_sanitizer_off.py --steps 10 --output .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_trace.json | tee .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_log.txt`
  Output: exit 0. Prefixes 1/2/5/10 ran with abort on first nonfinite. First guard-limit step was 1; first nonfinite step was 2; 10-step sanitizer-off acceptance was false.
- `python scripts/m6_bughunt_ab_toggle.py --output .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_ab_toggles.json | tee .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_ab_log.txt`
  Output: exit 0 after one reporting-bug retry. Seven suspect buckets ran; every variant reported first nonfinite step 2 and `ten_step_sanitize_off_acceptance=false`.
- `python -m json.tool .../proof_first_bad_trace.json`, `proof_ab_toggles.json`, `proof_column_coefficients.json`
  Output: all OK.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_no_regression.txt`
  Output: `54 passed in 792.58s (0:13:12)`.

## Proof Objects

- `proof_first_bad_trace.json`: first nonfinite step 2; field `u`; location `i_j_k=[0,0,0]` in the localized dycore rerun; value `nan`; previous value `-0.3547927141189575`; stage `post-recurrence`. Step 1 already has guard-limit hits.
- `proof_ab_toggles.json`: seven suspect buckets, all first nonfinite step 2; no variant reached 10-step sanitizer-off acceptance.
- `proof_column_coefficients.json`: center-column coefficients finite; metric finite-positive; `tri_b` positive and nonzero; weak diagonal dominance true.
- `proof_no_regression.txt`: mandatory regression suite passed.
- `verdict.md`: `NO-BUG-LOCALIZED`.

## Risks

- The stage localization is coarse inside the vertical recurrence: it distinguishes dycore post-recurrence from mu/physics/boundary, but it does not expose every interior tridiagonal solve local array.
- Step 1 guard-limit hits are already severe, so the sanitizer-off state is outside physical bounds before the first nonfinite at step 2.
- No single local toggle dominated the failure; the remaining risk is a broader recurrence/state-shape design defect rather than a one-line operator bug.
- Diagnostic scripts intentionally use host reads for proof capture; no GPU performance claim is made from these diagnostic runs.

## Handoff

Objective completed. Files changed are confined to the contract-owned scripts and sprint folder. Next decision needed: manager should treat this as `NO-BUG-LOCALIZED` and move to `M6-DYCORE-BLOCKER-MEMO` or a separate design sprint for the broader dycore recurrence/state architecture. Per sprint contract, no operator fix and no remote push were performed.
