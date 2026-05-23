# Worker Report

## Summary

Summary: Implemented the gated pressure-diagnose wiring fix for the WRF/base-state nonhydrostatic acoustic scan path. The nonhydrostatic path now preserves the MPAS recurrence's resident density-derived `p_perturbation` and refreshes only `al`/`alt`; hydrostatic and `base_state=None` legacy/smdiv behavior still use the diagnostic pressure replacement.

Before from Opus diagnostic: recurrence pressure `12.701 Pa` was overwritten to `4.37e-11 Pa`. After this patch, `probe_p_perturbation_survives_substep.json` reports recurrence `12.701117942007619 Pa`, carry `12.701117942007619 Pa`, absolute difference `0.0 Pa`, while diagnostic pressure on the same post-recurrence state remains `4.3655745685100555e-11 Pa`.

## Files Changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`: lines 821-822 gate initialization pressure replacement; lines 836-841 gate pre-vertical pressure source/replacement; lines 878-879 gate the post-vertical diagnostic overwrite.
- `tests/test_m6x_pressure_diagnose_wiring.py`: new nonhydrostatic survival test and hydrostatic behavior guard.
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/*`: proof artifacts and this report.

## Commands Run

- `pytest tests/test_m6x_pressure_diagnose_wiring.py -v 2>&1 | tee .../proof_wiring_gate.txt`
  - Output: `2 passed in 5.69s`.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py tests/test_m6x_adr023_path_unification.py -v 2>&1 | tee .../proof_no_regression.txt`
  - Output: `27 passed in 20.86s`.
- `pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v 2>&1 | tee .../proof_transfer_audit.txt`
  - Output: `5 passed in 2.69s`.
- `python scripts/m6_warm_bubble_test.py --output .../proof_warm_bubble_post_fix.json 2>&1 | tee .../proof_warm_bubble_post_fix.txt`
  - Output: finite to 600 s, `w_max_m_s=0.038687905089051594` at 600 s, verdict `FAIL_TARGETS_NOT_MET`.
- Direct probe generator for `probe_p_perturbation_survives_substep.json`
  - Output: post-recurrence and post-carry pressure both `12.701117942007619 Pa`; tolerance pass `true`.
- Warm-bubble bounds generator for `proof_warm_bubble_post_fix_bounds.json`
  - Output: final `theta_perturbation_max=1.5503358115530546 K`, `max|p_perturbation|=315.7192191584963 Pa`, `max|mu_perturbation|=86785.96188177825 Pa`.
- `git diff --check`
  - Output: clean.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_wiring_gate.txt`
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_warm_bubble_post_fix.json`
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_warm_bubble_post_fix.txt`
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_warm_bubble_post_fix_bounds.json`
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/probe_p_perturbation_survives_substep.json`

## Risks

- The architectural warm-bubble gap remains: `w_max` is still about `0.039 m/s` at 600 s, so this does not close the lift target.
- The temporary `_mu_continuity_increment` tanh limiter remains in place. The fix keeps theta and pressure perturbations physical in the 600 s probe, but mu perturbation still grows to about `86.8 kPa`, consistent with the known limiter/scratch-field issue.
- `diagnose_pressure_al_alt` itself still uses `theta_base` and omits `ph_perturbation` in `al`; this sprint intentionally did not rewrite it.

## Handoff

Objective: close the confirmed pressure-diagnose overwrite bug without changing the vertical recurrence or broader architecture. Files changed and proof objects are listed above. Unresolved risk is the ADR-023/ADR-021 architectural gap. Next decision needed: manager/reviewer should decide the separate architecture path; this branch only fixes the wiring bug.
