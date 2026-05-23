# Worker Report

Summary: Implemented the M6.x vertical-acoustic analytic oracle as a closed-form NumPy mode generator and added the requested red pytest oracle against the current `acoustic_wrf` scan. The focused proof shows all three new tests failing on the current implementation, which does not contain the WRF `advance_w` vertical `w`/`phi` implicit acoustic solve or an exposed `vertical_acoustic_update` hook.

## Objective

Provide a non-tautological 1-D vertical linear acoustic-gravity oracle that future ADR-021/ADR-022 pivot work can turn green.

## Files Changed

- `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py`
- `tests/test_m6x_vertical_acoustic_oracle.py`
- `.agent/sprints/2026-05-23-m6x-vertical-acoustic-analytic-oracle/proof.txt`
- `.agent/sprints/2026-05-23-m6x-vertical-acoustic-analytic-oracle/worker-report.md`

## Derivation

The oracle cites Skamarock et al. 2008 WRF Technical Note section 3.2 and derives the 1-D flat-column limit in the module header. The linearized equations are `d_t w = -(1/rho0) d_z p' + b`, `d_t p' = -rho0 c_s^2 d_z w`, and `d_t b = -N^2 w`, with `b = g theta' / theta0` and `c_s = sqrt(gamma R_d T_base)`. A normal mode gives `omega^2 = c_s^2 k_z^2 + N^2`; the module returns closed-form `w`, `ph_perturbation`, `theta_perturbation`, `period_s`, and `decay_rate_inv_s`.

## Tests Added

- `test_linear_acoustic_period_matches_dispersion_relation`: initializes a single vertical mode and checks quarter/half/full-period modal phase against the dispersion relation. Current failure: quarter-period modal amplitude remains `1.0` instead of `~0`.
- `test_no_drift_in_hydrostatic_rest_state`: verifies zero `w` and zero `ph_perturbation` drift over 1000 substeps, then fails because there is no operator-level `vertical_acoustic_update` hook to certify the WRF `advance_w` path.
- `test_amplitude_decay_within_2pct_of_analytic`: checks the zero-decay signed modal amplitude at half-period. Current failure: measured signed amplitude is `1.0`, expected `-1.0`.

## Commands Run

- `git switch -c worker/gpt/m6x-vertical-acoustic-analytic-oracle`
  - Output: `Switched to a new branch 'worker/gpt/m6x-vertical-acoustic-analytic-oracle'`
- `pytest tests/test_m6x_vertical_acoustic_oracle.py -v | tee .agent/sprints/2026-05-23-m6x-vertical-acoustic-analytic-oracle/proof.txt`
  - Output summary: `3 failed in 7.77s`.
  - Failing tests: all three required tests in `tests/test_m6x_vertical_acoustic_oracle.py`.
- `pytest -q`
  - Exit code: `1`.
  - Output summary: `29 failed, 506 passed, 1 skipped in 708.26s`.
  - New intentional failures: the three vertical-acoustic oracle tests.
  - Other failures are in pre-existing tests outside this sprint's ownership, including missing external fixture files, subprocess `ModuleNotFoundError: No module named 'gpuwrf'` in M2 bench comparisons, existing M3/M4/M5 numeric or artifact assertions, and missing Gen2/AIFS external paths.
- `git restore <test-generated out-of-scope artifact files>`
  - Output: none. This restored artifacts mutated by `pytest -q` outside the sprint write scope.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-vertical-acoustic-analytic-oracle/proof.txt`
- `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py`
- `tests/test_m6x_vertical_acoustic_oracle.py`

## Risks

- The hydrostatic-rest test currently fails on missing operator exposure after the zero-drift assertions pass. This is deliberate red evidence for the absent operator-level WRF `advance_w` path, but the pivot sprint should replace the fallback scan adapter with the ratified production update.
- `pytest -q` is not green because this branch intentionally adds failing tests and the repository already has unrelated failing tests requiring external data/artifacts or prior milestone fixes.

## Handoff

Objective complete: the analytic oracle and red proof are on disk. Next decision needed: ADR-021/ADR-022 pivot implementation must add the vertical acoustic `w`/`phi` update and make these oracle tests pass without weakening tolerances.
