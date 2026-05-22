# Worker Report — M6.x-A2 Bug-Hunt Fix-Hint Application

## objective

Apply the ordered bug-hunt sequence from the sprint contract in `/tmp/wrf_gpu2_m6x`:

1. add the missing `GRAVITY_M_S2` factor to PH evolution;
2. remove the asymmetric pressure relaxation/mass mask and audit inverse-density limiting;
3. if still red, update `mu` inside the acoustic small-step scan.

## files changed

- `src/gpuwrf/dynamics/acoustic.py`
  - Added `GRAVITY_M_S2` to the PH update.
  - Removed `PRESSURE_IMPLICIT_RELAXATION`.
  - Removed `_vertical_implicit_mass_weight` from the pressure update path.
  - Relaxed `MAX_INVERSE_DENSITY` from `0.02` to `5.0` after the fully uncapped variant failed the 6h probe.
  - Advanced `mu` inside `forward_backward_acoustic` using `compute_mu_tendency` on small-step-evolved velocity fields.
- `src/gpuwrf/dynamics/rk3.py`
  - Added `include_mu_continuity` to avoid applying horizontal dry-mass continuity once at RK stage entry when the following acoustic scan applies it every substep.
- `tests/test_m6x_dycore_completion.py`
  - Updated the pressure update oracle for the removed `0.05` relaxation factor.
  - Added the PH gravity-factor oracle.
  - Added a column-neutral acoustic conservation oracle.
- `artifacts/m6/performance/m6x_a2_fix1_6h_direct_probe.json`
- `artifacts/m6/performance/m6x_a2_fix2_6h_direct_probe.json`
- `artifacts/m6/performance/m6x_a2_fix2b_6h_direct_probe.json`
- `artifacts/m6/performance/m6x_a2_fix3_6h_direct_probe.json`

## commands run

- `PYTHONPATH=src pytest -q tests/test_m6x_dycore_completion.py tests/test_m6x_mu_continuity.py tests/test_m6x_cfl_diagnostic.py`
  - PASS after FIX #1: `10 passed`
  - PASS after FIX #2 / FIX #2b: `11 passed`
- `PYTHONPATH=src pytest -q tests/test_m6x_dycore_completion.py tests/test_m6x_mu_continuity.py tests/test_m6x_cfl_diagnostic.py tests/test_m4_acoustic.py`
  - PASS after FIX #3: `14 passed`
- `PYTHONPATH=src pytest -q tests/test_m6x_dycore_completion.py tests/test_m6x_mu_continuity.py tests/test_m6x_cfl_diagnostic.py tests/test_m4_acoustic.py tests/test_m4_tester_adversarial.py`
  - FAIL in pre-existing/unrelated M4 adversarial checks:
    - `test_tier1_artifact_pass_is_consistent_with_zero_error`
    - `test_dycore_advection_operator_is_NOT_what_tier1_checks`
- 6h direct probe after FIX #1.
- 6h direct probe after FIX #2 with uncapped inverse density.
- 6h direct probe after FIX #2b with `MAX_INVERSE_DENSITY = 5.0`.
- 6h direct probe after FIX #3.

## proof objects produced

- `artifacts/m6/performance/m6x_a2_fix1_6h_direct_probe.json`
  - FAIL: `sanitize_step_firing_rate = 0.7736111111111111`, final `mu = [1000, 120000]`, final `theta = [150, 550]`.
- `artifacts/m6/performance/m6x_a2_fix2_6h_direct_probe.json`
  - FAIL: uncapped inverse density made the run worse; `sanitize_step_firing_rate = 1.0`.
- `artifacts/m6/performance/m6x_a2_fix2b_6h_direct_probe.json`
  - FAIL: relaxed inverse-density cap still red; `sanitize_step_firing_rate = 0.999537037037037`.
- `artifacts/m6/performance/m6x_a2_fix3_6h_direct_probe.json`
  - FAIL: small-step `mu` integration did not recover stability; `sanitize_step_firing_rate = 0.999537037037037`, final `mu = [1000, 120000]`, final `theta = [150, 550]`.

## unresolved risks

- The three bug-hunt fixes do not stabilize the 6h coupled direct probe.
- Removing the pressure relaxation factor exposes a larger acoustic instability within the first simulated hour, even with `alpha <= 5.0`.
- The remaining `_vertical_implicit_weight` heuristic is still not the canonical Klemp-Skamarock tridiagonal vertical-implicit solve.
- The broader M4 adversarial test file is red for advection/artifact parity unrelated to this acoustic work.

## next decision needed

Invoke the c1 Klemp-Skamarock fallback contract. Per the sprint kill-gate, 24h probe, speedup rerun, ADR-007 PASS amendment, and ADR-015 were not attempted because the final 6h probe is red.
