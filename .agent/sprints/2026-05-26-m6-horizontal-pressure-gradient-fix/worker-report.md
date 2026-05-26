# Worker Report - M6 horizontal_pressure_gradient Fix

Summary: Implemented the HPG algebraic fix in `src/gpuwrf/dynamics/acoustic_wrf.py`: WRF `advance_uv` applies `dpxy` to mass-coupled small-step momentum, while GPUWRF `State.u/v` are velocities, so the returned velocity tendencies now divide by the face dry-column mass `(c1h*muu+c2h)` / `(c1h*muv+c2h)`. Removed the old V-only suppression behavior in operational mode; the legacy `_m6b_acoustic_tendencies` symbol remains as an identity shim only because existing diagnostic scripts import it. Added a step-49 regression test. Validation is partial: HPG fixture regression, B6 parity, real-IC 10-step compare, and required pytests pass; the guard-disabled replay still fails the contract stability gate, now at theta step 18 inside `acoustic`.

## Files changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6_horizontal_pressure_gradient_fix.py`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/proof_wrf_fortran_crosscheck.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/validation_*.txt`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/{baseline,fixed}/proof_*.json`

External ignored fixture:
- `data/fixtures/m6-horizontal-pressure-gradient-fix/step49_input_state.npz` (not committed; binary fixture under ignored `data/` per repo rule)

## Commands run and output

`taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 50 --output .agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/baseline/`
Output captured in `validation_baseline.txt`: `status OK`; first explosive step `p_perturbation`, step `49`, cell `[32, 52, 40]`, value `-314176.084627652`; first operator `horizontal_pressure_gradient`.

`python <inline extraction/cross-check>`
Output: wrote `proof_baseline_reproduces.json` and external `step49_input_state.npz`; baseline HPG output at bad cell before fix was `du_dt=-51.92551268047103`, `dv_dt=167.4927814184497`.

`pytest tests/test_m6x_c2_pgf.py tests/test_m6_horizontal_pressure_gradient_fix.py -v`
Output: `8 passed in 15.09s`.

`taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/fixed/`
Output captured in `validation_fixed_guard_disabled.txt`: command exit 0, but acceptance failed; first explosive field `theta`, step `18`, cell `[12, 30, 62]`, value `16207.9404296875`, ratio `23.154200613839286`; first operator `acoustic`.

`taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
Output captured in `validation_m6b6_coupled_step_compare.txt`: `passed true`; outcome `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`; diverging field count `0`.

`taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
Output captured in `validation_m6b_real_ic_operational_compare.txt`: `status PASS`; `final_max_abs_delta 0.0`; steps `10`.

`taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v`
Output captured in `validation_pytest_m6_guard_disabled_debug.txt`: `12 passed in 0.96s`.

`taskset -c 0-3 pytest tests/test_m6_horizontal_pressure_gradient_fix.py -v`
Output captured in `validation_pytest_m6_horizontal_pressure_gradient_fix.txt`: `1 passed in 12.10s`.

`python -m py_compile src/gpuwrf/dynamics/acoustic_wrf.py src/gpuwrf/runtime/operational_mode.py tests/test_m6_horizontal_pressure_gradient_fix.py`
Output captured in `validation_py_compile.txt`: no stdout/stderr, exit 0.

## Proof objects

- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/proof_wrf_fortran_crosscheck.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/baseline/proof_first_explosive_step.json`
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/fixed/proof_first_explosive_step.json`
- `data/fixtures/m6-horizontal-pressure-gradient-fix/step49_input_state.npz` (ignored binary fixture)

## Risks

- The specific HPG algebra is fixed and covered, but the sprint acceptance criterion "1h Canary with guards disabled" is not met. The next failure is theta at step 18 in `acoustic`, not the prior step-49 pressure blow-up.
- Operational mode now uses WRF-shaped HPG u/v tendencies instead of the old reduced advection V workaround. This is closer to `advance_uv`, but full guard-disabled stability still needs a follow-up acoustic/theta investigation.
- The `.npz` fixture is intentionally not committed; rerunning the extraction command is required if a fresh checkout needs that binary fixture.

## Handoff

Objective: replace the V-suppress workaround with a real `horizontal_pressure_gradient` algebra fix and prove the step-49 HPG fixture no longer emits nonphysical velocity tendencies.

Files changed: listed above.

Commands run: all validation commands from the contract were run; outputs are captured in the current sprint folder.

Proof objects produced: listed above.

Unresolved risks: guard-disabled replay still fails at theta step 18 inside `acoustic`.

Next decision needed: dispatch a follow-up focused on acoustic theta growth after the HPG correction, or revise the current sprint acceptance if the manager considers the HPG root cause closed and the step-18 theta failure a new blocker.
