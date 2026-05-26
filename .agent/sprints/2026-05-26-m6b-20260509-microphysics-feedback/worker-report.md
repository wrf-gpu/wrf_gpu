Summary: Fixed the 20260509 Thompson feedback failure by preventing Thompson from acting on thermodynamically invalid columns and by adding operational coupling guards that keep nonfinite/out-of-range moisture and nonfinite boundary replay values out of the next physics step. The original step-11 `qc=3.2233444e7 kg/kg` feedback no longer occurs; the 360-step 20260509 replay has theta in `[290.3363, 500.3132] K` and `qc_max=0.0082365 kg/kg`.

Files changed:
- `src/gpuwrf/physics/thompson_column.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6b_20260509_microphysics_fix.py`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/worker-report.md`

Commands run and output:
- `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .../baseline/`
  Output: `status: MATH:coftz`; proof reproduced `theta=2444869763072.0`, `qc=32233444.0` at step 11.
- `taskset -c 0-3 python scripts/diagnostic_limiter_activation_tracker.py --output .../`
  Output: failed as contracted command is missing required `--input`; rerun with `--input .../baseline/proof_first_bad_step_tracer.json` wrote `proof_limiter_activation_tracker.json`.
- WRF reference extraction from `wrfout_d02_2026-05-09_19:00:00`
  Output: same cell WRF `QCLOUD=0.0`, theta `348.160400390625 K`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2`
  Output: `status: PASS`, `final_max_abs_delta: 0.0`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
  Output: `status: PASS`, `final_max_abs_delta: 0.0`.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
  Output summary: `passed: true`, `outcome: SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, kill gate passed, tiers `column/golden/patch16=true`.
- `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .../fixed/`
  Output: `status: IC-SPECIFIC`; proof has `first_violation: null`.
- `taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .../fixed_521/`
  Output: `status: OK`, `verdict: NAMED-FIX:boundary_application`.
- `taskset -c 0-3 pytest tests/test_m6b_20260509_microphysics_fix.py -v`
  Output: `2 passed in 3.11s`.

Proof objects produced:
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/proof_wrf_microphysics_reference.json`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/proof_limiter_activation_tracker.json`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/proof_20260509_domain_extrema.json`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/baseline/`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/fixed/`
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/fixed_521/`

Risks:
- The fix includes an operational projection guard. It prevents nonfinite and moisture-out-of-range values from entering physics/boundary coupling, but it is not a source-equation correction.
- The 20260509 replay still shows finite but physically absurd pressure/wind excursions in some dynamics/boundary fields. Theta and qc acceptance pass, but a follow-up boundary/dynamics audit is still needed before interpreting broader forecast quality.

Handoff:
- objective: eliminate the 20260509 microphysics-theta feedback without modifying dynamics core.
- files changed: listed above.
- commands run: listed above.
- proof objects produced: listed above.
- unresolved risks: operational guard masks bad coupling values; pressure/wind dynamics remain suspect.
- next decision needed: dispatch the recommended boundary-forcing/dynamics audit if M6b requires physically interpretable p/u/v/w, not just bounded theta/qc.
