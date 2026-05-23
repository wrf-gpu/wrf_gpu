# Worker Report

Summary: Replaced the warm-bubble verdict from an unsourced `[5, 10] m/s` amplitude band with an operator-sanity gate. The integration loop and grid setup are unchanged. The new proof JSON reports signed/absolute `w`, theta/p/mu perturbation extrema, centroid, mu residual, physical-bound violations, R7/hydrostatic preconditions, and static anti-clamp warnings. ADR-024 records the policy change as PROPOSED.

## Objective

Implement Stage 1 of the critic's `CHANGE-THE-GATE` recommendation: keep warm bubble as an operator diagnostic, not an architecture-ratifying amplitude gate. Do not change production operators, oracles, c2-A2 PGF, or `mu_continuity_tendency`.

## Files Changed

- `scripts/m6_warm_bubble_test.py`
- `tests/test_m6x_warm_bubble_operator_sanity.py`
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md`
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_new_gate.txt`
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.json`
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.txt`
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/worker-report.md`

## Gate Change Rationale

The critic found the `[5, 10] m/s` target was not sourced for this pure-small-step Gaussian harness and that prior passes were contaminated by target-shaped stabilizers. Opus's diagnostic showed the old verdict hid the real failure mode: theta/p/mu blowup or limiter saturation can coexist with a small positive `w_max`. The new gate therefore reports amplitude but fails on finiteness, physical bounds, and hard anti-clamp detection.

## New Verdict Structure

Verdicts are now exactly:

- `PASS_OPERATOR_SANITY`
- `FAIL_FINITENESS`
- `FAIL_PHYSICAL_BOUNDS`
- `FAIL_ANTI_CLAMP_DETECTION`

The amplitude band is preserved only as `legacy_amplitude_band_m_s: {"range": [5.0, 10.0], "gate": false}`. Static scan hard-fails target-shaped tanh clamps, positive-only `w` clipping, theta-target clipping, and lift/updraft bias names. The current ADR-023 `0.38` and `1.35` constants emit warning-only entries because ADR-023 documents them as inherited slice-oracle constants.

## Commands Run

`python -m py_compile scripts/m6_warm_bubble_test.py tests/test_m6x_warm_bubble_operator_sanity.py`

Output: no output, exit 0.

`pytest tests/test_m6x_warm_bubble_operator_sanity.py -v | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_new_gate.txt`

Output summary: `4 passed in 9.80s`.

`pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py -v | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_no_regression.txt`

Output summary: `25 passed in 21.91s`.

`python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.json | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.txt`

Output summary: verdict `FAIL_PHYSICAL_BOUNDS`; `first_nonfinite_step = null`; preconditions OK; no hard anti-clamp failures. Bound violation: `mu_perturbation_max_Pa = 86374.47494781279` at step 300, bound `<= 50000.0`.

`pytest tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_transfer_audit.txt`

Output summary: `4 passed in 0.40s`.

## Proof Objects

- `proof_new_gate.txt`: new operator-sanity tests pass.
- `proof_no_regression.txt`: R7 oracle, ADR-023, c2 acoustic, MPAS slice, path unification, and pressure wiring pass.
- `proof_current_state_verdict.json` and `.txt`: current state is finite but fails physical bounds due to mu perturbation.
- `proof_transfer_audit.txt`: existing transfer audit tests pass.
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md`: PROPOSED policy ADR.

## Risks

- Current unified ADR-023 path does not pass operator sanity because `mu_perturbation_max_Pa` exceeds the conservative 50 kPa bound by 600 s. This is expected evidence, not tuned away.
- The static scan is source-pattern based. It catches known target-shaped clamps but does not replace reviewer inspection for future stabilizer changes.
- The harness CLI returns nonzero for non-`PASS_OPERATOR_SANITY`; the contract command pipes through `tee`, so the proof file was still captured. The JSON verdict is the source of truth.
- No remote push was performed because the sprint contract lists `No remote push` as a non-goal, despite the outer role wrapper mentioning push.

## Handoff

Objective: Stage-1 warm-bubble gate redesign is implemented on branch `worker/gpt/m6x-warm-bubble-gate-redesign`.

Files changed: listed above; no production `src/gpuwrf/` files were edited.

Proof objects produced: all required proof files are in `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/`.

Unresolved risks: ADR-023 still needs a follow-up operator sprint for the mu limiter/saturation path; a future amplitude gate requires a sourced WRF/CM1/MPAS reference.

Next decision needed: reviewer should decide whether to accept ADR-024's policy change and whether the next sprint should target mu-continuity stabilization or a sourced Stage-2 amplitude reference.
