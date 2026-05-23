# Sprint Contract — M6.x Pressure Diagnose Wiring Fix (Opus diagnostic finding)

## Objective

Opus diagnostic (`tester/opus/m6x-warm-bubble-failure-diagnostic @ e56c0e6`, §7 WIRING-BUG sub-finding) identified a confirmed wiring bug at `src/gpuwrf/dynamics/acoustic_wrf.py:875-876` and the function it calls (`diagnose_pressure_al_alt` at `:235-244`). Directly measured: `_mpas_recurrence_vertical_update` computes `p_perturbation ≈ 12.7 Pa` after one substep, but the immediately-following call to `diagnose_pressure_al_alt` overwrites it back to `4 × 10⁻¹¹ Pa`. The cited WRF source anchor (`module_big_step_utilities_em.F:1082-1087` for `calc_p_rho_phi`) uses **`state.theta`** (full θ, not `theta_base`) and **includes `ph_perturbation`** in `al`. The current implementation diverges in both respects.

This bug is **correct in isolation regardless of which architecture (ADR-023 vs ADR-021) wins** and should be landed as a small fix. It will not raise warm-bubble `w_max` to [5, 10] (the architectural gap blocks that), but it will:
- Stop the slow blowup in θ' (144 K) and p' (450 kPa) currently masked by the tanh mass limiter
- Make the recurrence's density-derived pressure persist as intended
- Close one path-split source from the original reviewer report

## Non-Goals

- **No new physics**, no new stabilizers, no carry expansion.
- No modification of `_mpas_recurrence_vertical_update`, `vertical_acoustic_update`, or the analytic / MPAS slice oracles or their tests.
- No claim that this closes the warm-bubble failure — the architectural gap remains and is the subject of a separate dispatch.
- No remote push.
- No host/device transfer regression.

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-pressure-diagnose-wiring-fix`:

- `src/gpuwrf/dynamics/acoustic_wrf.py` — touch ONLY the diagnose-pressure-overwrite at `:875-876` (and the parallel pre-vertical-update overwrite if applicable, `:838`). Implement the gated fix per Opus's proposal:
  - When `config.non_hydrostatic` is True (the path that uses the MPAS recurrence with density-derived p): keep the recurrence's `p_perturbation`. Re-derive only `al`/`alt` diagnostics from the new state.
  - When `config.non_hydrostatic` is False (hydrostatic path): preserve current behavior.
- `tests/test_m6x_pressure_diagnose_wiring.py` (new) — at minimum: unit test asserting that `acoustic_substep_carry` with `non_hydrostatic=True` does NOT erase the recurrence's `p_perturbation` after one substep. Probe `g_overwrite_check` in the diagnostic script is a ready-made fixture pattern.
- `.agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/` — proofs + worker-report.

Read-only everywhere else.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md`** §3, §7, §8 — the bug and the proposed fix
- **`scripts/diagnostic_warm_bubble_vs_slice.py`** — probe `g_overwrite_check` is your reference fixture
- `src/gpuwrf/dynamics/acoustic_wrf.py:230-244` (`diagnose_pressure_al_alt`)
- `src/gpuwrf/dynamics/acoustic_wrf.py:875-876` (the offending overwrite)
- `src/gpuwrf/dynamics/acoustic_wrf.py:838` (parallel pre-update overwrite if exists)
- WRF source `module_big_step_utilities_em.F:1082-1087` (canonical `calc_p_rho_phi`)
- `.agent/decisions/ADR-023-conservative-column-solver.md` (the conservative path's intent)

## Acceptance Criteria

1. **Direct probe**: `probe_p_perturbation_survives_substep.json` shows `p_perturbation` after `acoustic_substep_carry` matches the post-recurrence value (within fp tolerance ≤ 1e-9 Pa), not the diagnostic-pressure overwrite.

2. **Unit test PASSES**: `pytest tests/test_m6x_pressure_diagnose_wiring.py -v` → all PASS.

3. **No regression**: `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py tests/test_m6x_adr023_path_unification.py -v` → all PASS (count ≥ 27 from prior runs).

4. **Transfer audit clean**: 5/5 PASS.

5. **Warm-bubble side-effect**: After the fix, re-run `scripts/m6_warm_bubble_test.py` and document the new behavior. Expect: `w_max` may still be ~0.04 m/s (architectural gap unchanged), BUT θ_perturbation should no longer grow to 144 K — should stay in physical range (< 20 K). p_perturbation should stay physical (< 1000 Pa). Capture `proof_warm_bubble_post_fix.json`.

6. **mu_continuity stabilizer status**: The `_mu_continuity_increment` tanh limiter remains in place for now (separate concern; not removed by this sprint). Document whether the fix changes mu blowup behavior.

7. **Worker report** at `worker-report.md`. Must include: summary, the specific lines changed (file:line), probe measurements before/after, why the gated approach is correct, files changed, commands, proof objects, risks, handoff.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_wiringfix
pytest tests/test_m6x_pressure_diagnose_wiring.py -v | tee .agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_wiring_gate.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py tests/test_m6x_adr023_path_unification.py -v | tee .agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_no_regression.txt
pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_transfer_audit.txt
python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_warm_bubble_post_fix.json | tee .agent/sprints/2026-05-23-m6x-pressure-diagnose-wiring-fix/proof_warm_bubble_post_fix.txt
```

## Performance Metrics

None — bug fix.

## Proof Object

- `proof_wiring_gate.txt` — new test PASS
- `proof_no_regression.txt` — prior tests still PASS
- `proof_transfer_audit.txt`
- `proof_warm_bubble_post_fix.json` + `.txt` — informational
- `worker-report.md`
- New code on `worker/gpt/m6x-pressure-diagnose-wiring-fix`

Time budget: **2-4 hours**.

## Risks

- **The fix may break a hydrostatic path**: gate carefully on `config.non_hydrostatic`. Verify both branches via tests.
- **The fix may unmask a deeper bug** in the recurrence — if θ' or p' grow nonfinite after the fix without the diagnose-pressure mask, document and report. Do NOT silently re-add the mask.
- **Spec-gaming**: don't accidentally rewrite `diagnose_pressure_al_alt` itself in this sprint — just gate when it's called. The cleaner fix (rewriting it to use `state.theta`) is a separate, larger surface.

## Handoff Requirements

When all proof files are on disk and `worker-report.md` is committed on `worker/gpt/m6x-pressure-diagnose-wiring-fix`, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-pressure-diagnose-wiring-fix / codex] exit=<ec>`.
