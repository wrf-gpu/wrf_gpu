# Sprint Contract — F7.A: Cross-RK Save Family + `advance_uv_wrf` (first scoped rewrite)

**Sprint ID**: `2026-05-28-f7a-save-family-and-advance-uv`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/f7a-save-family-and-advance-uv`
**Worktree**: `/tmp/wrf_gpu2_f7a`
**Wall-time**: 2-3 days (best effort)
**GPU usage**: YES
**Sandbox**: `--dangerously-bypass-approvals-and-sandbox` (per user 2026-05-28; memory `feedback-codex-sandbox-caveats`)

## Why this sprint

F3 (Opus arch review), F5 (WRF cadence spec), and F6 (12-step transaction audit) all converge on the same diagnosis: the operational dycore fails at step 1 RK1 substep 1 inside `advance_mu_t` because (a) the RK1 saved family is not carried across RK2/RK3, (b) `advance_uv` is missing entirely from the acoustic substep, and (c) `_diagnose_pressure` is a one-line stub. F6 audit shows **zero u/v delta inside acoustic substeps** across all toggle combinations — confirming (b) empirically.

F6 recommends the targeted next-fix order:
1. Carry RK1 saved family across RK2/RK3 (this sprint)
2. Add/validate small-step `advance_uv_wrf` (this sprint)
3. Replace `_diagnose_pressure` stub (next sprint, F7.B)

This is the **first scoped rewrite sprint** in the dycore-repair series. Keep scope tight; F7.B-D follow.

## Binding goal (universal)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `proofs/f5/wrf_cadence_spec.md` — **binding spec for this sprint** (items 5, 6, 8 mapped to WRF file:line)
2. `.agent/sprints/2026-05-28-f3-agy-architecture-followup/findings.md` — Opus arch review
3. `.agent/sprints/2026-05-28-f6-first-12-step-transaction-audit/worker-report.md` — what to expect to fix
4. `proofs/f6/audit_summary.md` — failure signals (acoustic uv max delta = 0)
5. `proofs/f6/invariant_violations.json` — exact step/operator/invariant list
6. WRF Fortran reference at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`:
   - `module_small_step_em.F` `advance_uv` lines 654-942
   - `module_small_step_em.F` `small_step_prep` lines 125-285
   - `module_em.F` `rk_tendency` for context on what advance_uv receives
7. `src/gpuwrf/runtime/operational_mode.py` — `_rk_scan_step`, `_with_save_family`, `_acoustic_scan`
8. `src/gpuwrf/dynamics/core/acoustic.py` — current `acoustic_substep_core`
9. `src/gpuwrf/dynamics/mu_t_advance.py` — existing kernel to keep
10. `tests/unit/test_rk_scan_step_advection_active.py`, `test_mu_persistence_two_substeps.py`, `test_decouple_theta_state_reference.py` — these MUST pass after this sprint (F6 added them and they currently xfail)

## Approach (3 phases)

### Phase 1 — `small_step_prep_wrf` + `small_step_finish_wrf` cross-stage carry (1 day)

a. In `src/gpuwrf/dynamics/core/small_step_prep.py` (NEW file), implement `small_step_prep_wrf(state, rk_step, dt_rk)` that mirrors WRF `module_small_step_em.F:125-285`:
   - Saves stage-start physical perturbation: `mu_1`, `theta_1`, `ph_1`, `u_1`, `v_1`, `w_1`
   - Builds coupled work arrays: `muts`, `muus`, `muvs`, `mu_save`, `ww_save`, `c2a` (use real `c2a = cpovcv*(pb+p)/alt` per F5 spec item 7)
   - Returns a `SmallStepPrepState` dataclass/pytree carrying ALL of above for use in the acoustic loop
2. In `src/gpuwrf/dynamics/core/small_step_finish.py` (NEW file), implement `small_step_finish_wrf(prep_state, acoustic_out)` that mirrors WRF `module_small_step_em.F:364-430`:
   - Reconstructs physical perturbation fields from coupled work arrays
   - Restores `mu_2`, `ph`, `ww`
   - Returns the post-acoustic operational state
3. Modify `_rk_scan_step` in `operational_mode.py` to:
   - Replace `_with_save_family` body with a call to `small_step_prep_wrf` per RK stage
   - **Critical**: RK2 and RK3 must consume the RK1-produced `mu_save`, not rebuild it. Carry `mu_save` (and other invariant save arrays) through the RK loop.
   - After the acoustic loop, call `small_step_finish_wrf`

### Phase 2 — `advance_uv_wrf` (1 day)

a. In `src/gpuwrf/dynamics/core/acoustic.py`, add `advance_uv_wrf(state, prep, large_step_tend, dts_rk)` that mirrors WRF `module_small_step_em.F:654-942`:
   - Combines large-step momentum tendencies (RK-stage advection + horizontal PGF + Coriolis if applicable) with small-step PGF
   - Updates `u` and `v` work arrays in-place per substep
   - Honors the coupling `*muts`, `*c1h/c2h` per WRF
   - Returns updated state
b. Modify `acoustic_substep_core` to call `advance_uv_wrf` BEFORE `advance_mu_t_core` per WRF's `solve_em.F:3088, 3398` cadence.
c. Keep `advance_mu_t_core` (kernel is good per F5 item 9), but rewire inputs from the new `prep_state`.

### Phase 3 — Verification (0.5-1 day)

a. Run the 3 F6 unit tests:
   - `taskset -c 0-3 pytest -q tests/unit/test_rk_scan_step_advection_active.py` — should now PASS (advection + advance_uv active)
   - `taskset -c 0-3 pytest -q tests/unit/test_mu_persistence_two_substeps.py` — should PASS (mu_save carried)
   - `taskset -c 0-3 pytest -q tests/unit/test_decouple_theta_state_reference.py` — should PASS (theta_1 reference correct)
b. Re-run F6 audit driver: `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f7a` — emit comparison vs baseline.
c. Acoustic uv max delta should be > 0 in combination (a) — pure dycore.
d. RK2 saved-state theta_1 mismatch invariant should NOT fire.
e. `_diagnose_pressure` stub is OUT OF SCOPE (F7.B) — pressure bound may still violate.

## Acceptance

- **AC1**: `small_step_prep_wrf` and `small_step_finish_wrf` modules exist with WRF file:line references in docstrings.
- **AC2**: `advance_uv_wrf` exists in `core/acoustic.py` and is called before `advance_mu_t_core`.
- **AC3**: 3 F6 unit tests PASS (or document why one remains xfail).
- **AC4**: F6 re-run on combination (a) shows acoustic uv max delta > 0 AND rk_saved_state_theta_1 invariant does NOT fire.
- **AC5**: `proofs/f7a/regression_diff.md` produced documenting before/after improvement.
- **AC6**: `proofs/f7a/speedup_estimate.json` — current speedup numbers (we want to know rewrite impact).
- **AC7**: `worker-report.md` written, verdict `F7A_COMPLETE` or `F7A_PARTIAL`.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--dangerously-bypass-approvals-and-sandbox`.
3. **Files writable**: `src/gpuwrf/dynamics/core/small_step_prep.py` (NEW), `src/gpuwrf/dynamics/core/small_step_finish.py` (NEW), `src/gpuwrf/dynamics/core/acoustic.py` (add advance_uv_wrf only — keep existing acoustic_substep_core as a thin caller), `src/gpuwrf/runtime/operational_mode.py` (surgical: _rk_scan_step + _with_save_family region only), `proofs/f7a/**`, `.agent/sprints/2026-05-28-f7a-.../**`.
4. **Files NOT writable**: physics couplers, BC code, comparator scripts, state contracts, governance, plan, ADRs.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0:0 "AGENT REPORT: f7a DONE exit=$?" Enter`.
8. **End with verdict**: `F7A_COMPLETE` / `F7A_PARTIAL` + headline (acoustic uv max delta, theta_1 invariant status, T2 RMSE if measurable).

## Out of scope (deferred to F7.B+)

- `_diagnose_pressure` → `calc_p_rho_wrf` (F7.B)
- `advance_w_wrf` with full RHS + ph_tend (F7.B)
- `rk_addtend_dry` (F7.C)
- Scalar flux accumulation (F7.D)
- WRF flux-form mass-coupled advection (F7.C or D)
