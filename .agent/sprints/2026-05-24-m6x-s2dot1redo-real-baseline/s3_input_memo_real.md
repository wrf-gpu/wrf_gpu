# S3 Input Memo - M6.x S2 Baseline

## Top Operator Concerns

### 1. _mu_continuity_increment temporary limiter / hidden mass cap

Source table: .agent/decisions/source_mining_operator_table.md row `_mu_continuity_increment` limiter concern

Proof cites:
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_sanitizer_audit.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_limiter_activation_tracker.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_stabilizer_provenance_scanner.json`

Recommended source-cited fix: Replace or explicitly ratify mass update against WRF MUAVE/MUTS/ww or MPAS perturbation-state lines; do not use Rayleigh damping as a mass limiter replacement.

Expected baseline effect: Reduce sanitizer/limiter masking and improve U10/V10/T2 drift only if mass continuity is the dominant error source.

### 2. MPAS_OMEGA_TO_W_METRIC = 1.35 constant metric

Source table: .agent/decisions/source_mining_operator_table.md row `MPAS_OMEGA_TO_W_METRIC = 1.35` concern

Proof cites:
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_vertical_column_phase_space.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_operator_term_budget_tracer.json`

Recommended source-cited fix: Replace with per-column/per-level mass-flux metric from MPAS zz geometry, or keep the constant only in the synthetic slice oracle.

Expected baseline effect: Should improve w_k20/theta_k20 phase behavior and any terrain-amplified spatial divergence.

### 3. MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38 and missing WRF time averaging

Source table: .agent/decisions/source_mining_operator_table.md rows `0.38` buoyancy scale and time averaging

Proof cites:
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_field_rmse_timeline.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_operator_term_budget_tracer.json`

Recommended source-cited fix: Demote 0.38 to slice-only unless pinned by fixture evidence; derive production buoyancy from WRF t_2ave/muave or MPAS coupled coefficient terms.

Expected baseline effect: Should primarily improve W/theta error growth before surface RMSE if buoyancy forcing is currently mis-scaled.

## Exit-Rule Status

S3 plus one bounded fix sprint is plausible, but only if it removes or ratifies limiter/sanitizer masking before Tier-3 claims.
