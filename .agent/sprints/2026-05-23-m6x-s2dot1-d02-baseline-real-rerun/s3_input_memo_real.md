# S3 Input Memo Real - M6.x S2.1 d02 Baseline Rerun

## Baseline Status

Real Gen2 d02 baseline numbers are unavailable. The extended probe timed out after 1800s and the 1h forecast did not start. The artifacts in this sprint therefore do not authorize a real-data S3 operator fix sprint. They authorize a blocker decision: fix or isolate the replay probe/JAX/GPU startup path first, then rerun S2.1.

## Top Operator Concerns

### 1. _mu_continuity_increment temporary limiter / hidden mass cap

Source table: `.agent/decisions/source_mining_operator_table.md` row `_mu_continuity_increment` limiter concern

Proof cites:
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_sanitizer_audit.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_limiter_activation_tracker.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_stabilizer_provenance_scanner.json`

Real baseline numbers: unavailable; current counts come from synthetic fallback only.

Recommended source-cited fix: do not launch a physics-correction S3 based on this evidence. First make the real replay probe produce a proof object; then replace or explicitly ratify the mass update against WRF MUAVE/MUTS/ww or MPAS perturbation-state lines if real sanitizer/limiter evidence still implicates it.

Expected baseline effect: unknown until real replay exists. Synthetic fallback cannot estimate U10/V10/T2 drift improvement.

### 2. MPAS_OMEGA_TO_W_METRIC = 1.35 constant metric

Source table: `.agent/decisions/source_mining_operator_table.md` row `MPAS_OMEGA_TO_W_METRIC = 1.35` concern

Proof cites:
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_vertical_column_phase_space.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_operator_term_budget_tracer.json`

Real baseline numbers: unavailable; `w_k20` and `theta_k20` fallback values are synthetic.

Recommended source-cited fix: hold the metric change until the real replay reaches at least the 1-second probe. Once real vertical slices exist, replace the constant with per-column/per-level mass-flux metric evidence, or keep the constant limited to the synthetic slice oracle.

Expected baseline effect: unknown; real terrain-amplified vertical divergence was not measured.

### 3. MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38 and missing WRF time averaging

Source table: `.agent/decisions/source_mining_operator_table.md` rows `0.38` buoyancy scale and time averaging

Proof cites:
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_field_rmse_timeline.json`
- `.agent/sprints/2026-05-23-m6x-s2dot1-d02-baseline-real-rerun/proof_operator_term_budget_tracer.json`

Real baseline numbers: unavailable; synthetic fallback RMSEs were T2=0.4 K, U10=0.5 m/s, V10=0.6 m/s, and are non-evidence for S3.

Recommended source-cited fix: do not tune or demote the 0.38 scale from this sprint's fallback data. First resolve the real replay blocker; then use real field RMSE and term-budget traces to decide whether WRF t_2ave/muave or MPAS coupled coefficients should replace the current scale.

Expected baseline effect: unknown until real W/theta and surface RMSE growth are measured.

## Exit-Rule Status

BLOCKER. S3 should not proceed as a real-baseline operator-fix sprint. The next bounded decision should be whether to run an infrastructure/debug sprint on the real replay probe startup path, because extending the guard from 120s to 1800s still did not produce `replay_mode == "real"`.
