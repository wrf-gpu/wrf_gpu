# Worker Report

Summary: Ran the real Gen2-anchored d02 1h baseline on branch `worker/gpt/m6x-s2dot1redo-real-baseline` with no operator changes. The replay path was real, not synthetic, and all 12 sidecar JSON proofs include `replay_mode: real`. Replay proof status is `PASS` because post-sanitize state leaves stayed finite and transfer audit was clean, but aggregate baseline status is `NEEDS_S3_FIX` due massive sanitizer masking, physical-bound violation, and surface RMSE far above Gen2 noise anchors.

## Files Changed

- `scripts/m6_d02_baseline_run_instrumented.py`: scoped orchestration-only changes. Defaulted S2.1-redo to skip the short probe, redirected command logs to the selected output directory, emitted S2.1-redo proof filenames, stamped sidecar JSON with `replay_mode`, and added bound-violation finding classification.
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/.gitignore`: ignores the raw JAX trace directory and large intermediate sidecar-input arrays.
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/`: proof JSON/TXT files, command logs, `findings_real.md`, `s3_input_memo_real.md`, and this report.

No files under `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, or `src/gpuwrf/physics/` were modified.

## Commands Run

- `timeout 3600 python scripts/m6_d02_baseline_run_instrumented.py --duration-s 3600 --output-dir .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/ | tee .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_real_1h_run.txt`
  Output: exit 0. Replay child exit 0, timeout false, elapsed 1033.772 s. Summary printed `status=NEEDS_S3_FIX`, `replay_mode=real`, `fallback_reason=null`.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_no_regression.txt`
  Output: `54 passed in 797.51s (0:13:17)`.
- `python -m py_compile scripts/m6_d02_baseline_run_instrumented.py`
  Output: exit 0.
- `python -m json.tool .agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_s2dot1redo_summary.json >/dev/null`
  Output: exit 0.
- Sidecar replay-mode check over `proof_*.json`
  Output: `sidecar_count=12 bad_replay_mode=0`.
- Process check during run: replay child ran on CPUs 0-3 as observed by `ps -eo pid,psr,pcpu,pmem,etime,args | rg 'm6_d02_(baseline|boundary)|d02_boundary_replay'`.

## Baseline Numbers

- 1h replay scan wall time: 578.492 s; replay command elapsed wall: 1033.772 s including load, compare, static audit, and trace audit.
- Throughput: 6.223x realtime.
- Peak GPU memory: 9,360,634,112 bytes.
- First post-sanitize nonfinite step: null. First pre-sanitize candidate nonfinite step: 2.
- Transfer audit: 0 post-init H2D/D2H bytes.
- Gen2 truth RMSE: t=15/30/45 min not claimed because this sprint has only hourly Gen2 truth files. At t=60 min: T2 136.885 K vs 24h Gen2 anchor 0.628 K; U10 106.419 m/s vs 1.456 m/s; V10 102.232 m/s vs 1.591 m/s. W k20 RMSE 50.0005 m/s and theta k20 RMSE 63.9165 K have no Gen2 noise-floor anchors in `rmse_summary.csv`.

## Top Numerical Findings

1. Sanitizer changed 20,049,146,903 candidate values over the 1h run.
2. Pre-sanitize candidates had 17,237,505,011 nonfinite values across 3,599 steps; first candidate nonfinite step was 2.
3. Candidate clip count was 2,811,641,892.
4. Physical-bound tracer found `theta_K=550.0` at step 3600 against a 400 K bound; post-sanitize finiteness alone is not enough.
5. Surface RMSE is two orders of magnitude above 24h Gen2 noise anchors, so this is a real S3 operator-fix input, not an acceptable forecast baseline.

## Proof Objects

- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_real_1h_run.txt`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_d02_replay.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_s2dot1redo_summary.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_no_regression.txt`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_bound_violation_tracer.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_sanitizer_audit.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_limiter_activation_tracker.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_field_rmse_timeline.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_spatial_divergence_map.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_conservation_tracker.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_boundary_ring_error_profiler.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_vertical_column_phase_space.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_operator_term_budget_tracer.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_transfer_launch_timeline.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_timestep_convergence_dashboard.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/proof_stabilizer_provenance_scanner.json`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md`
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/s3_input_memo_real.md`

## Risks

- The replay proof is only post-sanitize finite. Pre-sanitize candidate counts show severe instability from step 2 onward.
- Limiter and operator-term sidecars ran on real replay data, but the preserved replay proof does not serialize raw/bounded dmu arrays or per-term RHS arrays, so S3-real still needs source-cited instrumentation or a direct operator fix before Tier-3 claims.
- The raw JAX trace directory remains local and ignored because it is about 800 MB; large sidecar-input arrays also remain local. The committed proof JSONs record the measured outputs.
- Sub-hourly RMSE is intentionally absent because only hourly Gen2 truth is available in this sprint.

## Handoff

Objective completed: real 1h d02 Gen2 baseline measured with all 12 S1 sidecars and no operator edits. S3-real should start from `s3_input_memo_real.md`, prioritizing `_mu_continuity_increment` limiter/mass update ratification, replacement of the constant `MPAS_OMEGA_TO_W_METRIC = 1.35`, and demotion or source derivation of `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38`. Exit-rule status: S3-real plus one bounded fix sprint is plausible only if sanitizer/limiter masking is removed or explicitly source-ratified before any Tier-3 PASS claim. Per sprint contract non-goal, no remote push was performed.
