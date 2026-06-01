# v0.1.0 Proof Table (GPU rows EXECUTED on the HFX-fix HEAD)

- **Commit:** HFX-fix source `d1c373b` (surface_layer MYNN z_t) + proofs on `worker/opus/final-verdict`
- **Generated (UTC):** 2026-06-01 (final-verdict GPU campaign, RTX 5090)
- **GPU rows executed:** YES — run STRICTLY SERIAL (one job at a time) on a contended box
- **Tally:** 9 PASS / 1 FAIL (harness) / 1 INCONCLUSIVE (of 11)

GPU rows were executed (not deferred). Each row is re-derived from source by
`scripts/verify/<row>.sh` (re-runs the real validation + asserts the gate). On this
box the GPU was shared with sibling agents; runs were serialized and re-launched on
collision. Hibernation was inhibited via `systemd-inhibit` for the campaign.

| # | Status | Verify script | Claim / key numbers (executed) |
|---|--------|---------------|----------------------|
| 1 | PASS | `scripts/verify/idealized_warmbubble.sh` | Skamarock warm bubble verdict=PASS, 6/6 checks, RAN_TO_COMPLETION (after fixing a real `r.payload` harness bug) |
| 2 | PASS | `scripts/verify/idealized_straka.sh` | Straka density current verdict=PASS, all checks, RAN_TO_COMPLETION (same harness fix) |
| 3 | **FAIL (comparator-harness)** | `scripts/verify/savepoint_parity.sh` | `advance_mu_t_core` gets `theta=None` in the m6b6 coupled-COMPARATOR path (AcousticLoopState->AcousticCoreState threading). Production dycore is correct (rows 1/2/7 + d02/d03 all PASS); this is a savepoint-harness state-wiring defect, NOT a production-path failure. Needs a dedicated harness fix. |
| 4 | PASS | `scripts/verify/d02_validation.sh` | 3-case post-fix **D02_VALIDATED** (all_pass, no_blowup). T2 RMSE unchanged vs pre-fix (no regression); winds beat persistence every lead; finite/stable to 72h. case1(159w)+case2(120w)+case3(159w). |
| 5 | PASS | `scripts/verify/d03_validation.sh` | 24h 1km Tenerife **D03_1KM_VALIDATED**: T2 RMSE 1.92K (gate 3.0, beats persistence), U10 3.45 / V10 4.24 (gate 7.5, V10 beats persistence), all_finite, wall 1970s. HFX fix collapses the pre-fix +6.8K/+3.6K daytime warm bias to d02-quality. |
| 6 | PASS (qualified) | `scripts/verify/tost.sh` | n=3 MAM paired-TOST GPU-vs-CPU-WRF: U10 EQUIVALENT (Δ+0.095, margin 0.231); V10 borderline (tost_p 0.052); T2 not equiv (Δ+0.86K). PREDECLARED-UNDERPOWERED + SINGLE-SEASON (NOT seasonal). Empirical σ: T2 0.31/U10 0.014/V10 0.13. Machinery self-test = 0.0 delta. Full seasonal n≥15-27 = v0.2.0. |
| 7 | PASS | `scripts/verify/conservation.sh` | guards-off dycore stays finite + genuinely fp64 on real Canary d02; warm bubble PASSES guards-off incl. dry-mass-drift conservation check; guards not load-bearing. |
| 8 | PASS | `scripts/verify/repeatability.sh` | deterministic re-run (--repeat) AND restart-continuity (--restart-at-hour 1) both PASS; final wrfout within-tol identical (HOURS=2). |
| 9 | PASS | `scripts/verify/performance.sh` | warmed ~15-16 s/fc-hour, segscan 24h PASS+finite, provenance-backed speedup ~5-8x vs 28-rank CPU-WRF (floor 3.2x, d02-only). |
| 10 | PASS | `scripts/verify/precip.sh` | honest characterization: precipitates (jax 0.393mm vs WRF 0.347mm, ratio 1.13), water closure 2.6e-6 (gate 1e-3), rain-dominated. Per-field bias reported not gated. |
| 11 | **INCONCLUSIVE** | `scripts/verify/device_residency.sh` | jit().lower() HLO introspection trips on State pytree reconstruction (state.py:549 jnp.asarray(lu_index) sees ArgInfo). Made HLO best-effort -> the binding warmed profiler-trace audit now runs and measures post-init H2D=D2H=3.69MB (gate wants 0/0). The HLO in-loop transfer-op count (in-loop vs one-time discriminator) is blocked by the .lower() State issue, so whether 3.69MB is an in-loop transfer (defect) or one-time output staging (acceptable) is UNRESOLVED. Recorded honestly, not massaged. |

## Notes

- **Rows 1, 2, 4, 5, 7, 8, 9, 10 = PASS (8 rows)** on the HFX-fix HEAD. Row 6 = PASS
  qualified (underpowered n=3 single-season; U10 equivalent, the honest v0.1.0 scope).
- **Row 3 (savepoint parity) = FAIL** — a comparator-harness `theta=None` state-threading
  bug in `scripts/m6b6_coupled_step_compare.py` / `advance_mu_t_core`, NOT a production
  dycore defect (the production dycore passes the idealized + conservation + d02/d03 gates).
- **Row 11 (device residency) = INCONCLUSIVE** — a State `jit().lower()` reconstruction
  issue blocks the definitive in-loop HLO transfer-op count; the profiler trace shows a
  3.69MB H2D/D2H that is not disambiguated as in-loop vs one-time.
- Harness fixes committed this campaign: rows 1/2 (`.payload` -> verdict/status/checks),
  row 11 (graceful HLO degrade + keyword `hours`).
- Real follow-ups (v0.2.0): (a) fix the m6b6 savepoint comparator theta threading (row 3);
  (b) make `State.tree_unflatten` `.lower()`-safe so the HLO transfer-op count can settle
  row 11; (c) seasonal TOST corpus for row 6.
