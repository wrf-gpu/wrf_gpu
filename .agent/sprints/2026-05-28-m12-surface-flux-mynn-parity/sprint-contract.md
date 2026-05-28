# Sprint Contract — M12: Surface flux + MYNN bottom-BC parity (largest skill gain expected)

**Sprint ID**: `2026-05-28-m12-surface-flux-mynn-parity`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m12-surface-flux-mynn-parity`
**Worktree**: `/tmp/wrf_gpu2_m12`
**Wall-time**: 6-18 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

M9.C confirmed HFX divergence (max 4105 W/m², mean RMSE 924 W/m² — physically impossible for sensible heat flux which peaks ~500 W/m² in realistic conditions) is a REAL model bug. M12 is the HIGHEST expected single-skill-gain milestone per the plan and per M9.C ranking. Specifically: GPU `HFX` is the writer fallback `theta_flux * rhosfc * cp` (per M9.C comparator audit) — this composition is suspect. The MYNN bottom-BC also consumes surface fluxes. Both ends need WRF-Fortran-bitwise parity.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m9/divergence_map_v2.json` — HFX/LH/PBLH defect evidence
2. `src/gpuwrf/coupling/physics_couplers.py` — surface adapter, MYNN adapter
3. `src/gpuwrf/io/wrfout_writer.py` — current HFX/LH writer fallback derivation
4. `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` — prior RCA
5. WRF `phys/module_sf_sfclay.F` and `phys/module_bl_mynn.F` for sign/magnitude reference (read-only)

## Acceptance

### AC1 — Surface flux magnitude/sign parity

For Canary 20260521 hour 1, GPU `HFX` and `LH` outputs reproduce WRF wrfout `HFX`/`LH` to within **5 % per-cell** on the cells where the WRF value is non-trivial (|HFX_wrf| > 5 W/m²). Emit `proofs/m12/surface_flux_parity_hour_1.json`.

### AC2 — MYNN bottom-BC sign convention verified

Read `src/gpuwrf/coupling/physics_couplers.py` MYNN adapter. Verify that `theta_flux`, `qv_flux`, `tau_u`, `tau_v` are passed with WRF's bottom-BC sign convention. If sign is inverted, fix. Document the convention in `.agent/sprints/2026-05-28-m12-surface-flux-mynn-parity/mynn_bottom_bc_audit.md`.

### AC3 — HFX writer fix

`src/gpuwrf/io/wrfout_writer.py`: the writer fallback for HFX (currently `theta_flux * rhosfc * cp`) is replaced with the actual WRF formula. WRF computes HFX = `rho * cp * (theta_surface - theta_air) / r_a` where `r_a` is aerodynamic resistance. Use the same intermediate variables WRF stores in surface-layer state. If `surface_layer.F`'s direct output is already available in the State, prefer it.

### AC4 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC5 — 24h skill gain

Re-run Canary 20260521 24h with surface flux fix. Re-run `scripts/m7_gpu_vs_cpu_skill_diff.py`. Emit `proofs/m12/post_m12_skill_diff.json`. Acceptance:
- T2 RMSE ≤ 5.0 K (currently 10.80 K post-iter2; M12 target ≤ 5.0 K).
- HFX mean RMSE drops ≥ 70 % vs `divergence_map_v2.json` HFX value (924 W/m² → ≤ 280 W/m²).
- U10/V10 RMSE does not worsen.

### AC6 — Worker report

`.agent/sprints/2026-05-28-m12-surface-flux-mynn-parity/worker-report.md`: standard format. Verdict `M12_COMPLETE` if AC1-AC5 all pass; `M12_PARTIAL` with remaining gaps otherwise. If AC5 cannot reach T2 ≤ 5.0 K but achieves ≥ 30 % reduction, that's PARTIAL but worth reporting as progress.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`.
3. **Files writable**: `src/gpuwrf/coupling/physics_couplers.py` (surface + MYNN sections only), `src/gpuwrf/io/wrfout_writer.py` (HFX/LH only), `proofs/m12/**`, `.agent/sprints/2026-05-28-m12-surface-flux-mynn-parity/**`.
4. **Files NOT writable**: radiation, dycore, BC, state contracts, governance.
5. **Coordination with M11+M13**: avoid touching anything else; if a needed dependency is in another sprint's scope, document and stop with PARTIAL.
6. **No remote push.**
7. **Manager repo ONLY**.
8. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m12 DONE exit=$?" Enter`.
9. **End with verdict**: `M12_COMPLETE` / `M12_PARTIAL` + headline T2 RMSE post-fix value.
