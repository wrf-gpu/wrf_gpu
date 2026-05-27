# Gap Analysis Against Community Acceptance Criteria

Statuses:

- `HAVE`: proof object exists on disk and supports the claim.
- `PARTIAL`: proof object exists but is narrower than the community criterion.
- `MISSING`: no adequate proof object found in this checkout.
- `OUT_OF_SCOPE_V0`: valuable, but not required for the near-term paper if honestly scoped.

## Current Evidence Snapshot

| Evidence item | Status | Proof object or source |
|---|---|---|
| B6 coupled-step savepoint parity | HAVE | `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/proof_coupled_step_parity.json`; worker report says column, patch16, and golden tiers passed with max observed delta `0.0`. |
| Three V3 initial-condition Tier-4 RMSE gate | HAVE | `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/proof_tier4_rmse_all3.json`; aggregate T2/U10/V10 all PASS. Note: the same sprint worker report still marked broader acceptance blocked by guard-disabled and parity caveats; the later manager closeout accepted Tier-4 RMSE as the operational M6 gate. |
| Corrected current speedup | HAVE | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json`; worker report states d02-only speedup `22.25579686534753`. |
| Zero inter-kernel D2H in forecast loop | HAVE | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`; status PASS, `counts.d2h_inter_kernel_inside_window == 0`, `bytes.d2h_inter_kernel_inside_window == 0`. |
| Restart continuity | HAVE | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`; worker report says 20-step reference vs 10+restart+10 max delta `0.0` for every State field. |
| One-day side-by-side AEMET CPU/GPU scoring | HAVE but FAILING | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` and `verdict.md`; same scoring code, 73 stations, 24 hours, 1747 rows, GPU materially worse than CPU. Iteration 2 still fails in `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json`. |
| 24 h end-to-end pipeline | HAVE | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json` and worker report; current iteration is finite, restart PASS, speedup PASS, but exits partial because scoring was not requested in the pipeline command. |
| 1 km memory feasibility | PARTIAL | `.agent/sprints/2026-05-27-m7-1km-memory-audit/operational_gaps.md` and closeout references. Synthetic memory feasibility is useful but not a full 1 km forecast or transient-memory proof. |
| WRF-compatible output path | PARTIAL | `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md`, `.agent/sprints/2026-05-27-m7-netcdf-writer/compat_matrix_v2.md`; subset matrix, not full WRF I/O ecosystem. |

## Detailed Gap Table

| Criterion | Status | Current state | Gap to publication-grade evidence |
|---|---|---|---|
| ARW formulation described with WRF v4 references | PARTIAL | `publication/research_brief/english_brief.txt` cites ARW technical note; `publish/paper/paper.md` discusses C-grid, RK3, savepoints, and scope. | Need paper-facing equation/deviation table and citation verification before submission. |
| Warm-bubble idealized case | PARTIAL | There are warm-bubble diagnostic tests under `tests/test_m6x_warm_bubble_operator_sanity.py` and related sprints, but they are failure-diagnostic infrastructure, not a publishable idealized benchmark. | Need a frozen WRF/reference warm-bubble case with input, run command, field metrics, conservation checks, and proof JSON. |
| Density-current idealized case | MISSING | No density-current proof object found. | Add analytic or WRF-like density-current benchmark with cold-pool front metric, theta perturbation metric, mass and energy budgets. |
| Mountain-wave / hill-flow idealized case | PARTIAL | There are M4/M6 hill/warm-bubble-style components and `em_hill2d_x` mentioned in roadmap, but no publication-grade mountain-wave proof found. | Add WRF `em_hill2d_x` or equivalent mountain-wave comparison with vertical velocity phase/amplitude metric. |
| Baroclinic-wave / synoptic dycore benchmark | MISSING | No baroclinic-wave proof object found. | Add WRF standard or DCMIP/Jablonowski-Williamson style benchmark, likely MEDIUM priority if overnight budget is tight. |
| Stock NCAR WRF test-suite coverage | MISSING | Current coverage is project-specific fixtures and Canary replay. | At least two stock WRF ideal/test cases should be run or explicitly documented as unsupported. |
| Tier-1 operator/savepoint parity | HAVE | B0-B6 savepoint ladder proof objects exist, including B6 coupled-step parity. | Paper must describe the limited scope: validation-mode parity, not full 24 h meteorological equivalence. |
| Dry-mass conservation over 24 h | PARTIAL | Tier-2 invariant tests exist (`tests/test_m6_tier2_coupled.py`, `artifacts/m6/tier2_coupled_invariants.json`) but no publication-specific 24 h conservation proof found. | Add closed-domain or boundary-budget-corrected 24 h mass proof for idealized and Canary cases. |
| Energy budget over 24 h | MISSING | No explicit energy-budget proof found. | Add dry/total-energy diagnostic with a documented approximation and residual threshold. |
| Water-vapour / moisture budget | PARTIAL | M5/M6 physics invariant tests exist for Thompson/MYNN/RRTMG; no full 24 h moisture-budget proof found. | Add moisture budget for active-physics Canary or mark as required future if precipitation not yet claim-bearing. |
| CFL / timestep stability margin | MISSING | Guard-disabled and 1 h probes exist, but no systematic timestep margin sweep found. | Add nominal, half-step, and stressed-step comparison for idealized cases and Canary smoke. |
| Multi-regime Canary evaluation | MISSING | One side-by-side day exists for 2026-05-21. M6 has three V3 one-hour IC checks. | Need 7-10 day corpus spanning regimes before claiming robust meteorological validity. |
| Forecast-vs-observation point metrics | PARTIAL | AEMET scaffold and one day of side-by-side scoring exist; result currently fails relative to CPU. | Need multi-day station verification and explicit CPU/GPU/obs tables. |
| Neighborhood precipitation FSS | PARTIAL | Forecast-vs-obs scaffold references FSS and tests exist, but no multi-day/event proof object found. | Add at least one precip/cloud event with FSS thresholds and radii. |
| Object-based precipitation SAL / MODE | MISSING | No SAL/MODE proof found. | Add MEDIUM-priority precip-event object verification or mark future work if no event corpus exists. |
| Cross-hardware reproducibility | MISSING | Current proof is single RTX 5090 plus repeatability. | Need second GPU/driver/backend if feasible; otherwise paper must state this is future work. |
| Restart bitwise continuity | HAVE | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` and pipeline restart proof. | Preserve as a paper proof object. |
| Public repo / license / citation / docs | PARTIAL | Repository has governance and proof-object discipline. Public release metadata is not complete in the tested paper path. | Add release checklist: public URL, release commit, LICENSE, CITATION, CONTRIBUTING, install docs, tutorial, CI, data policy. |
| Independent-review access | PARTIAL | Proof objects are local and numerous; raw binary profiler traces may be outside git. | Need manifest mapping claims to proof files, plus external archive for large artifacts or scripts to regenerate them. |

## Honest Publication Implications

`wrf_gpu` can currently support these claims:

- A nontrivial JAX/Python WRF-compatible GPU prototype exists on disk.
- Short-run validation and savepoint evidence are substantial.
- The current corrected d02 24 h path remains fast: about `22.26x` apples-to-apples d02-only speedup on the measured 2026-05-21 case.
- The forecast loop preserves the key whole-state residency invariant: zero inter-kernel D2H inside the profiled window.
- Restart continuity is bitwise for the tested path.

`wrf_gpu` cannot currently support these stronger claims:

- It is an operational WRF replacement.
- It has meteorological skill comparable to CPU WRF.
- It is validated across regimes or seasons.
- Its physics suite is fully WRF-equivalent.
- Its precipitation, cloud, water, and energy behavior are publication-grade.
- Its cross-hardware reproducibility is known.

The next execution sprint should therefore prioritize evidence that is both community-standard and likely to change the paper's acceptance status: idealized tests, conservation budgets, multi-day side-by-side skill, and public release readiness.
