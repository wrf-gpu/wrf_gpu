# V0.14 Release Checklist

Date: 2026-06-10 07:23 WEST
Owner: manager

## Release Rule

Do not tag v0.14 until the code is stable under the current grid-parity,
memory, and validation gates. The release name is secondary; the invariant goal
is a WRF-faithful-enough, GPU-optimized, near compute- and memory-optimal,
scalable GPU rewrite.

## Current Active Lanes

| Lane | State | Gate before release |
|---|---|---|
| Grid-cell parity | Active. MYNN source-output deficit, MYNN surface first-call semantics, WRF `LANDUSE.TBL`-backed `TSK/ZNT/MAVAIL`, WRF `phy_prep` thermodynamic inputs, local `SFCLAY1D_mynn` output algebra, MYNN dry-theta/phy_prep source coupling, WRF surface/land flux handoff localization, and the JAX Step-1 NoahMP-disabled configuration gap are fixed/proven or bounded. `proofs/v014/noahmp_step1_closure.*` now proves `use_noahmp=True`, `sf_surface_physics=4`, and WRF-derived NoahMP land/static/radiation state in Step-1. Strict Step-1 remains red against pinned one-run WRF truth at max_abs `1489.5135568470864`, RMSE `13.2001844004901`. Latest primary blocker: NoahMP land-tile energy/HFX chain at strict worst cells, now localized to θ_m/dry-θ forcing semantics plus secondary radiation forcing. New audit `proofs/v014/moist_theta_physics_consumer_audit.*` expands the θ_m boundary to multiple physics adapters; state/LBC/dycore keep moist theta, physics adapters require dry views and moist writeback. Secondary RRTMG GLW/RTHRATEN residual is localized, not closed: `proofs/v014/rrtmg_step1_forcing_parity.*` names the remaining boundary as clear-sky derived RRTMG optical/gas/top-buffer profile or downstream kernel boundary; no production fix is proven. | Close NoahMP forcing/energy parity and rerun `proofs/v014/noahmp_step1_closure.py`; close or explicitly scope the broader moist-theta physics adapter boundary for v0.14 validation; then close RRTMG with a temporary WRF RRTMG forcing hook for derived LW/SW profiles, fluxes, and heating arrays, or record a formal demotion before release. |
| Memory/FP32 Mythos lane | Closed and manager-merged. Accepted commits: `26815feb` MYNN BouLac tiling + shared RK-stage transport velocities, `bc847db2` default-inert FP32 acoustic precision-mode contract, `8f735a56` proofs/roadmaps/closeout. Exact-branch memory preflight is green at 8116 MiB compute peak and 378 s warm-cache. Mixed FP32 R1/R2 remains blocked by the open fp64 dynamics frontier. | Rerun exact-branch memory preflight only on the final candidate branch before long validation. Escalate FP32 implementation to Fable/Mythos only if the fp64 grid-parity frontier closes and GPT cannot directly solve mixed precision. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. Offline Atlas tooling is merged (`07e1ab2e`) and ready for post-parity validation data. Pre-result tolerance candidate is accepted in `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`: ten hard documented fields, static exact/tight checks, and `P/PH/MU/RAINC` critical report-only. | Final scoring uses the accepted manifest, produces summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. |
| Switzerland/Gotthard | CPU truth exists. A tracked 24-rank CPU rerun is complete at `/mnt/data/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z`: 25 `wrfout`, `rc=0`, `SUCCESS COMPLETE WRF`, last-frame finite PASS, total wall `1084.6 s`, mainloop `1078.4 s`, peak sampled total WRF-rank RSS `12563.766 MiB`. No post-fix GPU run has been accepted. | Post-stabilization GPU-vs-CPU proof with field-level comparison, finite/stability evidence, and GPU resource CSVs matching the CPU resource baseline. |
| Powered TOST | Three cases are durable; marathon paused. | Resume only after grid-field divergence is fixed/bounded and memory changes are merged. Interpret together with Grid-Delta Atlas, never alone. |

## Merge Discipline

- Memory/FP32 Mythos changes are already merged after proof review and focused
  regression gates. Future source changes still require proof objects, JSON
  validation, `git diff --check`, and focused regression gates before
  acceptance.
- Keep memory/FP32 semantic changes separate from bit-identical layout fixes in
  commits where possible.
- Do not start long GPU validation from a branch that has unreviewed source
  changes from another worker.

## Final v0.14 Gate Sequence

1. Close the active NoahMP land-tile energy/HFX Step-1 blocker with a
   production/proof fix if local, or an exact WRF-anchored narrower blocker.
2. Close or bound secondary RRTMG GLW/RTHRATEN parity, then rerun the strict
   Step-1 and short grid-field falsifier before launching
   longer campaigns.
3. Run exact-branch memory preflight on the final candidate branch.
4. Run Switzerland/Gotthard post-fix GPU-vs-CPU validation.
5. Run Grid-Delta Atlas on the selected paired cases using the accepted
   pre-result tolerance manifest before claiming equivalence.
6. Resume powered TOST and publish station results together with the atlas.
7. Update README, `docs/KNOWN_ISSUES.md`, `PROJECT_PLAN.md`, release notes, and
   proof links.
8. Tag and push v0.14 only after all required gates pass or are honestly
   demoted with a recorded manager decision and independent review.

## Current Do-Not-Run List

- No TOST marathon while Step-1 field divergence is still unresolved.
- No Switzerland GPU campaign before the parity candidate branch is stable.
- No broad FP32/mixed-precision claim from default-inert scaffolding.
- No station-only equivalence claim without all-cell field evidence.
