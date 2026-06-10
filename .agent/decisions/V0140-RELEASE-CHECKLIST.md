# V0.14 Release Checklist

Date: 2026-06-10 05:58 WEST
Owner: manager

## Release Rule

Do not tag v0.14 until the code is stable under the current grid-parity,
memory, and validation gates. The release name is secondary; the invariant goal
is a WRF-faithful-enough, GPU-optimized, near compute- and memory-optimal,
scalable GPU rewrite.

## Current Active Lanes

| Lane | State | Gate before release |
|---|---|---|
| Grid-cell parity | Active. MYNN source-output deficit, MYNN surface first-call semantics, WRF `LANDUSE.TBL`-backed `TSK/ZNT/MAVAIL`, WRF `phy_prep` thermodynamic inputs, local `SFCLAY1D_mynn` output algebra, and MYNN dry-theta/phy_prep source coupling are fixed/proven or bounded. Strict Step-1 remains red at max_abs `438.5379097262689`, RMSE `5.4654420375782955`. Current blocker is narrower and upstream of MYNN: WRF preserves UST between `SFCLAY1D_mynn` output and MYNN-driver input (max_abs `4.998779168374767e-12`) but changes HFX (max_abs `277.80298614281253`) and QFX (max_abs `1.4684322196e-05`). | Surface/land flux handoff sprint is opened at `.agent/sprints/2026-06-10-v014-step1-surface-land-flux-handoff/`: hook WRF immediately before/after `module_surface_driver` `sf_surface_physics=4` flux updates (`HFX/QFX/LH/TSK/GRDFLX` and available CH diagnostics), compare with the JAX Step-1 path, wire/fix the land flux overlay into MYNN bottom-boundary handles if local, then rerun `step1_mynn_source_coupling.py` and strict Step-1. |
| Memory/FP32 Mythos lane | Closed and manager-merged. Accepted commits: `26815feb` MYNN BouLac tiling + shared RK-stage transport velocities, `bc847db2` default-inert FP32 acoustic precision-mode contract, `8f735a56` proofs/roadmaps/closeout. Exact-branch memory preflight is green at 8116 MiB compute peak and 378 s warm-cache. Mixed FP32 R1/R2 remains blocked by the open fp64 dynamics frontier. | Rerun exact-branch memory preflight only on the final candidate branch before long validation. Escalate FP32 implementation to Fable/Mythos only if the fp64 grid-parity frontier closes and GPT cannot directly solve mixed precision. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. Offline Atlas tooling is merged (`07e1ab2e`) and ready for post-parity validation data. | Atlas produces manifest, summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. |
| Switzerland/Gotthard | CPU truth exists. No post-fix GPU run has been accepted. | Post-stabilization GPU-vs-CPU proof with field-level comparison and finite/stability evidence. |
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

1. Close the active Step-1 surface/land heat-moisture flux handoff blocker
   before MYNN with WRF raw hook evidence and a JAX fix if local.
2. Rerun the strict Step-1 and short grid-field falsifier before launching
   longer campaigns.
3. Run exact-branch memory preflight on the final candidate branch.
4. Run Switzerland/Gotthard post-fix GPU-vs-CPU validation.
5. Run Grid-Delta Atlas on the selected paired cases and freeze tolerance
   manifest before claiming equivalence.
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
