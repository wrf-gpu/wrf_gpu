# V0.14 Release Checklist

Date: 2026-06-09 23:15 WEST
Owner: manager

## Release Rule

Do not tag v0.14 until the code is stable under the current grid-parity,
memory, and validation gates. The release name is secondary; the invariant goal
is a WRF-faithful-enough, GPU-optimized, near compute- and memory-optimal,
scalable GPU rewrite.

## Current Active Lanes

| Lane | State | Gate before release |
|---|---|---|
| Grid-cell parity | Active. Latest committed manager HEAD: `5f8916f9`. Current boundary: Step-1 tendency construction, specifically WRF `first_rk_step_part2` `T_TENDF` and RK1 `T_TEND/PH_TEND/RW_TEND` versus JAX advection/augment paths. | The first remaining material divergence is fixed, or explicitly bounded with proof showing it does not invalidate the v0.14 field-equivalence claim. |
| Memory/FP32 Mythos lane | Active in `.codex/worktrees/mythos-memory-v014` on `worker/mythos/v014-memory-fp32`. Mythos claims MYNN BouLac tiling, transport-velocity reuse reclassification, FP32 R0 inert contract, and exact-branch GPU preflight; manager has not yet reviewed/merged. | Manager reviews source diffs and proof objects, reruns focused CPU/GPU gates, and merges only proof-backed commits. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. | Atlas produces manifest, summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. |
| Switzerland/Gotthard | CPU truth exists. No post-fix GPU run has been accepted. | Post-stabilization GPU-vs-CPU proof with field-level comparison and finite/stability evidence. |
| Powered TOST | Three cases are durable; marathon paused. | Resume only after grid-field divergence is fixed/bounded and memory changes are merged. Interpret together with Grid-Delta Atlas, never alone. |

## Merge Discipline

- Do not merge Mythos memory/FP32 changes just because a closeout exists.
- Require proof objects, JSON validation, `git diff --check`, and focused
  regression gates before accepting any source change.
- Keep memory/FP32 semantic changes separate from bit-identical layout fixes in
  commits where possible.
- Do not start long GPU validation from a branch that has unreviewed source
  changes from another worker.

## Final v0.14 Gate Sequence

1. Close the active Step-1 tendency-boundary sprint.
2. If the tendency sprint finds/fixes a source bug, rerun the strict Step-1 and
   short grid-field falsifier before launching longer campaigns.
3. Review and merge or reject the Mythos memory/FP32 branch.
4. Run exact-branch memory preflight on the final candidate branch.
5. Run Switzerland/Gotthard post-fix GPU-vs-CPU validation.
6. Run Grid-Delta Atlas on the selected paired cases and freeze tolerance
   manifest before claiming equivalence.
7. Resume powered TOST and publish station results together with the atlas.
8. Update README, `docs/KNOWN_ISSUES.md`, `PROJECT_PLAN.md`, release notes, and
   proof links.
9. Tag and push v0.14 only after all required gates pass or are honestly
   demoted with a recorded manager decision and independent review.

## Current Do-Not-Run List

- No TOST marathon while Step-1 field divergence is still unresolved.
- No Switzerland GPU campaign before the parity/memory candidate branch is
  stable.
- No broad FP32/mixed-precision claim from default-inert scaffolding.
- No station-only equivalence claim without all-cell field evidence.
