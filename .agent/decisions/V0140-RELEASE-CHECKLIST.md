# V0.14 Release Checklist

Date: 2026-06-10 19:35 WEST
Owner: manager

Update 2026-06-10 20:10 WEST: the nested-pipeline Noah-MP source fix is merged
and pushed on the manager branch (`c2310c5b`), and the h1-h4 land-gate scorer is
merged/pushed (`c6800bfa`). CPU proof is
`proofs/v014/noahmp_nested_pipeline_activation.*`, verdict
`NOAHMP_NESTED_ACTIVATION_CPU_PROVEN`; 32 focused/regression tests passed after
merge. A lock-based detached watcher is armed to run the post-fix memory
preflight and Canary h1-h4 land gate after the current pre-fix Canary 72h GPU
baseline releases the GPU lock.

## Release Rule

Do not tag v0.14 until the code is stable under the current grid-parity,
memory, and validation gates. The release name is secondary; the invariant goal
is a WRF-faithful-enough, GPU-optimized, near compute- and memory-optimal,
scalable GPU rewrite.

## Current Active Lanes

| Lane | State | Gate before release |
|---|---|---|
| Grid-cell parity | Active closeout. RRTMG `T3D=t` dry-temperature input bug is fixed and proof-bounded: GLW RMSE `17.5203 -> 0.3515 W/m2`; mass-coupled RTHRATEN RMSE `2.4884 -> 0.3646`, max_abs `19.4253 -> 2.7984`. Strict Step-1 remains red/bounded at max_abs `55.9297`, RMSE `0.4997`, p99 `0.9529`; MYNN owns the worst-cell max/floor, while remaining RRTMG is still field-significant. | Commit the RRTMG fix, then record an explicit tolerance-policy decision for the non-bitwise MYNN/RRTMG mass-coupled Step-1 gate. Before long validation, run a short operational all-field rollout falsifier. |
| Memory/FP32 Mythos lane | Closed and manager-merged. Accepted commits: `26815feb` MYNN BouLac tiling + shared RK-stage transport velocities, `bc847db2` default-inert FP32 acoustic precision-mode contract, `8f735a56` proofs/roadmaps/closeout. Exact-branch memory preflight is green on the current candidate branch: `proofs/v014/exact_branch_memory_preflight.md`, verdict `PASS_SHORT_GPU_PREFLIGHT`, peak total VRAM `8858 MiB`, compute app `8159 MiB`, no OOM markers. Mixed FP32 R1/R2 remains blocked until the fp64 validation frontier is fully closed. | Done for v0.14 except final release-note framing: no broad FP32 claim from default-inert scaffolding. FP32 acoustic becomes a v0.15 high-priority implementation lane unless field-gate failure forces a v0.14 revisit. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. Offline Atlas tooling is merged (`07e1ab2e`) and ready for post-parity validation data. Pre-result tolerance candidate is accepted in `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`: ten hard documented fields, static exact/tight checks, and `P/PH/MU/RAINC` critical report-only. | Final scoring uses the accepted manifest, produces summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. A 72h/120h field-parity/stability run is stronger evidence than station-only TOST and is now the primary validation artifact. |
| Switzerland/Gotthard | CPU72 truth is complete at `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`: 73 `wrfout_d01_*`, `rc=0`, `SUCCESS COMPLETE WRF`, last-frame finite PASS. Timing: total wall `2906.3 s`, mainloop `2887.6 s`, 24 dmpar MPI ranks. Resource CSVs are under `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources`; peak 24-rank `wrf.exe` RSS sum `12636.176 MiB`. Proof: `proofs/v014/switzerland_cpu72_reference_resource_summary.md`. | Do not start Switzerland GPU until the post-fix Canary h1-h4 Noah-MP land gate is green/bounded. Gotthard is land dominated, so frozen land `TSK` would invalidate the campaign. |
| Canary field parity | L2 d02 has retained CPU-WRF 72h truth: 15 complete backfill cases in `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`, each with 73 d02 frames. The selected gate case is `20260501_18z_l2_72h_20260519T173026Z`. Prior blockers closed: LBC cadence (`53770411`), PSFC diagnostic, and moist-cqw pressure dynamics (`7c819067`, default ON). Current detached 72h GPU run `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_moistcqw_20260610T171818Z` is continuing as a pre-fix frozen-land baseline, not a release-green gate. Fable h24 review proved `nested_pipeline` never activated Noah-MP; source fix is now merged and pushed in `c2310c5b`, with CPU activation/carry proof green. Commit `c6800bfa` adds a repeatable h1-h4 land-gate scorer. h48 compare still showed no renewed LBC/dycore runaway; formal failures were static `MUB/PB` plus marginal `QVAPOR`. | A lock-based watcher is armed from clean worktree `.claude/worktrees/v014-gpu-gates` (`worker/gpt/v014-gpu-gates @ c6800bfa`) to run post-fix exact-branch memory preflight and Canary h1-h4 land gate as soon as the current GPU lock releases. If green/bounded: rerun full Canary 72h from fixed candidate, then Switzerland 72h GPU. |
| Powered TOST | Three cases are durable; marathon paused. | Secondary station sanity only. TOST is no longer a v0.14 release gate and must not delay or override Switzerland/Canary all-field evidence. |

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

1. Commit the RRTMG dry-temperature input fix and refreshed proof bound.
2. Apply `.agent/decisions/V0140-STEP1-TOLERANCE-POLICY.md`: keep the old
   strict Step-1 tolerance as a diagnostic alarm, not the release-green gate.
   Then run the short grid-field falsifier before launching longer campaigns.
3. CPU-WRF truth is now available for both mandatory 72h gates:
   Switzerland/Gotthard d01 at
   `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
   and selected Canary L2 d02 `20260501_18z_l2_72h_20260519T173026Z`.
4. Run exact-branch memory preflight on the final candidate branch. Current
   candidate is green: `proofs/v014/exact_branch_memory_preflight.md`.
5. Commit/push the accepted root-domain LBC cadence fix
   (`proofs/v014/lbc_cadence_root_cause.*`). Done in `53770411`.
6. Close/bound the 3D moist-cqw pressure-state dynamics lane. Done: moist
   `calc_cq` / `pg_buoy_w` is default ON with GPU h1-h4 proof.
7. Merge the nested-pipeline Noah-MP land activation fix. Done in `c2310c5b`;
   it proves `sf_surface_physics=4` activates/seeds Noah-MP per nested domain
   and prevents the frozen-land `TSK` path from silently running
   (`proofs/v014/noahmp_nested_pipeline_activation.*`).
8. Rerun exact-branch memory preflight and Canary h1-h4 GPU land gate. Armed
   via `/tmp/v014_noahmp_gate_watcher_v2.log` and clean worktree
   `.claude/worktrees/v014-gpu-gates`:
   land-mean `TSK` bias within 2 K and land `HFX` bias within 40 W/m2 at h2-h4.
9. Rerun Canary L2 d02 72h GPU-vs-CPU field-parity/stability from the fully
   fixed candidate branch with resource CSVs.
10. Run Switzerland/Gotthard 72h GPU-vs-CPU field-parity/stability with resource
   CSVs after Canary releases the GPU lock.
11. Run Grid-Delta Atlas on the selected paired cases using the accepted
   pre-result tolerance manifest before claiming equivalence.
12. Optionally resume powered TOST as secondary station evidence and publish it
   together with the atlas if it completes cleanly. It is not a tag gate.
13. Update README, `docs/KNOWN_ISSUES.md`, `PROJECT_PLAN.md`, release notes, and
   proof links.
14. Tag and push v0.14 only after all required gates pass or are honestly
   demoted with a recorded manager decision and independent review.

## Current Do-Not-Run List

- No TOST marathon as a substitute for the mandatory 72h field gates.
- No Switzerland GPU campaign before the post-fix Canary h1-h4 Noah-MP land
  gate and final memory preflight are green/bounded.
- No silent strict-gate tolerance change; any respec needs an explicit manager
  decision and independent review recorded in the roadmap.
- No broad FP32/mixed-precision claim from default-inert scaffolding.
- No station-only equivalence claim without all-cell field evidence.
