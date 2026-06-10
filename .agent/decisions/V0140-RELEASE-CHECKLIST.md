# V0.14 Release Checklist

Date: 2026-06-10 13:45 WEST
Owner: manager

## Release Rule

Do not tag v0.14 until the code is stable under the current grid-parity,
memory, and validation gates. The release name is secondary; the invariant goal
is a WRF-faithful-enough, GPU-optimized, near compute- and memory-optimal,
scalable GPU rewrite.

## Current Active Lanes

| Lane | State | Gate before release |
|---|---|---|
| Grid-cell parity | Active closeout. RRTMG `T3D=t` dry-temperature input bug is fixed and proof-bounded: GLW RMSE `17.5203 -> 0.3515 W/m2`; mass-coupled RTHRATEN RMSE `2.4884 -> 0.3646`, max_abs `19.4253 -> 2.7984`. Strict Step-1 remains red/bounded at max_abs `55.9297`, RMSE `0.4997`, p99 `0.9529`; MYNN owns the worst-cell max/floor, while remaining RRTMG is still field-significant. | Commit the RRTMG fix, then record an explicit tolerance-policy decision for the non-bitwise MYNN/RRTMG mass-coupled Step-1 gate. Before long validation, run a short operational all-field rollout falsifier. |
| Memory/FP32 Mythos lane | Closed and manager-merged. Accepted commits: `26815feb` MYNN BouLac tiling + shared RK-stage transport velocities, `bc847db2` default-inert FP32 acoustic precision-mode contract, `8f735a56` proofs/roadmaps/closeout. Exact-branch memory preflight is green at 8116 MiB compute peak and 378 s warm-cache. Mixed FP32 R1/R2 remains blocked by the open fp64 dynamics frontier. | Rerun exact-branch memory preflight only on the final candidate branch before long validation. Escalate FP32 implementation to Fable/Mythos only if the fp64 grid-parity frontier closes and GPT cannot directly solve mixed precision. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. Offline Atlas tooling is merged (`07e1ab2e`) and ready for post-parity validation data. Pre-result tolerance candidate is accepted in `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`: ten hard documented fields, static exact/tight checks, and `P/PH/MU/RAINC` critical report-only. | Final scoring uses the accepted manifest, produces summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. A 72h/120h field-parity/stability run is stronger evidence than station-only TOST and is now the primary validation artifact. |
| Switzerland/Gotthard | CPU24 proof exists at `/mnt/data/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z`: 25 `wrfout`, `rc=0`, `SUCCESS COMPLETE WRF`, last-frame finite PASS, total wall `1084.6 s`, mainloop `1078.4 s`, peak sampled total WRF-rank RSS `12563.766 MiB`. The 24h run is not resumable: no `wrfrst_d0*`, `restart=.false.`, and boundary times only cover 0-21h. | Build/run a new 72h CPU-WRF truth case, then run matched GPU-JAX 72h and Grid-Delta Atlas with resource CSVs. The first 72h gate uses the 129x129/128-mass-point grid; 151x151 remains a later larger benchmark. |
| Canary field parity | L2 d02 has retained CPU-WRF 72h truth: 15 complete backfill cases in `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`, each with 73 d02 frames. The selected gate case is `20260501_18z_l2_72h_20260519T173026Z`, because it is already the current h1 field-falsifier case and has WRF `rc=0` backfill provenance. d03/1 km retained truth exists but is mainly 24h (`run_hours=24`, no `wrfrst_*`, 4 boundary times), so it is not a clean 72h resume. Inventory: `proofs/v014/canary_cpu_truth_inventory.md`. | Mandatory Canary gate is selected L2 d02 72h CPU-vs-GPU field-parity/stability. Do not launch a new Canary CPU baseline unless this retained truth is disproven. d03/1 km is a stretch after the two 72h release gates are green. |
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
3. Build/run or confirm matched CPU-WRF truth for both mandatory 72h gates:
   Switzerland/Gotthard d01 and selected Canary L2 d02
   `20260501_18z_l2_72h_20260519T173026Z`.
4. Run exact-branch memory preflight on the final candidate branch.
5. Run Switzerland/Gotthard 72h GPU-vs-CPU field-parity/stability with resource
   CSVs.
6. Run Canary L2 d02 72h GPU-vs-CPU field-parity/stability with resource CSVs.
7. Run Grid-Delta Atlas on the selected paired cases using the accepted
   pre-result tolerance manifest before claiming equivalence.
8. Optionally resume powered TOST as secondary station evidence and publish it
   together with the atlas if it completes cleanly. It is not a tag gate.
9. Update README, `docs/KNOWN_ISSUES.md`, `PROJECT_PLAN.md`, release notes, and
   proof links.
10. Tag and push v0.14 only after all required gates pass or are honestly
   demoted with a recorded manager decision and independent review.

## Current Do-Not-Run List

- No TOST marathon as a substitute for the mandatory 72h field gates.
- No Switzerland GPU campaign before the matching 72h CPU truth and final
  memory preflight are available.
- No silent strict-gate tolerance change; any respec needs an explicit manager
  decision and independent review recorded in the roadmap.
- No broad FP32/mixed-precision claim from default-inert scaffolding.
- No station-only equivalence claim without all-cell field evidence.
