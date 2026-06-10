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
| Memory/FP32 Mythos lane | Closed and manager-merged. Accepted commits: `26815feb` MYNN BouLac tiling + shared RK-stage transport velocities, `bc847db2` default-inert FP32 acoustic precision-mode contract, `8f735a56` proofs/roadmaps/closeout. Exact-branch memory preflight is green on the current candidate branch: `proofs/v014/exact_branch_memory_preflight.md`, verdict `PASS_SHORT_GPU_PREFLIGHT`, peak total VRAM `8858 MiB`, compute app `8159 MiB`, no OOM markers. Mixed FP32 R1/R2 remains blocked until the fp64 validation frontier is fully closed. | Done for v0.14 except final release-note framing: no broad FP32 claim from default-inert scaffolding. FP32 acoustic becomes a v0.15 high-priority implementation lane unless field-gate failure forces a v0.14 revisit. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. Offline Atlas tooling is merged (`07e1ab2e`) and ready for post-parity validation data. Pre-result tolerance candidate is accepted in `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`: ten hard documented fields, static exact/tight checks, and `P/PH/MU/RAINC` critical report-only. | Final scoring uses the accepted manifest, produces summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. A 72h/120h field-parity/stability run is stronger evidence than station-only TOST and is now the primary validation artifact. |
| Switzerland/Gotthard | CPU72 truth is complete at `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`: 73 `wrfout_d01_*`, `rc=0`, `SUCCESS COMPLETE WRF`, last-frame finite PASS. Timing: total wall `2906.3 s`, mainloop `2887.6 s`, 24 dmpar MPI ranks. Resource CSVs are under `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources`; peak 24-rank `wrf.exe` RSS sum `12636.176 MiB`. Proof: `proofs/v014/switzerland_cpu72_reference_resource_summary.md`. | Next serial GPU job after Canary frees the GPU: run matched GPU-JAX 72h and Grid-Delta Atlas with resource CSVs using the command recorded in `docs/GPU_RUNBOOK.md`. The first 72h gate uses the 129x129/128-mass-point grid; 151x151 remains a later larger benchmark. |
| Canary field parity | L2 d02 has retained CPU-WRF 72h truth: 15 complete backfill cases in `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`, each with 73 d02 frames. The selected gate case is `20260501_18z_l2_72h_20260519T173026Z`, because it is already the h1 field-falsifier case and has WRF `rc=0` backfill provenance. The first detached Canary d02 72h GPU run at `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z` was intentionally stopped at h26 (`gpu_rc=143`) after h08/h10/h18/h24 `FAIL`. Fable high proved and fixed the LBC cadence root cause; commit `53770411` is pushed. The fixed LBC characterization run at `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z` was stopped after h24 (`gpu_rc=143`) because it used the old `PSFC` diagnostic. Fable high then fixed `PSFC`: WRF runtime `PSFC` is moist hydrostatic `grid%p_hyd_w(kts)`, not `P+PB` extrapolation. Short GPU h1-h4 validation is green for this diagnostic lane: run root `/mnt/data/wrf_gpu_validation/v014_canary_d02_psfcfix_h4_20260610T160708Z`, GPU rc `0`, peak VRAM `15507 MiB`, `PSFC` h1/h4 RMSE `57.823/35.487 Pa`. | Close/bound the deeper 3D pressure-state lane: operational acoustic w-equation still uses dry `cqw` / `pg_buoy_w_dry`. Do not launch Switzerland GPU or claim field parity until moist-cqw dynamics is fixed or formally bounded. |
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
6. Close/bound the 3D moist-cqw pressure-state dynamics lane (`dry_cqw` /
   `pg_buoy_w_dry` currently active in the acoustic w-equation).
7. Rerun Canary L2 d02 72h GPU-vs-CPU field-parity/stability from the fully
   fixed candidate branch with resource CSVs.
8. Run Switzerland/Gotthard 72h GPU-vs-CPU field-parity/stability with resource
   CSVs after Canary releases the GPU lock.
9. Run Grid-Delta Atlas on the selected paired cases using the accepted
   pre-result tolerance manifest before claiming equivalence.
10. Optionally resume powered TOST as secondary station evidence and publish it
   together with the atlas if it completes cleanly. It is not a tag gate.
11. Update README, `docs/KNOWN_ISSUES.md`, `PROJECT_PLAN.md`, release notes, and
   proof links.
12. Tag and push v0.14 only after all required gates pass or are honestly
   demoted with a recorded manager decision and independent review.

## Current Do-Not-Run List

- No TOST marathon as a substitute for the mandatory 72h field gates.
- No Switzerland GPU campaign before the matching 72h CPU truth and final
  memory preflight are available.
- No silent strict-gate tolerance change; any respec needs an explicit manager
  decision and independent review recorded in the roadmap.
- No broad FP32/mixed-precision claim from default-inert scaffolding.
- No station-only equivalence claim without all-cell field evidence.
