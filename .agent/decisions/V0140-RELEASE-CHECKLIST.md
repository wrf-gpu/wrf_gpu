# V0.14 Release Checklist

Date: 2026-06-11 05:05 WEST
Owner: manager

Update 2026-06-11 16:42 WEST: v0.14 release remains blocked by the
Switzerland/Gotthard h36 strong-flow dry-dynamics residual. GPT advance-w term
split completed as `NARROWED_NO_FIX`:
`.agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md` and
`proofs/v014/switzerland_advance_w_term_split.{py,json}`. The proof rejects
post-`advance_mu_t` mass inputs as the primary creator: forcing WRF-call-21602
`mu/muts/muave` into `advance_w` improves `p` by only `0.112%` and `ph` by
`0.0018%`. Surface-`w`, moist coefficient choice, `calc_p_rho` denominator,
and `smdiv` are also not primary. The remaining target is inside
`advance_w_wrf()` or an unexposed input, with strongest clues around
`rw_tend`/vertical PGF-buoyancy and `ph_tend` contribution into RHS/implicit
solve. Next required proof loop is a short WRF-native call-`21601 -> 21602`
intra-`advance_w` dump and Python-harness comparison. Do not launch
Switzerland 72h, TOST, FP32 implementation, or the Fable/GPT performance audits
until this short-gate blocker is fixed or explicitly bounded.

Update 2026-06-11 05:05 WEST: Canary L2 d02 72h is no longer in flight; it is
accepted as `PROCEED_BOUNDED_WITH_FOLLOWUP` in
`proofs/v014/canary_d02_72h_field_gate_summary.md`. Run root:
`/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`.
GPU `rc=0`, compare `rc=0`, atlas `rc=0`, 72 paired hours, peak total GPU
memory `21108 MiB`, process RSS peak `20950 MiB`, GPU wall `8244 s`.
Remaining Canary misses are bounded/static or saturating (`MUB/PB` nest-frame
seam, `QVAPOR` slightly over hard limit) and are not a current v0.14 launch
blocker.

Canary wall-clock benchmark is now recorded in the proof summary: GPU
`wall_clock_l2_d02.json` reports `8226.936 s` total and `8152.310 s`
forecast-only for 72h. The retained CPU-WRF 28-rank backfill lacks an explicit
`cpu_timing.json`; using first-to-last `wrfout_d02` timestamps gives an honest
approximate CPU denominator of `8713.126 s`. Current Canary gate speedup is only
`1.059x` GPU-total or `1.069x` GPU-forecast-only, so v0.14 must frame this run
as field-stability evidence, not a major speed headline.

Update 2026-06-11: principal review escalated the weak Canary speedup from
"documentation caveat" to a v0.14 release decision gate. Before tagging, the
manager must understand whether the weak number is measurement unfairness,
small-grid/strong-CPU regime, compile/cache/IO/orchestration overhead, or a real
GPU-kernel efficiency regression introduced by v0.13/v0.14 correctness and
memory work. Two independent audit contracts are prepared:
`.agent/sprints/2026-06-11-v014-fable-performance-regression-audit/` and
`.agent/sprints/2026-06-11-v014-gpt-performance-regression-audit/`. Do not make
public speedup claims until their findings are reconciled and either a fair
`>2x` benchmark is produced or the release notes explicitly demote speed claims.

Update 2026-06-11 (principal clarification): do not start the Fable+GPT
performance-analysis pair until Switzerland/Gotthard 72h has passed or has been
explicitly accepted/bounded. Once it does, launch both in parallel. GPT is
report-only. Fable's endpoint is a **high-performance identical** v0.14
candidate: it may implement only simple, obviously identity-preserving speedups
(debug/status removal from hot paths, safe caching, redundant initialization
removal, mathematically identical simplifications, unnecessary synchronization
removal). Complex/high-yield work stays in the report and v0.15 roadmap. After
manager review/merge of any Fable simple-speed patch, rerun Canary and
Switzerland GPU gates with all safe caches enabled to get the true current
maximum speed. That measured value is the v0.14 speed number; do not delay
v0.14 for complex optimization beyond this simple-speed pass.

Update 2026-06-11 (performance premise): when the Fable+GPT performance pair
starts, both agents must also work backwards from the original speed premise.
For sufficiently parallel regional-NWP kernels, a GPU should normally beat a CPU
by large factors; if this implementation cannot reach the rough
pre-architecture `~10x` expectation, the reports must explain why with evidence.
Acceptable explanations are unfair measurement/small-grid overhead, current
kernel/runtime inefficiency, IO/transfer/orchestration overhead, or a real
algorithmic limit. They must also assess whether a kernel-architecture change
-- graph/stencil/matrix representation, custom Triton/CUDA/Pallas kernels,
persistent kernels, data-layout rewrite, larger fusion boundaries, or physics
batching -- could move the system toward computational near-optimum efficiency.
Compute speed has strategic priority over extra memory savings: optional
caching/precompute/residency modes are desirable if they give major speedups and
remain below stable memory targets for RTX 5090, H200, or GB300-class hardware.

Switzerland/Gotthard is the active blocker. The first 72h GPU run exposed an
hourly driver LBC-clock bug; the fix is merged and proven (`9cbdfe31`,
`eaff102c`, `tests/test_daily_boundary_clock.py`,
`proofs/v014/switzerland_lbc_clock_root_cause.*`). The post-fix Switzerland
72h rerun no longer has boundary-ring drift, but still fails due to a
dry-dynamics strong-flow mass-venting residual in the h36-h72 Alpine storm
window. Accepted proof:
`.agent/reviews/2026-06-11-v014-switzerland-post-lbc-residual-fable.md` and
`proofs/v014/switzerland_post_lbc_residual.*` (`333661f6`). Latest merged
debug status: the HPG-input WRF-faithfulness fix (`3d0b439c`,
`hypsometric_opt=2` LOG-form `al/alt/p` + LOG base `alb`) and the real-case
`rhs_ph` / edge-faithful stage-omega fix (`79b0c22e`) are merged via
`82f6b703`. These are real WRF-anchored subfixes: `rhs_ph`'s horizontal
geopotential advection now matches the WRF real-case map-factored order<=6
specified-boundary operator to machine precision, and specified-domain stage
omega no longer uses periodic-wrap band values. The h36->h37 residual improved
from `-27.697` to `-21.883 Pa/cell/h` versus the hypso baseline, but excess
outflux only improved `-28.328 -> -27.204 Pa/cell/h`, so the venting blocker is
still open. Current next endpoint: stage-3/end-of-step wrapper cadence and
residual lateral-band amplifier; use the exact handoff in
`.agent/sprints/2026-06-11-v014-switzerland-acoustic-substep-continuation/manager-handoff.md`.

Do not start a new 72h Switzerland GPU gate until the h36 strong-flow short
gate collapses the excess dry-mass venting or the manager records an explicit
bounded-release decision with independent review. Do not start the Fable xhigh
kernel efficiency review until this correctness blocker is closed or formally
bounded enough for v0.14.

Update 2026-06-11 (Fable token discipline + latest h36 status): the Fable xhigh
stage-3/wrapper-cadence sprint completed at worker commit `a5f28252` in
`/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`. It
delivered a WRF-specified LBC cadence + specified-boundary advection-degradation
patch that is flag-gated/default-off and term-proves large boundary-band
improvements against WRF dumps, but it **does not close** the h36->h37 venting
gate. Its hourly evidence indicates the previous wrong boundary band was partly
compensating an interior dry-mass sink. The active root lane is now the interior
acoustic `advance_w` / `phi` hydrostatic sink/pressure-rise pair, with
`rw_tend`/`ph_tend` consumption and post-stage `calc_p_rho`/grid-p refresh as
secondary suspects. GPT-5.5 xhigh verifier report
`proofs/v014/gpt_stage3_wrapper_verifier.md` accepted the boundary/advection
proof only after a local `dry_spec_only` specified-`w` wrapper correction; that
correction is committed on the Fable branch as `019bc71b` (`tests/test_v014_specified_bdy_cadence.py`
now `6 passed`). Do not immediately launch another Fable/Mythos xhigh sprint.
Continue first with GPT-5.5 xhigh verification/residual-fix sprints on the
`advance_w`/`phi` discriminator, then Fable medium/high if GPT stalls. Reserve
another Fable/Mythos xhigh run for roughly ten inconclusive cheaper follow-up
sprints or an explicitly documented exceptional kernel-level impasse.

Update 2026-06-10 22:49 WEST: the nested-pipeline Noah-MP source fix and h1-h4
land-gate scorer are merged/pushed (`c2310c5b`, `c6800bfa`). The d01 LU16
Noah-MP nonfinite blocker is now closed by Fable high end-to-end:
`22a2cc0c` fixes WATER 1-based soil/veg category indexing,
`aff7d124`/`5a708074` record closure proof and GPU confirmation. Root cause:
ISLTYP=1 sand read row 0 of WRF's 1-based soil parameter table because WATER
used `category - 1`, yielding `SMCMAX=0` and NaNs; all other soil categories
were also one hydraulic row off. GPU confirmation
`/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`
is `rc=0`, `PASS_SHORT_GPU_PREFLIGHT` / `PIPELINE_GREEN`,
all_domains_finite=true, peak total VRAM `9783 MiB`. The post-fix Canary h1-h4
Noah-MP land gate is accepted:
`/mnt/data/wrf_gpu_validation/v014_canary_d02_noahmp_lu16fix_h4_20260610T212056Z`,
`NOAHMP_NESTED_GPU_H4_ACCEPT`, peak GPU memory `20975 MiB`; h2-h4 land TSK and
HFX biases are inside thresholds. The full Canary L2 d02 72h GPU run is now in
flight at
`/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
with resource CSV logging and an armed h24 intermediate grid compare.

Operational note: reports rooted in `/home/enric/src/canairy_waves` are from a
different project and are ignored for this roadmap.

## Release Rule

Do not tag v0.14 until the code is stable under the current grid-parity,
memory, and validation gates. The release name is secondary; the invariant goal
is a WRF-faithful-enough, GPU-optimized, near compute- and memory-optimal,
scalable GPU rewrite.

## Current Active Lanes

| Lane | State | Gate before release |
|---|---|---|
| Grid-cell parity | Active closeout. RRTMG `T3D=t` dry-temperature input bug is fixed and proof-bounded: GLW RMSE `17.5203 -> 0.3515 W/m2`; mass-coupled RTHRATEN RMSE `2.4884 -> 0.3646`, max_abs `19.4253 -> 2.7984`. Strict Step-1 remains red/bounded at max_abs `55.9297`, RMSE `0.4997`, p99 `0.9529`; MYNN owns the worst-cell max/floor, while remaining RRTMG is still field-significant. | Commit the RRTMG fix, then record an explicit tolerance-policy decision for the non-bitwise MYNN/RRTMG mass-coupled Step-1 gate. Before long validation, run a short operational all-field rollout falsifier. |
| Memory/FP32 Mythos lane | Closed and manager-merged. Accepted commits: `26815feb` MYNN BouLac tiling + shared RK-stage transport velocities, `bc847db2` default-inert FP32 acoustic precision-mode contract, `8f735a56` proofs/roadmaps/closeout. Latest post-LU16-fix exact-branch preflight is green (`/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`: `rc=0`, peak total VRAM `9783 MiB`, all domains finite, no OOM markers). Mixed FP32 R1/R2 remains blocked until the fp64 validation frontier is fully closed. | Done for v0.14 except final release-note framing. No broad FP32 claim from default-inert scaffolding. FP32 acoustic becomes a v0.15 high-priority implementation lane unless field-gate failure forces a v0.14 revisit. |
| Validation tooling | Grid-Delta Atlas gate is specified in `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. GPU runbook exists in `docs/GPU_RUNBOOK.md`. Offline Atlas tooling is merged (`07e1ab2e`) and ready for post-parity validation data. Pre-result tolerance candidate is accepted in `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`: ten hard documented fields, static exact/tight checks, and `P/PH/MU/RAINC` critical report-only. | Final scoring uses the accepted manifest, produces summary, markdown report, compact plots, and README-ready dashboard for all common numeric wrfout fields. A 72h/120h field-parity/stability run is stronger evidence than station-only TOST and is now the primary validation artifact. |
| Switzerland/Gotthard | CPU72 truth is complete at `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`: 73 `wrfout_d01_*`, `rc=0`, `SUCCESS COMPLETE WRF`, last-frame finite PASS. Timing: total wall `2906.3 s`, mainloop `2887.6 s`, 24 dmpar MPI ranks. Resource CSVs are under `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources`; peak 24-rank `wrf.exe` RSS sum `12636.176 MiB`. LBC-clock bug fixed and proven. HPG native-face mismatch fixed but refuted as blocker. Real-case `rhs_ph` + edge-faithful stage omega + WRF dycore constants are merged/proven and improve residual `-27.697 -> -21.883 Pa/cell/h`, but excess outflux only moves ~4%, so blocker remains open. Fable xhigh then proved stage/wrapper boundary cadence + specified advection fixes collapse boundary-band errors, but the hourly gate still fails and points to an interior acoustic `advance_w`/`phi` sink. GPT verifier accepted that conclusion after one local Fable-branch correction (`019bc71b`). GPT advance-w term split now rejects post-`advance_mu_t` mass inputs as primary and narrows the remaining target to intra-`advance_w_wrf()` terms or an unexposed input. | Do not rerun 72h yet. Next gate is a WRF-native call-`21601 -> 21602` intra-`advance_w` dump for RHS, `rw_tend`/`ph_tend`, implicit coefficients, Thomas solve, finished `ph`, and immediate `calc_p_rho` `p/al/alt`, then compare it against the Python harness before h36 storm-state short-gate collapse/bounding and the 72h GPU rerun. |
| Canary field parity | L2 d02 has retained CPU-WRF 72h truth: 15 complete backfill cases in `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`, each with 73 d02 frames. The selected gate case is `20260501_18z_l2_72h_20260519T173026Z`. Prior blockers closed: LBC cadence (`53770411`), PSFC diagnostic, moist-cqw pressure dynamics (`7c819067`, default ON), nested Noah-MP activation (`c2310c5b`), and the d01 LU16/sand nonfinite blocker (`22a2cc0c` + `aff7d124` + `5a708074`). The post-fix 72h GPU run completed at `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`; proof summary `proofs/v014/canary_d02_72h_field_gate_summary.md`; atlas `rc=0`. | Accepted as bounded/proceed. Keep plots/benchmarks for release docs; do not spend correctness tokens here unless a later review overturns the bounded decision. |
| Performance regression | Prepared, conditional on Switzerland passing. Canary 72h shows only `1.059x` to `1.069x` against an approximate 28-rank CPU denominator, far below the project speed premise and below the old v0.12 speed story. | After Switzerland/Gotthard 72h is green/bounded, run the prepared independent Fable and GPT performance audits in parallel. Fable may implement simple identity-preserving speedups; GPT reports only. Reconcile findings, rerun Canary+Switzerland with all safe caches enabled, and use that measured current-max speed for v0.14 while moving complex optimization to v0.15. Both reports must explain `WHY_NOT_10X_YET`, `NEAR_OPTIMUM_KERNEL_PATHS`, and `COMPUTE_OVER_MEMORY_OPTIONS`. |
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
4. Run exact-branch memory preflight on the final candidate branch. Done after
   LU16 fix: `/mnt/data/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`,
   `rc=0`, `PASS_SHORT_GPU_PREFLIGHT`, peak total VRAM `9783 MiB`.
5. Commit/push the accepted root-domain LBC cadence fix
   (`proofs/v014/lbc_cadence_root_cause.*`). Done in `53770411`.
6. Close/bound the 3D moist-cqw pressure-state dynamics lane. Done: moist
   `calc_cq` / `pg_buoy_w` is default ON with GPU h1-h4 proof.
7. Merge the nested-pipeline Noah-MP land activation fix. Done in `c2310c5b`;
   it proves `sf_surface_physics=4` activates/seeds Noah-MP per nested domain
   and prevents the frozen-land `TSK` path from silently running
   (`proofs/v014/noahmp_nested_pipeline_activation.*`).
8. Fix or exactly bound the d01 LU16 Noah-MP nonfinite preflight failure. Done:
   root cause/fix in `22a2cc0c`, proof/GPU confirmation in
   `aff7d124`/`5a708074`.
9. Rerun Canary h1-h4 GPU land gate: land-mean `TSK` bias within 2 K and land
   `HFX` bias within 40 W/m2 at h2-h4. Done:
   `/mnt/data/wrf_gpu_validation/v014_canary_d02_noahmp_lu16fix_h4_20260610T212056Z`,
   `NOAHMP_NESTED_GPU_H4_ACCEPT`.
10. Rerun Canary L2 d02 72h GPU-vs-CPU field-parity/stability from the fully
   fixed candidate branch with resource CSVs. Done and bounded/proceed:
   `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`.
11. Close Switzerland strong-flow dry-dynamics residual with the h36 short gate.
   Merged subfixes (`3d0b439c`, `79b0c22e`, merge `82f6b703`) close the HPG
   native-face and real-case `rhs_ph`/stage-omega defects, but not the gate.
   Fable worker commit `a5f28252` proves the stage/wrapper boundary cadence and
   specified advection lane is real but not the venting driver; GPT verifier
   report `proofs/v014/gpt_stage3_wrapper_verifier.md` required and received
   the local specified-`w` correction `019bc71b`. GPT advance-w term split
   rejects post-`advance_mu_t` mass inputs as primary. Current active lane:
   generate a WRF-native intra-`advance_w` dump at call `21601 -> 21602` and
   compare RHS, `rw_tend`/`ph_tend`, implicit coefficients, Thomas solve,
   finished `ph`, and immediate `calc_p_rho` `p/al/alt` against the Python
   harness. Do not start the 72h gate until the h36 short-gate residual
   materially collapses or is formally bounded.
12. Rerun Switzerland/Gotthard 72h GPU-vs-CPU field-parity/stability with
   resource CSVs after the h36 strong-flow gate is fixed or formally bounded.
13. Run Grid-Delta Atlas on the selected paired cases using the accepted
   pre-result tolerance manifest before claiming equivalence. The release
   artifact must include stable-through-time plots for all common numeric
   `wrfout` fields and volumes, not only scalar summaries.
14. Record latest CPU-vs-GPU wall-clock benchmarks for both mandatory regions:
   Canary L2 d02 CPU truth vs GPU 72h, and Switzerland/Gotthard d01 CPU truth
   vs GPU 72h. Include wall-clock, forecast-hours/hour, peak GPU memory, peak
   process RSS, and CPU peak RSS where available. Canary is recorded in
   `proofs/v014/canary_d02_72h_field_gate_summary.md`; Switzerland remains
   blocked pending the h36 strong-flow HPG fix and the final GPU72 rerun.
15. Close the v0.14 performance-regression decision gate after Switzerland is
    green/bounded. Run the prepared Fable and GPT audits in parallel. Fable may
    implement only simple identity-preserving speedups; GPT is report-only.
    Manager reviews/merges any Fable simple-speed patch, then reruns Canary and
    Switzerland GPU gates with all safe caches enabled. Use that measured
    current-max speed for v0.14. Do not delay v0.14 for complex optimization
    beyond this simple-speed pass; move complex/high-yield work to v0.15. Both
    reports must include `WHY_NOT_10X_YET`, `NEAR_OPTIMUM_KERNEL_PATHS`, and
    `COMPUTE_OVER_MEMORY_OPTIONS` so README and v0.15 can honestly state the
    measured speed, plausible speed ceiling, and engineering route toward
    near-optimum GPU efficiency.
16. Optionally resume powered TOST as secondary station evidence and publish it
   together with the atlas if it completes cleanly. It is not a tag gate.
17. Start the prepared Fable/Mythos xhigh kernel memory/compute efficiency
   review only after Canary and Switzerland are both green/bounded enough that
   no scarce Fable tokens are needed for v0.14 correctness debugging. Its
   output feeds the complete v0.15 roadmap, not v0.14 source changes.
18. Write/update `.agent/decisions/V0150-ROADMAP-DRAFT.md` with the complete
    list of deferred, easy, and high-value kernel improvements, preserving the
    full candidate list for principal review. Include estimates from the Fable
    and GPT reports for current small cases on RTX 5090, optimal RTX 5090
    32 GB grids that fit in VRAM, and asymptotic H200/GB300 large-grid regimes
    where initialization/compile is amortized.
19. Send a paper/documentation worker to update the paper to the latest facts:
   remove stale relativizations, describe the new 72h field-parity/stability
   validation method, include the wall-clock/memory benchmarks, and reference
   the Grid-Delta Atlas plots.
20. Update README, `docs/KNOWN_ISSUES.md`, `PROJECT_PLAN.md`, release notes, and
    proof links. README must include the measured v0.14 Canary/Switzerland
    current-max speed numbers plus the clearly labelled projection table for
    small RTX 5090 cases, optimal RTX 5090 in-VRAM grids, and asymptotic
    H200/GB300 large-grid cases.
21. Tag and push v0.14 only after all required gates pass or are honestly
   demoted with a recorded manager decision and independent review.

## Current Do-Not-Run List

- No TOST marathon as a substitute for the mandatory 72h field gates.
- No Switzerland 72h GPU rerun before the h36 strong-flow dry-dynamics short
  gate is fixed or formally bounded.
- No Fable/Mythos xhigh efficiency review while v0.14 still needs Fable for a
  correctness blocker.
- No silent strict-gate tolerance change; any respec needs an explicit manager
  decision and independent review recorded in the roadmap.
- No broad FP32/mixed-precision claim from default-inert scaffolding.
- No station-only equivalence claim without all-cell field evidence.
