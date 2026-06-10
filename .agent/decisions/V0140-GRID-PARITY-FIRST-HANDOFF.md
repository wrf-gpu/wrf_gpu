# V0.14 Grid-Parity-First Handoff

Date: 2026-06-08 23:11 WEST
Owner: manager

Update 2026-06-08 23:20 WEST: the principal authorized unlimited time, CPU/GPU
use, and parallel agents within resource sanity. After two failed GPT
search/debug attempts on the same root-cause problem, try one targeted Opus
4.8 xhigh/max run via `claude --permission-mode auto` if tokens are available.

Update 2026-06-09 10:40 WEST: the live-nest base-source fix is confirmed useful
but **not** a grid-symptom closer. Two fresh GPT proof sprints are closed and
committed:

- `fab94a79` / `proofs/v014/same_state_momentum_mass.*`: selected h10 dynamic
  state already mismatches CPU-WRF at `post_after_all_rk_steps_pre_halo`;
  first failing field is `U`, max_abs `6.292358893898424`, RMSE
  `2.032497018496295`.
- `6d1f7cf9` / `proofs/v014/grid_after_live_nest_base.*`: one bounded h12 GPU
  run after the base-source fix is `L2_D02_GREEN`, but direct d02 h1-h12 grid
  comparison verdict is `GRID_SYMPTOM_NOT_CLOSED`. `V10` RMSE remains
  `2.55039100124724` m/s, worst h11 RMSE `4.277008742661733`; `PSFC/P/MU/PH`
  remain large.

Therefore TOST stays paused. The manager cadence now requires a targeted Opus
xhigh critic/debugger before the next root-cause conclusion. Sprint:
`.agent/sprints/2026-06-09-v014-dynamic-root-cause-opus-critic/sprint-contract.md`.

Update 2026-06-09 10:55 WEST: Opus critic completed and was accepted
(`a101a2bb`). Verdict:
`MANAGER_FINAL_RK_TARGET_NOT_JUSTIFIED_INPUT_ALREADY_DIVERGED`. The critic
found that `proofs/v014/pre_rk_input_boundary.json` already shows the JAX input
to step 6000 is divergent before final RK (`MU'` worst about `267` Pa, `P'`
worst about `590` Pa, `T_OLD` worst about `6.2` K). Therefore final-RK output
instrumentation is too late as the next proof boundary. Active sprint is now
`.agent/sprints/2026-06-09-v014-same-input-single-rk-parity/sprint-contract.md`:
WRF pre-RK input -> one JAX dynamics step -> WRF post-RK/pre-halo output, with
tendency-control and patch-width blockers preferred over a weak comparison.

Update 2026-06-09 10:58 WEST: the same-input sprint is closed and pushed as
`6bc6402a`. Verdict:
`SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.
No comparison was run because the current WRF pre-RK hook emits only `MASS_K1`
`T/P/PB/MU/MUB`-related fields, not full native `U/V/W/PH/PHB`, full columns,
controlled `Tendencies`, `DryPhysicsTendencies`, or an `OperationalCarry`
loader. The active next sprint is now
`.agent/sprints/2026-06-09-v014-full-pre-rk-savepoint-hook/sprint-contract.md`:
build a full WRF pre-RK native-state/tendency savepoint and proof-only JAX
loader, then rerun the strict same-input one-step boundary or name the next
exact blocker.

Update 2026-06-09 11:36 WEST: the full pre-RK savepoint sprint completed.
Final verdict:
`FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`. CPU-WRF successfully
compiled, ran to `2026-05-02_04:00:00`, and emitted two full native pre-RK hook
files at `d02` step `6000`; duplicate tile overlap max delta is `0.0`. Full dry
state, active moisture, and scalar records are present. The strict one-step JAX
comparison still did not run because WRF has not yet produced current-step
`*_tendf`, `h_diabatic`, `*_save`, `moist_old`, and `scalar_old` leaves at the
step-entry boundary. Next sprint is not broad debugging: place a second WRF
source/save hook after those leaves exist and before any dynamics state mutation,
or prove the comparison boundary must move.

Update 2026-06-09 11:45 WEST: source/save-boundary sprint opened:
`.agent/sprints/2026-06-09-v014-source-save-boundary/sprint-contract.md`. This
is the active next debug step. It must find the first WRF boundary where
current-step source/save leaves exist and either preserve the step-entry native
state or move the whole comparison boundary consistently. No production
`src/gpuwrf/**` edits are authorized.

Update 2026-06-09 12:15 WEST: source/save-boundary sprint completed. Final
verdict:
`SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.
WRF now emits the current-step dry source/save leaves at a valid boundary after
`first_rk_step_part1/part2` and `rk_tendency`, before dry/acoustic mutation.
Native dry state preservation versus the full pre-RK savepoint is exact on
overlap, worst max abs `0.0`. The strict JAX comparison still did not run. The
remaining blocker is proof construction: full-domain same-boundary
carry/boundary leaves, full-domain/full-vertical truth, a proof-only wrapper
into `_rk_scan_step_with_pre_halo_capture`, and `scalar_old`/old-field handling.
The next sprint should not edit dycore broadly; it should build that wrapper and
truth surface or name the next exact blocker.

Update 2026-06-09 12:18 WEST: full-domain source/save wrapper sprint opened:
`.agent/sprints/2026-06-09-v014-full-domain-source-wrapper/sprint-contract.md`.
It is instrumentation/proof-only, CPU-only, and production `src/gpuwrf/**`
read-only. The gate is a strict same-input single-RK comparison against
WRF-emitted post-RK/pre-halo truth, or one exact blocker naming the missing
wrapper contract, field, old-field strategy, boundary conflict, or patch width.

Update 2026-06-09 12:24 WEST: Opus management review completed:
`.agent/reviews/2026-06-09-v014-management-review-01.md`. Goal-change gate is
`NO_GOAL_CHANGE`, but the path needs resequencing. The review accepts grid-first
and TOST-hold, but flags that the same-input discriminator became a 4-sprint
blocked instrumentation ladder at the hardest step-6000/h10 instance. Manager
adopts the method correction: let the current full-domain wrapper sprint finish
because it is already running and may produce a strict result or exact blocker;
after that, do not open another one-blocker micro-sprint. The next decisive
debug step must consolidate early-step same-input parity plus drift-onset
bisection from shared `wrfinput` (for example steps 1, 60, 600, 3000, 5999) and
must execute at least one strict comparison or name all remaining blockers in one
pass.

Update 2026-06-09 12:26 WEST: the next-sprint plan is now staged in
`.agent/decisions/V0140-EARLY-STEP-DISCRIMINATOR-PLAN.md`. It is a pending
plan, not a second active sprint while the full-domain wrapper worker runs.

Update 2026-06-09 12:35 WEST: full-domain source-wrapper sprint closed
fail-closed. Verdict:
`FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.
Existing step-6000 source/save and post-RK surfaces are patch-only and lack the
same-boundary full wrapper carry/boundary leaves. No strict JAX comparison ran.
Do not continue the step-6000 wrapper ladder; open the staged early-step
same-input discriminator.

Update 2026-06-09 12:38 WEST: early-step same-input discriminator sprint
opened:
`.agent/sprints/2026-06-09-v014-early-step-discriminator/sprint-contract.md`.
This is the active next debug step and replaces the step-6000 wrapper ladder.

Update 2026-06-09 12:45 WEST: early-step same-input discriminator sprint closed
fail-closed. Verdict:
`EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.
The proof covers candidate steps `1`, `60`, `600`, `3000`, and `5999` in one
pass. No strict comparison ran; weak WRF-output, JAX-vs-JAX, one-cell, and
mixed-source comparisons were avoided. Common blockers:
CPU-only real-case replay loader is GPU-only at `State.zeros`, no candidate-step
WRF post-RK/pre-halo full-field surface exists, no WRF-controlled same-input
`OperationalCarry` sequence exists, and the field/staggering schema is not yet
frozen. Next debug step is therefore a tooling/contract sprint: build a
CPU-compatible proof loader or checkpoint reader plus candidate-step WRF
post-RK/pre-halo full-field truth surface, then rerun the discriminator.

Update 2026-06-09 12:50 WEST: same-input contract-builder sprint opened:
`.agent/sprints/2026-06-09-v014-same-input-contract-builder/sprint-contract.md`.
This is a debug-tooling sprint, not a production source-fix sprint. It must
build or precisely block the CPU-compatible proof loader/checkpoint reader, WRF
candidate-step post-RK/pre-halo full-field surface, and frozen field/staggering
schema needed to make same-input comparisons cheap and strict.

Update 2026-06-09 13:03 WEST: same-input contract-builder sprint closed
fail-closed. Verdict:
`SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.
The JAX/CPU side is now ready for initial d02 same-input construction:
`State`, `Tendencies`, `BaseState`/metrics, `OperationalNamelist`, parent
boundary package, and initial `OperationalCarry` build under CPU-only JAX
without `State.zeros`. The WRF/JAX schema is frozen for 16 comparison fields.
No strict numerical comparison ran because the required full-domain CPU-WRF d02
step-1 `post_after_all_rk_steps_pre_halo` truth surface does not exist. The next
validation-enabling sprint is a disposable CPU-WRF step-1 truth hook that emits
`/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`,
then reruns `proofs/v014/same_input_contract_builder.py`.

Update 2026-06-09 13:32 WEST: step-1 same-input truth sprint closed with the
first strict full-domain comparison. Verdict:
`STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.
The accepted comparison is CPU-WRF d02 step-1 post-RK/pre-halo truth versus JAX
one-step `_rk_scan_step_with_pre_halo_capture(...).pre_halo_state`, not JAX
initial state. The truth npz exists at
`/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.
First divergent schema field is `T`; largest residuals are base/mass fields:
`MUB` max_abs `2635.640625`, `PB` `2627.3828125`, `PHB` `2237.9423828125`,
and `P` `1561.1123921205437`. This points the next sprint back to native
live-nest child base-state initialization or a decisive init-override falsifier
before any late-RK, FP32, memory, Switzerland, or TOST work.

Update 2026-06-09 13:51 WEST: step-1 live-nest init rerun sprint closed.
Verdict:
`STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`. The accepted strict
comparison was rerun with native live-nest child base initialization semantics
mirrored in the CPU-only proof loader. The large raw-init base residuals are now
closed: `MUB` max_abs `0.05002361937658861`, `PB` `0.05357326504599769`, `PHB`
`0.10811684231157415`. The Step-1 comparison still diverges. First divergent
schema field is `T`; largest residual is now `P` max_abs
`1561.2503728885986`, RMSE `305.9413510899027`, with `PH/MU/W` also material.
Therefore the next sprint is Step-1 operator/source localization, not more
base-init work and not TOST/Switzerland/FP32/memory.

Update 2026-06-09 14:00 WEST: step-1 T/P operator-localization sprint opened:
`.agent/sprints/2026-06-09-v014-step1-t-p-operator-localization/sprint-contract.md`.
This is the active debug sprint. It must use a focused Step-1
substage-truth/comparator path to localize the first `T` and dominant
`P/PH/MU` residuals, or deliver a narrow performance-compatible fix with
before/after proof. No TOST, Switzerland, FP32, memory source work, GPU, or
Hermes.

Update 2026-06-09 14:22 WEST: step-1 T/P operator-localization sprint closed.
Verdict:
`STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.
The worker compiled and ran scratch-only env-gated WRF instrumentation and
produced 168 substage truth files under
`/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`. The first
strict and first material T/P-family mismatch is `T_STATE` at
`after_rk_addtend_before_small_step_prep`, RK1. The largest residual at that
boundary is tendency-family (`PH_TEND` max_abs `794096.1875`; `RW_TEND`,
`PH_TENDF`, `T_TEND`, `T_TENDF` also large). RK1 `small_step_prep` work arrays
then match for `T_WORK` and `P_WORK` max_abs `0.0`. Therefore the next sprint is
not acoustic: split WRF `first_rk_step_part1/part2` and JAX
`_physics_step_forcing` / dry `*_tendf` construction.

Update 2026-06-09 14:27 WEST: step-1 RK1 source-boundary sprint opened:
`.agent/sprints/2026-06-09-v014-step1-rk1-source-boundary/sprint-contract.md`.
This is the active debug sprint. It must split WRF `first_rk_step_part1/part2`,
`rk_tendency`, `rk_addtend_dry/spec_bdy_dry` against JAX
`_physics_step_forcing`, `_augment_large_step_tendencies`, and dry `*_tendf`
construction. No acoustic continuation unless this earlier boundary is closed.

Update 2026-06-09 14:50 WEST: step-1 RK1 source-boundary sprint closed.
Verdict:
`STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.
The first material source-boundary mismatch is now `T_STATE` immediately after
WRF `first_rk_step_part1`. WRF vs JAX operational carry has max_abs
`5.490173101425171`, RMSE `1.9175184863907806`; WRF vs
`_physics_step_forcing.state` has max_abs `5.490142455570492`, RMSE
`1.9174736017582765`. RK1 `small_step_prep` continuity remains exact for
`T_WORK` and `P_WORK` max_abs `0.0`. The next sprint must split internal WRF
`first_rk_step_part1` surfaces against the JAX physics adapter output. Do not
continue acoustic, TOST, Switzerland, FP32, or memory work until this earlier
T-state mutation is explained or fixed.

Update 2026-06-09 14:50 WEST: step-1 part1 physics-state mutation sprint
opened:
`.agent/sprints/2026-06-09-v014-step1-part1-physics-state-mutation/sprint-contract.md`.
This is the active debug sprint. It must instrument/compare internal WRF
`first_rk_step_part1` surfaces against JAX `_physics_step_forcing` and
scheme-adapter state/tendency outputs. The target is an exact T-state mutation
source, an input-already-diverged proof, or one exact missing-truth blocker.

Update 2026-06-09 15:20 WEST: step-1 part1 physics-state mutation sprint
closed. Verdict:
`STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`. The full `T_STATE` residual is
already present at WRF `part1_entry_before_init_zero_tendency`: max_abs
`5.490173101425171`, RMSE `1.9175184863907806` against JAX live-nest
step-entry state. WRF `first_rk_step_part1` does not materially change
`T_STATE`; largest internal delta from entry is max_abs `0.0`. The next sprint
must move upstream to the live-nest/WRF call-site handoff immediately before
`first_rk_step_part1`, not into radiation/surface/PBL/cumulus or acoustic.

Update 2026-06-09 15:22 WEST: step-1 pre-part1 handoff sprint opened:
`.agent/sprints/2026-06-09-v014-step1-pre-part1-handoff/sprint-contract.md`.
This is the active debug sprint. It must compare WRF solve_em call-site state
immediately before `first_rk_step_part1` with JAX live-nest loader/carry/state
surfaces, and explicitly validate full-theta vs perturbation-theta mapping.

Update 2026-06-09 15:40 WEST: step-1 pre-part1 handoff sprint closed. Verdict:
`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`. WRF `T_STATE` is unchanged from
`after_step_increment` to `before_first_rk_step_part1_call` (max_abs `0.0`) and
is continuous with the prior part1-entry hook (max_abs `0.0`). Full-vs-
perturbation theta mapping was explicitly checked: WRF `grid%t_2`/`T_STATE`
maps to JAX `State.theta - 300 K`. The full residual remains in raw JAX
live-nest Step-1 state/carry before `_physics_step_forcing`: max_abs
`5.490173101425171`, RMSE `1.9175184863907806`. The next sprint is JAX
live-nest loader/carry construction for `T_STATE`.

Update 2026-06-09 15:44 WEST: step-1 JAX loader `T_STATE` sprint opened:
`.agent/sprints/2026-06-09-v014-step1-jax-loader-tstate/sprint-contract.md`.
This is the active debug sprint. It must split the JAX stages
`raw_child_state -> live_child_state -> boundary_package -> initial_carry ->
haloed_step_entry` against the accepted WRF solve_em pre-call truth. The main
current suspicion is live-nest base initialization updating `PB/PHB/MUB` while
leaving `State.theta` from raw `wrfinput_d02`, but this is only a hypothesis
until the proof names the exact first material stage. No TOST, Switzerland,
FP32, memory source work, GPU, Hermes, or broad source edit is authorized.

Update 2026-06-09 15:58 WEST: step-1 JAX loader `T_STATE` sprint closed.
Verdict:
`STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`. `T_STATE`
max_abs versus WRF pre-call stays `5.490173101425171` through raw, live,
boundary-package, carry, and haloed-entry stages, with all `T_STATE`
stage-transition max_abs values `0.0`. At the same time, live-nest base init
improves `PB` from raw max_abs `2627.3828125` to live max_abs
`0.05357326504599769` (`PHB/MUB` also close). The residual is not boundary-only:
haloed-entry interior max_abs is `5.490173101425171`, boundary-band max_abs
`5.284271240234375`. Therefore boundary package, carry, halo, and physics are
ruled out for this residual. The next sprint is WRF live-nest
`T_STATE`/theta semantics: prove and port the `med_nest_initial` /
`start_domain_em` `t_2` initialization path after terrain/base blending.

Update 2026-06-09 16:03 WEST: step-1 live-nest theta semantics sprint opened:
`.agent/sprints/2026-06-09-v014-step1-live-nest-theta-semantics/sprint-contract.md`.
The proof/fix target is now exact: WRF `med_nest_initial` blends
`ht/mub/phb`, then calls `adjust_tempqv(nest%mub, nest%mub_save, nest%c3h,
nest%c4h, nest%znw, nest%p_top, nest%t_2, nest%p, QVAPOR, use_theta_m, ...)`.
The worker must transcribe `adjust_tempqv` proof-locally, compare the candidate
`T_STATE`/`QVAPOR` against accepted WRF pre-call truth, and only then apply an
initialization-only source patch if the candidate closes the residual. No GPU,
TOST, Switzerland, FP32, memory source work, Hermes, or validation campaign.

Update 2026-06-09 16:08 WEST: final v0.14 validation requirements expanded per
principal direction. TOST remains required, but it must be paired with a
Grid-Delta Atlas over all paired CPU/GPU wrfout cases, lead times, cells, and
common numeric fields. Required release artifacts and README plot/dashboard
requirements are recorded in
`.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. This does not unblock TOST
now; it defines the validation output once the current grid-parity bug is fixed.

Update 2026-06-09 16:18 WEST: step-1 live-nest theta semantics closed as
`STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.
The dominant missing semantics are WRF `USE_THETA_M=1` dry-to-moist theta
conversion plus `adjust_tempqv`: this reduces `T_STATE` max_abs from
`5.490173101425171` to `0.00541785382188209`, but it does not meet the prior
`1e-3 K` material gate. No production patch is allowed yet. The companion
QVAPOR schema sprint closed as
`STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`: accepted pre-call
truth lacks same-boundary `QVAPOR`; existing QVAPOR truth is post-RK/pre-halo.
Next required sprint is a CPU-only WRF savepoint extension at
`before_first_rk_step_part1_call` to emit `moist(i,k,j,P_QV)` as `QVAPOR`, then
rerun the theta proof.

Update 2026-06-09 18:17 WEST: step-1 live-nest theta/QV production wiring is
closed as `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`. The
production `build_replay_case(..., live_nest_parent=...)` path now applies WRF
`USE_THETA_M=1` theta conversion plus `adjust_tempqv` using the transient
post-`blend_terrain`/pre-`start_domain` `MUB` surface. The final
post-`start_domain` BaseState is unchanged. Corrected theta max_abs vs
same-boundary WRF pre-call truth is `5.788684885033035e-05 K`; corrected
QVAPOR max_abs is `5.970267497393267e-08`. The Step-1 16-field comparison
still diverges: first divergent schema field `T`, largest residual field `P`
max_abs `974.9820434775493`, RMSE `135.98147360593399`, worst Fortran
`i=1,j=30,k=1`, boundary band true. Next active work is Step-1 `P/PH/MU`
boundary/operator localization. Do not resume TOST, Switzerland, FP32 source
work, or memory follow-ups from this artifact.

Update 2026-06-09 18:22 WEST: opened the next sprint:
`.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization`. The
method is a focused Step-1 boundary/substage comparator, reusing the current
post-theta/QV-closure proof as baseline. It may apply a production source fix
only if the exact bug is proven and the fix remains narrow and GPU-performance
compatible. The immediate target is the boundary-band `P` residual while
accounting for first divergent `T` and material `PH/MU/W/U`.

Update 2026-06-09 18:26 WEST: opened read-only Management Review 02 because 18
v0.14 sprints have closed since Management Review 01. Review sprint:
`.agent/sprints/2026-06-09-v014-management-review-02`. This does not replace
the active P/PH/MU debug sprint; it is a parallel drift-control check of
roadmap, method, sprint sizing, validation gates, and whether the current
boundary-localization plan is still the fastest rigorous path.

Update 2026-06-09 18:31 WEST: Opus could not run the P/PH/MU sprint or
Management Review 02 because the Claude session limit is exhausted until about
21:20 WEST. No P/PH/MU artifact was produced by Opus. The active P/PH/MU
debug sprint is now running as a GPT-5.5 xhigh tmux worker in window `0:4`.
Management Review 02 remains opened but pending retry after Opus availability
returns; it is not on the critical path for the current debug proof.

Update 2026-06-09 18:50 WEST: P/PH/MU boundary-localization closed as
`STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`. The current first
material P-family state residual is WRF `after_first_rk_step_part1` versus JAX
`_physics_step_forcing.carry.state`, field `P_STATE`, max_abs `69.96875`.
`MU_STATE` and `W_STATE` are material at that same checked boundary. RK1
`small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for checked
`T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK`. The final strict Step-1 residual still
has `P` max_abs `974.9820434775493`, so this is localization, not closure. Next
debug sprint should emit one internal WRF `first_rk_step_part1` surface around
`phy_prep`/`calc_p_rho_phi` state writes for `P/MU/W`, or split
post-acoustic/pre-refresh pressure before source edits.

Update 2026-06-09 18:57 WEST: opened the next sprint:
`.agent/sprints/2026-06-09-v014-step1-first-rk-part1-p-state-split`. This is a
debug-boundary sprint, not a validation campaign and not a source-fix sprint by
default. It must emit the smallest internal WRF `first_rk_step_part1` split
around `phy_prep` / `calc_p_rho_phi` for `P/MU/W`, compare it to current JAX
`_physics_step_forcing` state/carry surfaces, and return an exact boundary,
narrow performance-compatible fix, or exact missing-truth blocker. TOST,
Switzerland, FP32 source work, and memory source work remain paused.

Update 2026-06-09 19:08 WEST: first-RK part1 P-state split closed as
`STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`. WRF
`before_first_rk_step_part1_call -> after_first_rk_step_part1` is exact for
`P_STATE/MU_STATE/W_STATE/PH_STATE`; WRF part1 entry to `after_phy_prep` is
exact for `P_STATE/MU_STATE`. JAX `raw_child_state` already differs from WRF
pre-call for `P/MU/W` and the same residuals persist through `live_child_state`,
boundary package, initial carry, haloed step entry, and
`_physics_step_forcing.carry.state`: `P_STATE=69.96875`,
`MU_STATE=13.256103515625`, `W_STATE=0.7605466246604919`. The exact next target
is live-nest raw-child to live/pre-part1 perturbation-state initialization for
`P_STATE/MU_STATE/W_STATE`, not WRF part1, carry, halo, or acoustic refresh.

Update 2026-06-09 19:15 WEST: opened sprint
`.agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init`. The worker
starts from the hypothesis that live-nest base/theta/QV correction still leaves
`P/MU/W` perturbation leaves from raw `wrfinput_d02`, but the contract now asks
the worker to actively disprove that hypothesis, rank alternate causes, and try
cheap proof-local falsifiers if the hypothesis fails. Production edits are
limited to a narrow GPU-native live-nest/init fix only after formula proof.

Update 2026-06-09 19:36 WEST: live-nest perturb-state init sprint closed and
pushed as `f73542c0`. Verdict:
`STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`.
WRF recomputes/adjusts `P/MU/W` before `first_rk_step_part1_call`, while JAX
keeps raw `wrfinput_d02` perturbation leaves through raw child, live child,
boundary package, initial carry, halo entry, and `_physics_step_forcing`.
Proof-local WRF transcriptions reduce residuals:
`P_STATE 69.96875 -> 3.9458582235092763 Pa`, `MU_STATE
13.256103515625 -> 0.047773029698646496 Pa`, and `W_STATE
0.7605466246604919 -> 1.2992081932505783e-07 m/s`. No production source edit
was applied because `P_STATE` still needs one internal WRF `start_domain`
`al/alt` plus pre/post-`press_adj` truth surface before patching.

Update 2026-06-09 19:42 WEST: non-colliding Memory/FP32 side-manager branch
merged and pushed as `ee6cbbe1`. Source edit is limited to
`src/gpuwrf/coupling/scan_adapters.py`: WDM6 `slmsk` now passes a per-column
vector instead of materializing a full-column broadcast. Proof verdict:
`WDM6_SLMSK_SHAPE_CLEANUP_EXACT`; WDM6 savepoint parity `85 passed`, WDM6
operational smoke `1 passed`, target saving `76.92176055908203 MiB`. FP32
source work remains blocked by the live-nest/dycore grid-parity lock.

Next active debug step: open a CPU-only disposable-WRF start-domain subsurface
sprint. It must emit WRF live-nest `start_domain(nest,.TRUE.)` internal
surfaces after the hypsometric `P/al/alt` recompute and immediately before/after
`press_adj`, including `P_STATE`, `MU_STATE`, `al`, `alt`, `alb`, `PH_STATE`,
`PB`, `MUB`, `PHB`, `theta`, `qv`, `HT`, and `HT_FINE`. Only after that surface
closes `P_STATE` should a narrow GPU-native `d02_replay` source patch be
attempted. TOST, Switzerland, and FP32 source work remain paused.

## Manager Directive

Release labels are secondary. The current priority order is:

1. Find why GPU cells diverge from CPU-WRF, across all written fields, and fix it.
2. FP32 acoustic / mixed precision.
3. Remaining memory problems.
4. Powered TOST, only after the cell fields are no longer radically divergent.

The operating motto for this phase is "no slob": do not hide behind station scores
if the actual grid fields are not WRF-close.

Principal communication directive 2026-06-08 late WEST: do not send
Hermes/Telegram process-progress updates. Keep manager/agent handoffs and
top-level validation output context-sparing: concise verdicts and proof paths in
text, full field tables in JSON/CSV artifacts.

Tooling directive 2026-06-09 12:45 WEST, expanded 12:55 WEST: for long
runtime/kernel-debug ladders, ask at every planning step whether the current
tools/methods are right and whether the plan is the fastest rigorous wall-clock
path. A focused harness, savepoint emitter, comparator, schema freezer, or
visualization sprint is cheap if it turns repeated slow reproductions into a
fast falsifiable proof loop. It can also be faster to send one worker in
parallel or serially to prove/refute a hypothesis than to keep narrowing the bug
through slow runs. Prefer expert runtime-debug methods that minimize steps and
false assumptions.

## Current Evidence

- Powered TOST Case 1 and Case 2 were durable before the memory-fix pause.
- Case 3 completed on 2026-06-08 and the watcher stopped TOST before Case 4.
- `proofs/v014/v10_grid_diagnostics.json` currently reports:
  - V10 grid RMSE above 1.5 m/s in 3/3 cases.
  - V10 grid bias signs are `-, -, +`; this is not a simple constant-bias issue.
  - Station V10 is outside the tight ADR-029 margin in 1/3 cases.
  - Case 3 has retained wrfouts and shows V10 RMSE 2.524 m/s, U10 RMSE
    2.068 m/s, PSFC RMSE 525 Pa, and T2 RMSE 0.994 K.
  - Case 3 V10 error is worst around h10-h14, strongest in NW/SW quadrants and
    ocean/low-terrain bins, with weak correlation to T2 and modest negative
    correlation to PSFC.
- Existing docs already classify this as KI-9 lead-time wind/mass divergence, but
  the exact operator root cause is not closed.

## TOST Status

TOST is intentionally paused. The runner was stopped cleanly after Case 3:

- Log: `/mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.log`
- Stop watcher log: `/mnt/data/wrf_gpu_validation/v0130_marathon/stop_after_case3_watch.log`
- Case JSONs: `proofs/v0120/powered_tost_n15/case_*.json`
- Case 3 proof dir:
  `proofs/v0120/powered_tost_n15/pipeline_proofs/20260501_18z_l2_72h_20260519T173026Z/`

Do not resume TOST until a manager explicitly records why the grid-field envelope
is acceptable or what root-caused residual remains.

## Active Sidecar Agents

- `019ea948-6d45-78d3-b06a-bc0ad1df40ff` (`Peirce`):
  prior V10/wind-divergence attribution synthesis. Completed:
  `.agent/reviews/2026-06-08-gpt-v014-v10-prior-attribution.md`.
- `019ea948-81c9-7161-b50c-04eaff1eb010` (`Raman`):
  v0.14 cell-level validation envelope design. Completed:
  `.agent/reviews/2026-06-08-gpt-v014-cell-envelope-gate.md`.
- `019ea948-ec75-76e0-b708-44aabd02af0b` (`Heisenberg`):
  FP32 acoustic status freeze. Completed:
  `.agent/reviews/2026-06-08-gpt-v014-fp32-status-freeze.md`.

FP32 freeze verdict: feasible in principle, v0.14 P1, but source work waits
until the grid-cell divergence root cause is clearer. Naive/global fp32 remains
rejected; only mixed perturbation-authoritative acoustic is a candidate.

Cell-envelope design verdict: start with the 10 frozen core fields from
`docs/equivalence-demo.md` as hard-fail fields (`T2`, `U10`, `V10`, `PSFC`,
`RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`), while inventorying every current-common
writer field. Other fields stay report-only until per-field tolerances are
frozen before seeing promotion results.

Prior-attribution verdict: do not re-debug the old fixed boundary-normal or
missing-Coriolis causes unless a current regression probe proves them. The next
useful work is current-code spatial/vertical anatomy, then first-divergence /
component-tendency localization.

Memory verdict: RRTMG column tiling is fixed and was the only true memory
blocker. Exact-branch memory preflight exists, and the empirical/static memory
map is now complete (`proofs/v014/empirical_memory_map.*`). No remaining
non-radiation memory fix blocks the first long validation after grid parity if
the selected exact-branch preflight fits. Do not rewrite MYNN/PBL/post-physics
merge/moisture limiter/acoustic memory paths blindly. The 2026-06-09 direct
grid-after-base proof confirms memory is not the next gate; dynamic grid
divergence is.

Wave-1 grid attribution verdict: the first fix target is static grid,
vertical-coordinate, and base-state parity, not a dynamic operator edit. Case 3
emitted wrfouts have 31 non-exact static/grid fields; largest mismatches are
`C2H/C2F` max 95,000 Pa, `C4H/C4F` about 26.7 kPa, `RDN` max 161.7, and `HGT`
max 228 m. Dynamic divergence remains broad (`PSFC`, `P`, `PH`, `MU`, `U`, `V`,
`U10`, `V10`), but no dycore/radiation/FP32 fix should start until the static
metric/base payload is exact or root-caused as writer-only.

Manager preliminary 2026-06-08 23:55 WEST: the vertical static mismatch has a
specific writer-vs-runtime hypothesis. `build_replay_case()` constructs
`grid = run.grid(domain).as_grid_spec()` before loading real WRF metrics.
`GridSpec.__post_init__` fills `grid.metrics` with `DycoreMetrics.flat` when no
metrics are provided. The pipelines pass `case.metrics` into
`OperationalNamelist`, so runtime dynamics may use the real loaded metrics while
`wrfout_writer._add_grid_static_fields()` reads `grid.metrics` and emits the
flat fallback. Local read-only compare supports this pattern: GPU `C1H/C3H`
match `ZNU`, `C1F/C3F` match `ZNW`, `C2*/C4*` are zero, and `DN[0]/RDN[0]`
match the flat fallback. Huygens must prove/refute this before any source fix.
`HGT/XLAT/MAPFAC/F/E/SINALPHA` should be treated separately because GPU HGT
matches the retained `wrfinput_d02` while CPU wrfout HGT differs from that input.

Sprint contract update: Huygens may patch `src/gpuwrf/integration/d02_replay.py`
if stale `GridSpec.metrics` plumbing is proven, or `src/gpuwrf/io/wrfout_writer.py`
if writer payload selection is proven. `contracts/grid.py` and runtime dycore
remain read-only without a follow-up contract.

Manager closeout 2026-06-09 00:05 WEST: Huygens completed the static
metric/base-state sprint and the narrow source patch is accepted. The fix loads
WRF `DycoreMetrics` immediately after `GridSpec` creation in
`src/gpuwrf/integration/d02_replay.py` and attaches them with
`dataclasses.replace(grid, metrics=metrics)`. Runtime `case.metrics` /
`namelist.metrics` behavior is preserved; no dycore, radiation, runtime, or
writer source was changed. Proofs:
`proofs/v014/static_metric_base_parity.*` and
`.agent/reviews/2026-06-08-v014-static-metric-base-parity.md`.

Static proof verdict: retained GPU h1 vertical C/DN/RDN and MAPFAC mismatches
were stale writer payload from flat `GridSpec.metrics`, not runtime dynamics.
Current patched synthetic writer payload matches wrfinput for those fields.
`XLAT/XLONG` remain a separate writer-fallback issue because runtime State lacks
lat/lon arrays. `HGT` and much of `PHB` are CPU wrfinput-vs-CPU-wrfout
conventions. `PB/MUB` need a fresh GPU h0/h1 writer artifact or same-state
probe to split forecast-step drift from h1 writer reconstruction. Because the
grid-envelope script uses retained old GPU wrfouts, its static mismatch count
cannot improve until a fresh writer artifact exists.

Manager closeout 2026-06-09 00:20 WEST: committed `a42865e8`
(`v014 fix static metric writer payload`) and `4374ca77`
(`v014 add wrfout grid comparator`). The new comparator
`scripts/compare_wrfout_grid.py` is now the primary context-sparing
CPU-WRF-vs-GPU wrfout grid validation tool. Smoke proof:
`proofs/v014/grid_comparison_framework_smoke.*`; method:
`proofs/v014/grid_comparison_method.md`; review:
`.agent/reviews/2026-06-08-v014-grid-comparison-framework.md`. Case 3 retained
smoke: 24 paired h1-h24 d02 files, 100 common variables, 99 numeric fields,
37 dynamic, 61 static/time-invariant, 2 metadata, report-only verdict because
no frozen tolerance manifest was supplied. Top retained static failures
(`C2F/C2H/C4F/C4H`) are expected to disappear only after a fresh post-fix
writer artifact; retained old wrfouts still carry the stale writer payload.

## Active Wave 1

- `019ea94e-898f-7211-9561-e70af150fcfd` (`Averroes`):
  all-comparable-field grid-cell envelope harness and report. Completed:
  `proofs/v014/grid_cell_envelope.*` and
  `.agent/reviews/2026-06-08-v014-grid-parity-attribution.md`.
- `019ea950-77c3-7750-9adb-7e1c1e05bc1d` (`Godel`):
  CPU-only wind/mass vertical-spatial anatomy probe. Completed:
  `proofs/v014/wind_mass_divergence_probe.*` and
  `.agent/reviews/2026-06-08-v014-wind-mass-divergence-probe.md`.
- `019ea950-93c0-7a60-8598-8da51ae2d2fb` (`Planck`):
  v0.14 memory research integration and memory-fix roadmap. Completed:
  `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md` and
  `.agent/reviews/2026-06-08-v014-memory-research-integration.md`.
- `019ea950-e500-7cd0-8292-15576f327532` (`Descartes`):
  Switzerland validation prep, no GPU run. Completed:
  `proofs/v014/switzerland_validation_plan.md` and
  `.agent/reviews/2026-06-08-v014-switzerland-validation-prep.md`.
- `019ea957-da82-7891-9a9b-3ad594d8b671` (`Nietzsche`):
  exact-branch memory preflight. Completed:
  `proofs/v014/exact_branch_memory_preflight.*` and
  `.agent/reviews/2026-06-08-v014-exact-branch-memory-preflight.md`. Static
  memory controls are present; short GPU nested smoke timed out at 600 s with no
  OOM and peak total VRAM about 3204 MiB, so this is not a full long-validation
  memory-fit pass.

Wave deliverables are expected under `proofs/v014/` and
`.agent/reviews/2026-06-08-v014-*.md`.

## Active Wave 2

- `019ea95e-f825-7a92-a5d2-bfc1e1082aee` (`Huygens`):
  primary static metric/base-state parity sprint. Completed and accepted.
  Deliverables: `proofs/v014/static_metric_base_parity.*` and
  `.agent/reviews/2026-06-08-v014-static-metric-base-parity.md`. The source
  patch is limited to `src/gpuwrf/integration/d02_replay.py` and fixes stale
  `GridSpec.metrics` for future writer output.
- `019ea95f-15e9-70b2-b6bf-cc4c1de48047` (`Curie`):
  read-only same-state tendency localization design. Completed:
  `proofs/v014/same_state_tendency_localization_plan.md`,
  `proofs/v014/same_state_tendency_inventory.json`, and
  `.agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md`.
  This becomes the next dynamic-debug sprint only after static/base parity is
  green or formally writer-only.
- `019ea968-c876-71c3-886a-133a3e740ab2` (`Hypatia`):
  v0.14 grid comparison framework sprint
  `.agent/sprints/2026-06-08-v014-grid-comparison-framework/sprint-contract.md`.
  Completed and accepted. Commit `4374ca77`.

## Active Wave 3

- Manager-owned GPU smoke:
  `.agent/sprints/2026-06-09-v014-post-static-writer-smoke/sprint-contract.md`.
  Purpose is a short h1 live-nested d01->d02 run through
  `proofs/v0120/powered_tost_n15/run_one_case_v0120.py` on the current branch,
  then `scripts/compare_wrfout_grid.py --min-lead 1 --max-lead 1`, to prove
  the static metric writer payload is fixed on disk and clear stale retained
  artifacts from the comparator's top fields. First attempted command using
  `scripts/m7_l2_d02_replay.py` was blocked immediately because that old
  single-domain d02 path demands missing `wrfbdy_d02`; this is a known wrong
  path for L2 nested cases, not a GPU/OOM issue. Corrected command uses
  `/tmp/v0120_merged_run_root` and the live-nested TOST per-case runner.
  Completed 2026-06-09 00:36 WEST: h1 live-nested run is `L2_D02_GREEN`;
  `bounds/RMSE/wall_clock` all PASS; d02 wrfout written under
  `/tmp/v014_post_static_writer_smoke/l2_d02_20260501_18z_l2_72h_20260519T173026Z`.
  h1 comparator proof: `proofs/v014/post_static_writer_grid_compare.*`.
  Former static metric writer failures are exact on disk:
  `C1/C2/C3/C4`, `DN/DNW/RDN/RDNW`, and all `MAPFAC_*` have
  `rmse=max_abs=bias=0`. Remaining top h1 static/base fields are
  `PHB/MUB/PB/HGT/XLAT/XLONG`; dynamic top fields include
  `PSFC/MU/P/HFX/PBLH/PH` and radiation fluxes. Closeout:
  `.agent/reviews/2026-06-09-v014-post-static-writer-smoke.md`.
- `019ea977-234a-7b52-b87b-b6fb709e2d2d` (`Helmholtz`):
  CPU-only dynamic field attribution sprint
  `.agent/sprints/2026-06-09-v014-dynamic-field-attribution/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/dynamic_field_attribution.*` and
  `.agent/reviews/2026-06-09-v014-dynamic-field-attribution.md`. No `src/`
  edits, no GPU. Manager reran the CPU-only script and revalidated JSON/compile.
  Verdict: first materially bad lead is h1 (`W`, `PSFC` threshold hits), but
  selected same-state localization lead is h10
  (`2026-05-02T04:00:00+00:00`) because it is the strongest h10-h14 primary
  debug window. The manifest selects 24 mass-grid cells with U/V/W/PH native
  stagger context and first-probe levels
  `0,1,2,16,17,18,24,25,26,28,29,30,31,32`. Top dynamic suspects are
  `PSFC`, `V`, `P`, `T`, `U`, `V10`, `MU`, and `PH`. Next target is CPU-WRF
  term savepoints for these h10 cells before any JAX-only operator conclusion.
- `019ea97b-119c-7561-ae35-948a3fc1405a` (`Sartre`):
  CPU-only same-state WRF savepoint feasibility sprint
  `.agent/sprints/2026-06-09-v014-same-state-wrf-savepoint-feasibility/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/same_state_wrf_savepoint_feasibility.*` and
  `.agent/reviews/2026-06-09-v014-same-state-wrf-savepoint-feasibility.md`.
  Manager corrected one stale manifest-path string from
  `dynamic_field_attribution_summary.json` to the real
  `proofs/v014/dynamic_field_attribution.json` and revalidated JSON. Verdict:
  fastest reliable source-truth path is a disposable instrumented copy of
  `/home/enric/src/wrf_pristine/WRF`, which has built CPU `main/wrf.exe` and
  `main/real.exe` but is dirty/apparently serial. Do not patch it in place. The
  historical Case 3 path
  `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF` exists but has no
  active build/executable. No Case 3 restart shortcut was found, so h10 source
  truth requires an instrumented forward run from `2026-05-01_18:00:00` unless
  the implementation sprint first creates a restart. Old
  `external/wrf_savepoint_patch` is not usable as-is.
- `019ea980-b391-7923-a63f-81cd9f6dae48` (`Ampere`):
  CPU-only base-state writer attribution sprint
  `.agent/sprints/2026-06-09-v014-base-state-writer-attribution/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/base_state_writer_attribution.*` and
  `.agent/reviews/2026-06-09-v014-base-state-writer-attribution.md`. No `src/`
  edits, no GPU. Verdict: no runtime input mismatch; CPU and GPU native
  `wrfinput_d02` are exact for `PHB/MUB/PB/HGT/XLAT/XLONG`. Classifications:
  `PHB` and `HGT` are `cpu_output_convention`, `XLAT/XLONG` are
  `writer_fallback`, and `PB/MUB` are `forecast_step_change` /
  dynamic state-split symptoms. Same-state dynamic localization can proceed
  with these exclusions recorded; an exact `XLAT/XLONG` wrfout fix is
  writer-only and not a dycore blocker.
- `019ea988-9f3c-7571-9073-f2b6d41b09f6` (`Kierkegaard`):
  CPU-only same-state savepoint request manifest sprint
  `.agent/sprints/2026-06-09-v014-same-state-savepoint-request/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/same_state_savepoint_request.*` and
  `.agent/reviews/2026-06-09-v014-same-state-savepoint-request.md`. No `src/`
  edits, no WRF edits, no GPU. Manager reran the CPU-only generator and
  revalidated JSON/compile. Verdict: manifest packages h10 `d02`
  (`2026-05-02T04:00:00+00:00`) with exactly 24 selected mass-grid cells,
  native U/V/W/PH stagger context, halo-8 patch bounds, full native vertical
  column requirement, RK stages `1,2,3`, first/last acoustic substep samples,
  and 15 WRF source term groups:
  `stage_input`, `mass_coupling`, `momentum_advection`,
  `scalar_theta_mu_advection`, `diffusion`, `horizontal_pgf`, `coriolis`,
  `source_tendency_folding`, `small_step_prep`, `acoustic_uv`, `mu_theta`,
  `w_ph`, `pressure_rho_refresh`, `boundary_spec_relax`,
  `final_stage_state`.

## Active Wave 4

- `019ea992-7045-7db3-bd21-f09583024532` (`Herschel`):
  WRF same-state marker savepoint sprint
  `.agent/sprints/2026-06-09-v014-wrf-same-state-marker-savepoint/sprint-contract.md`.
  Write scope in repo: `proofs/v014/wrf_same_state_marker_savepoint.*`,
  `proofs/v014/wrf_same_state_marker_patch.diff`, and
  `.agent/reviews/2026-06-09-v014-wrf-same-state-marker-savepoint.md`.
  External scratch scope:
  `/mnt/data/wrf_gpu2/v014_same_state_wrf/**` or fallback
  `/tmp/wrf_gpu2_v014_same_state_wrf/**`. No repo `src` edits, no GPU, no
  Hermes. Objective: copy `/home/enric/src/wrf_pristine/WRF` to a disposable
  tree, add env-gated WRF marker hooks, prove h10/d02 step and selected native
  indices against CPU h10 wrfout, then emit the first routine-boundary
  source-term layer only if the marker is green.

Manager closeout 2026-06-09 02:58 WEST: Herschel completed and the sprint is
accepted. Deliverables:
`proofs/v014/wrf_same_state_marker_savepoint.{json,md}`,
`proofs/v014/wrf_same_state_marker_patch.diff`, and
`.agent/reviews/2026-06-09-v014-wrf-same-state-marker-savepoint.md`.
Verdict: `MARKER_GREEN`. The final CPU-only dmpar run used 28 MPI ranks and no
GPU/Hermes. It proves `d02` h10 maps to `grid%itimestep=6000`,
`current_timestr_before_step=2026-05-02_03:59:54`, `lead_seconds_after_step=36000`,
and the selected mass/U/V/W/PH patch indices match the requested native WRF
coordinates. The final post marker samples WRF history `T` from
`grid%th_phy_m_t0`; earlier `grid%t_2`/`grid%t_1` attempts were useful but
misleading for wrfout history `T`.

Final comparison against the scratch h10 wrfout is roundoff-level
(`T/P/PB=0`, `U/V<=8.88e-16`, `W=4.44e-16`, `PH=5.33e-15`). Against the
provided CPU h10 wrfout, `T/P/PB=0`, `U=4.77e-7`, `V=9.54e-7`,
`W=1.19e-7`, and `PH=1.91e-6` max_abs. Repo `src/` stayed untouched; marker
hooks live only in the disposable WRF scratch tree and are preserved as a patch
diff proof. No WRF process remains running. Next sprint: dynamic localization
from this green post-marker point, with routine-boundary term emitters around
the same location before any GPU or FP32 work.

## Completed Wave 5

- `019ea9ad-aa61-7ab0-b3af-6b4734d886c0` (`Sagan`):
  Lat/Lon writer-only payload sprint
  `.agent/sprints/2026-06-09-v014-latlon-writer-payload/sprint-contract.md`.
  Write scope in repo: `src/gpuwrf/io/wrfout_writer.py`,
  `src/gpuwrf/integration/daily_pipeline.py`,
  `src/gpuwrf/integration/nested_pipeline.py`,
  `proofs/v014/latlon_writer_payload.{py,json,md}`, and
  `.agent/reviews/2026-06-09-v014-latlon-writer-payload.md`. No GPU, no
  Hermes, no WRF-source or same-state scratch edits. Objective: replace the
  writer's synthetic projection fallback for `XLAT`/`XLONG` with real WRF
  host-only lat/lon payloads when available, without adding JIT-visible
  `State`, `GridSpec`, or `OperationalNamelist` leaves. This is not a dycore
  correctness fix; it removes a known writer-only comparator distraction before
  the next fresh wrfout comparison.

Manager closeout 2026-06-09 01:13 WEST: Sagan completed and the sprint is
accepted. Local commit `2b7d022c` routes host-only static lat/lon payloads
through the daily and nested writers and keeps the synthetic fallback for callers
without payloads. Manager validation reran:

- `python -m json.tool proofs/v014/latlon_writer_payload.json`
- `python -m py_compile src/gpuwrf/io/wrfout_writer.py src/gpuwrf/integration/daily_pipeline.py src/gpuwrf/integration/nested_pipeline.py proofs/v014/latlon_writer_payload.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/latlon_writer_payload.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_netcdf_writer.py tests/test_m7_daily_pipeline.py tests/test_auxhist_stream.py tests/test_auxhist_multistream.py`

Verdict: `XLAT`, `XLONG`, `XLAT_U`, `XLONG_U`, `XLAT_V`, and `XLONG_V`
are exact against the GPU-native `wrfinput_d02` payload
(`rmse=max_abs=0.0`); the listed pytest set passed `26 passed, 1 skipped`.
This is a writer-only cleanup and does not change the dynamic divergence root
cause queue.

## Completed Wave 6

- `019eaa1d-13b8-7e02-a1b3-ebabb5e2d44a` (`Ptolemy`):
  CPU-only dynamic same-state term localization sprint
  `.agent/sprints/2026-06-09-v014-dynamic-term-localization/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/wrf_dynamic_term_localization.{py,json,md}`,
  `proofs/v014/wrf_dynamic_term_localization_patch.diff`, and
  `.agent/reviews/2026-06-09-v014-dynamic-term-localization.md`.
  Manager validation reran JSON validation and Python compilation; no WRF/GPU
  process remained active.

Verdict: `TERM_LAYER_EMITTED_final_stage_small_step_finish`. The sprint emitted
compact CPU-WRF source-derived values immediately before and after final-stage
`small_step_finish` at `d02` step 6000 / h10. The accepted post-RK marker remains
green vs CPU h10 (`T/P/PB=0`; `U/V/W/PH <= 1.91e-6` max_abs), while retained
GPU/JAX h10 on the same marker patch still diverges (`T=3.36`, `P=590.0`,
`PB=1047.0`, `U=6.29`, `V=11.59`, `W=1.73`, `PH=5.10` max_abs). The
tile-local post-`small_step_finish` surface is not yet history-aligned for
`P/V/W`, so no root cause is claimed. The next exact WRF layer is the
pressure/rho/post-RK refresh path before or around `after_all_rk_steps`, then a
JAX CPU same-state wrapper against the green post-marker surface.

## Completed Wave 7

- `019eaa42-3b01-7b10-8a19-145943a0e9a2` (`Boole`):
  CPU-only pressure/rho post-RK refresh localization sprint
  `.agent/sprints/2026-06-09-v014-pressure-rho-post-rk-localization/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/wrf_post_rk_refresh_localization.{py,json,md}`,
  `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`, and
  `.agent/reviews/2026-06-09-v014-post-rk-refresh-localization.md`.
  Manager validation reran JSON validation and Python compilation; no WRF/GPU
  process remained active.

Verdict: `REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo`. The final
`calc_p_rho_phi` boundary closes the large `P` gap from Ptolemy's
post-`small_step_finish` layer, and `after_all_rk_steps` closes the remaining
`V/W` gap to the green marker. The accepted JAX compare target is now the state
immediately after `dyn_em/solve_em.F::after_all_rk_steps` and before RK halo
exchanges. Candidate vs provided CPU h10 is exact/roundoff:
`T/P/PB=0`, `U=4.77e-7`, `V=9.54e-7`, `W=1.19e-7`, `PH=1.91e-6`,
`MU=4.55e-13`, `MUB=0` max_abs. Retained GPU/JAX h10 remains far away on the
same patch (`T=3.36`, `P=590`, `PB=1047`, `U=6.29`, `V=11.59`, `W=1.73`,
`PH=5.10`, `MU=267`, `MUB=1050` max_abs), so the next sprint is a JAX CPU
same-state wrapper at this surface, not another WRF-only emitter.

## Completed Wave 8

- `019eaa5c-e784-7291-8294-2e83b3b597b9` (`Ramanujan`):
  CPU-only JAX after-all-RK same-state wrapper sprint
  `.agent/sprints/2026-06-09-v014-jax-after-all-rk-wrapper/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/jax_after_all_rk_wrapper.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-jax-after-all-rk-wrapper.md`.
  Manager validation reran JSON validation and Python compilation.

Verdict: `WRAPPER_BLOCKED_NO_JAX_PRE_HALO_STATE_API`. The WRF truth surface is
green and parsed, but the current JAX runtime exposes only post-halo/post-guard
state. In `runtime/operational_mode.py`, `_acoustic_scan` reaches the desired
state via `_carry_from_finished_stage(...)`, then immediately wraps it in
`apply_halo(...)` before returning; `_rk_scan_step` also applies halo before
returning each stage. A public forecast/API run would therefore compare the
wrong cadence surface. The retained GPU/JAX h10 wrfout mismatch remains a
diagnostic only, not same-surface CPU evidence. Next sprint: add a narrow,
default-off CPU-only pre-halo capture/debug hook around `_acoustic_scan` before
its `apply_halo` return, then rerun the JAX same-state comparison.

## Completed Wave 9

- `019eaa68-cbe4-7af3-bded-ac00ed10d98a` (`Gauss`):
  source-changing but non-corrective JAX pre-halo capture hook sprint
  `.agent/sprints/2026-06-09-v014-pre-halo-capture-hook/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `src/gpuwrf/runtime/operational_mode.py`,
  `proofs/v014/jax_pre_halo_capture.{py,json,md}`,
  `tests/test_v014_pre_halo_capture.py`, and
  `.agent/reviews/2026-06-09-v014-pre-halo-capture-hook.md`.

Verdict: `HOOK_GREEN_COMPARE_BLOCKED_NO_JAX_H10_PRESTEP_CARRY`. The hook is
private/proof-only and default-off. Normal `_rk_scan_step` still returns
`OperationalCarry`, public forecast signatures do not expose capture arguments,
and the focused tests passed. The capture path returns the same normal carry
plus the final RK3 pre-halo `State`; manager validation reran compile, proof
script, JSON validation, `tests/test_v014_pre_halo_capture.py`, and
`tests/test_m6_guard_disabled_debug.py` (`14 passed`). The h10 same-surface
JAX-vs-WRF comparison remains blocked because no CPU-loadable JAX
`OperationalCarry` exists immediately before `d02` step 6000/h10 with the real
state, promoted carry leaves, metrics/tendencies/boundary config, and boundary
leaves. Next sprint: build or locate that h10 pre-step carry checkpoint and run
the hook against Boole's WRF green target.

## Completed Wave 10

- `019eaa76-c97e-7730-9e2e-66397a3c5096` (`Euler`):
  CPU-only h10 pre-step carry checkpoint availability sprint
  `.agent/sprints/2026-06-09-v014-h10-prestep-carry-checkpoint/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/jax_h10_prestep_carry.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`.

Original verdict: `CHECKPOINT_BLOCKED_NO_H10_PRESTEP_CARRY`. Euler inspected existing
candidate checkpoints and found no CPU-loadable `OperationalCarry` at completed
step 5999, immediately before `d02` step 6000/h10. Existing APIs can serialize
full carries (`runtime.checkpoint.write_checkpoint(..., runtime_state=carry)`
and restart carry helpers), but current drivers do not write this required
artifact. No same-surface JAX-vs-WRF comparison ran. Next sprint: produce the
step-5999 full carry checkpoint with existing APIs if possible, then rerun
`proofs/v014/jax_h10_prestep_carry.py` with
`WRFGPU2_H10_PRESTEP_CARRY=/abs/path/to/d02_step5999_full_carry.pkl`.

## Completed Wave 11

- `019eaa81-0fee-7cf3-82b4-8879b3026c09` (`McClintock`):
  h10 pre-step carry producer sprint
  `.agent/sprints/2026-06-09-v014-h10-prestep-carry-producer/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/jax_h10_prestep_carry_producer.{py,json,md}`,
  `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`, and the
  external checkpoint
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
  The manager then reran the canonical compare and updated
  `proofs/v014/jax_h10_prestep_carry.{json,md}` plus
  `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`.

Verdict: `JAX_MISMATCH_T`. The checkpoint is CPU-loadable, contains an
`OperationalCarry`, paired `OperationalNamelist`, step index `5999`, grid
shape `159 x 66 x 44`, and SHA256
`0896e4a272cbeaa85d1bb969ecae82b047e75a028df45a87ddab4f4572af8dde`.
The canonical same-surface comparison now runs. First mismatch by the proof's
field order is `T`: max_abs `3.3545763228707983`, RMSE
`1.0296598586362888`, worst native key `[12, 17]`, JAX candidate
`-5.813209321660452`, WRF truth `-9.16778564453125`. Other fields remain far
from WRF on the same patch (`P=590`, `PB=1047`, `MU=267`, `MUB=1050`,
`U=6.29`, `V=11.59`, `W=1.73`, `PH=5.10` max_abs), so this is not a narrow
writer-only success. However, WRF's green history `T` source is
`grid%th_phy_m_t0`; the next sprint must attribute JAX theta/history source
semantics before any production dycore source fix.

## Completed Wave 12

- GPT tmux worker (`0:4`, closed after DONE):
  T history/source-attribution sprint
  `.agent/sprints/2026-06-09-v014-t-history-source-attribution/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/jax_t_history_source_attribution.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`.

Verdict: `T_EVOLUTION_MISMATCH_CONFIRMED`. The proof uses the produced h10
step-5999 carry checkpoint and Boole's same-surface WRF target, confirms the
checkpoint hash/size matches both the producer and canonical h10 compare
records, and explicitly separates WRF history `T_HIST_SRC`
(`grid%th_phy_m_t0`) from WRF `T_THM`. Best WRF history `T_HIST_SRC` match is
still `captured_pre_halo_state.theta_minus_300` with max_abs
`3.3545763228707983`, RMSE `1.0296598586362888`. Best WRF `T_THM` match is
`captured_final_carry.t_2ave_minus_300` with max_abs `3.677881697025043`.
P/PB/MU/MUB remain divergent on the same patch (`P=590`, `PB=1047`, `MU=267`,
`MUB=1050` max_abs), so this is not a lone history/source artifact.

## Completed Wave 13

- GPT tmux worker (`0:4`, closed after artifact validation):
  theta-evolution localization sprint
  `.agent/sprints/2026-06-09-v014-theta-evolution-localization/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/jax_theta_evolution_localization.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`.

Verdict: `THETA_MISMATCH_PRESTEP_OR_INPUT`. The proof uses the produced h10
step-5999 carry checkpoint and existing WRF source-derived surfaces, and proves
the proof-local RK mirror agrees with the existing pre-halo helper for theta
(`max_abs=0.0`). The first reachable mismatch is already present before
current-step physics/RK: WRF `T_OLD` / `grid%t_1` versus JAX prestep theta has
max_abs `6.218735851548047`, RMSE `4.638818160588427`; `MU_OLD` context also
differs with max_abs `267.01919069732367`. The current WRF input/reference
surface does not expose explicit step-6000 pre-RK `P/PB/MUB`, so source-changing
dycore edits remain premature. The next sprint must emit or hook explicit WRF
and JAX step-6000 pre-RK `T/P/PB/MU/MUB`.

## Completed Wave 14

- GPT tmux worker (`0:5`, closed after manager validation):
  empirical/static memory map sprint
  `.agent/sprints/2026-06-09-v014-empirical-memory-map/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/empirical_memory_map.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`.

Verdict:
`NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`.
All candidates have `blocks_v014_long_validation_after_grid_parity=false`.
Smallest safe optional memory source sprint is WDM6 `slmsk` shape-only cleanup
(`0.075119 GiB` fp64 at 641x321x50); the only material bit-identical cleanup is
moisture transport velocity reuse when active moisture advection matters
(`0.237621-0.620881 GiB` static estimate). MYNN BouLac, non-radiation column
tiling, post-physics merge, moisture limiter workspace, acoustic carry split,
and FP32 acoustic remain measurement-first or grid-parity-gated.

## Completed Wave 15

- GPT tmux worker (`0:4`, closed after manager superseded the sandbox-blocked
  WRF run with a successful dmpar manager run):
  pre-RK input-boundary sprint
  `.agent/sprints/2026-06-09-v014-pre-rk-input-boundary/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/pre_rk_input_boundary.{py,json,md}`,
  `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`, and
  `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`.

Verdict: `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`. The proof emits
explicit CPU-WRF d02 h10 step-6000 pre-RK truth at
`dyn_em/solve_em.F` after `grid%itimestep` increment and before
`cpl_store_input` / current-step physics/RK. The worker initially reached a
PMIx/socket sandbox blocker; the manager reran the hook outside the worker
sandbox using the existing dmpar `v014_post_rk_refresh` WRF lineage. Two hook
tiles were emitted and parsed with no duplicate disagreement.

All target fields differ before the current step begins: `T` max_abs
`6.218735851548047`, RMSE `4.638818160588427`; `P` max_abs
`589.6789731315657`, RMSE `526.4973831519894`; `PB` max_abs `1047.015625`;
`MU` max_abs `267.01919069732367`; `MUB` max_abs `1050.3046875`. Therefore the
first source-changing fix should not target current-step RK/acoustic,
`small_step_finish`, post-RK refresh, or history-source remapping. The next
debug sprint must trace the JAX h10 step-5999 checkpoint/prestep carry producer
and previous-step WRF/JAX update path until it names the first wrong write,
state handoff, restart/load, or cadence boundary.

## Completed Wave 16

- GPT tmux worker (`0:4`, completed after manager validation):
  prestep carry source trace sprint
  `.agent/sprints/2026-06-09-v014-prestep-carry-source-trace/sprint-contract.md`.
  Completed and manager-validated 2026-06-09. Deliverables:
  `proofs/v014/prestep_carry_source_trace.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`.

Verdict: `PRODUCER_WRITES_BAD_FINAL_CARRY`. The proof rules out checkpoint
serialization/load corruption: raw pickle runtime state, checkpoint API runtime
state, top-level State payload, and `/tmp` round-trip all preserve
`T/P/PB/MU/MUB` exactly. The live nested replay producer starts from native L2
domain load and writes the generated d02 `OperationalCarry` directly via
`write_checkpoint(..., runtime_state=d02_carry)`. The target leaves in that
carry still differ from CPU-WRF h10 step-6000 pre-RK truth: `T` max_abs
`6.218735851548047`, `P` `589.6789731315657`, `PB` `1047.015625`, `MU`
`267.01919069732367`, and `MUB` `1050.3046875`. Scratch leaves such as
`t_2ave` and `mu_save` are sometimes closer but still outside tolerance and are
not eligible as the target pre-RK State leaves.

Manager validation reran:

- `python -m py_compile proofs/v014/prestep_carry_source_trace.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/prestep_carry_source_trace.py`
- `python -m json.tool proofs/v014/prestep_carry_source_trace.json`

## Completed Wave 17

- GPT tmux worker (`0:4`, completed after manager validation):
  previous-step handoff bisection sprint
  `.agent/sprints/2026-06-09-v014-previous-step-handoff-bisect/sprint-contract.md`
  with deliverables `proofs/v014/previous_step_handoff_bisect.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-previous-step-handoff-bisect.md`.

Verdict: `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`. The final producer-shaped replay
matches the existing bad checkpoint target leaves exactly, so the reproducer is
valid. The bad state is already present at d02 completed step 5997 before
parent step 2000, `_operational_force`, or child steps 5998-5999. At that
earliest captured surface, `MUB` is the worst target field with max_abs
`1050.3046875`; `PB` is also already wrong with max_abs `1047.015625`.

CPU live replay remains blocked because `_load_domains` reaches `State.zeros`,
which requires a visible JAX GPU. The targeted replay used `JAX_PLATFORMS=cuda`,
`CUDA_VISIBLE_DEVICES=0`, platform allocator, peak sampled VRAM `9851` MiB,
and wall time about `1215` s. The required CPU validation command then reused
the compact replay artifact and regenerated the repo proof objects.

Manager validation reran:

- `python -m py_compile proofs/v014/previous_step_handoff_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/previous_step_handoff_bisect.py`
- `python -m json.tool proofs/v014/previous_step_handoff_bisect.json`

## Completed Wave 18

- GPT tmux worker (`0:4`, completed after manager validation):
  earlier-source bisection sprint
  `.agent/sprints/2026-06-09-v014-earlier-source-bisect/sprint-contract.md`
  with deliverables `proofs/v014/earlier_source_bisect.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`.

Verdict: `BASE_STATE_SPLIT_DEFINITION_MISMATCH`. Initial d02
`OperationalCarry` `PB/MUB` match native `wrfinput_d02`, but not CPU-WRF
h0/h1/h10 or h10 pre-RK truth. CPU-WRF `PB/MUB` are stable across those
surfaces on the target patch, so replay-time drift is not needed to explain the
bad h10 base carry. Worst base leaf remains `MUB` with max_abs `1050.3046875`;
`PB` max_abs is `1047.015625`.

Targeted GPU replay was required because `_load_domains` reaches
`State.zeros`, which is GPU-gated in this branch. The run used
`JAX_PLATFORMS=cuda`, `CUDA_VISIBLE_DEVICES=0`, platform allocator, and peak
sampled VRAM `9091` MiB. The required CPU validation command then reused the
compact replay artifact and regenerated the repo proof objects.

Manager validation reran:

- `python -m py_compile proofs/v014/earlier_source_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/earlier_source_bisect.py`
- `python -m json.tool proofs/v014/earlier_source_bisect.json`

## Opened Wave 19

- Sprint contract opened:
  `.agent/sprints/2026-06-09-v014-base-state-split-fix/sprint-contract.md`
  with prompt
  `.agent/sprints/2026-06-09-v014-base-state-split-fix/agent-prompt.md`.
  Objective: patch or precisely block
  `src/gpuwrf/integration/d02_replay.py::build_replay_case` native child
  base-state split construction. The fix must reproduce WRF's
  post-initialization `PB/MUB` split or name the exact WRF routine/formula/hook
  needed; a hidden normal production dependency on CPU-WRF `wrfout` history is
  not acceptable.

## Completed Wave 19

- GPT tmux worker (`0:4`, completed after manager validation):
  base-state split fix sprint
  `.agent/sprints/2026-06-09-v014-base-state-split-fix/sprint-contract.md`
  with deliverables `proofs/v014/base_state_split_fix.{py,json,md}` and
  `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`.

Verdict: `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`. No
production source patch was applied. CPU-WRF h0 `PB/MUB` are WRF base-formula
values on post-nest blended h0 terrain, and formula-on-h0-HGT matches within
about `0.06` Pa. A simplified local bilinear+blend reconstruction is rejected:
`PB` patch max `796.2565574348409` Pa and `MUB` patch max
`798.7609739865584` Pa.

Exact missing WRF chain:

- `share/mediation_integrate.F` live-nest `input_from_file` branch after
  `med_interp_domain` and after `blend_terrain` for `nest%ht/nest%mub/nest%phb`.
- generated `inc/nest_interpdown_interp.inc` parent-to-child interpolation for
  `phb/mub/pb`, via `share/interp_fcn.F::interp_fcn_sint` and `share/sint.F`.
- `dyn_em/nest_init_utils.F::blend_terrain` with `spec_bdy_width=5` and
  `blend_width=5`.
- `dyn_em/start_em.F::start_domain_em` base-state recomputation after terrain /
  base blend.

Manager validation reran:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/base_state_split_fix.py`
- `python -m json.tool proofs/v014/base_state_split_fix.json`

## Opened Wave 20

- Sprint contract opened:
  `.agent/sprints/2026-06-09-v014-live-nest-base-hook/sprint-contract.md`
  with prompt
  `.agent/sprints/2026-06-09-v014-live-nest-base-hook/agent-prompt.md`.
  Objective: capture or reproduce the WRF live-nest parent-interpolated /
  blended `HGT/MUB/PHB` and post-`start_domain_em` base recomputation oracle
  needed for the next source-fix sprint.

## Next Manager Actions

1. Run the live-nest base hook/oracle sprint. Start from
   `proofs/v014/base_state_split_fix.json`. The sprint must either produce a
   disposable-WRF savepoint/oracle for post-`blend_terrain` and
   post-`start_domain_em` base fields, or provide a native-port plan precise
   enough to implement WRF's parent interpolation/blend/base recomputation.
2. Keep runtime dycore, pressure-gradient, acoustic, radiation, and
   surface-layer code read-only unless the base-state split fix proof directly
   requires a narrower follow-up contract.
3. Launch source-changing dynamic fixes only after a proof names the first
   failing operator, write, state handoff, or cadence path.
4. Use Opus 4.8 xhigh/max via `claude --permission-mode auto` only after two
   failed GPT attempts on the same static/base or tendency root-cause problem.
5. Keep GPU time for short targeted probes only; no powered TOST, no
   Switzerland equivalence, no FP32 source landing until the dynamic grid
   divergence is named, fixed, or explicitly bounded.

## Non-Goals Until Grid Parity Moves

- No v0.13/v0.14 tag decision based on station TOST alone.
- No FP32 dycore landing that masks the current fp64 grid divergence.
- No broad scheme-long-tail work unless it directly supports the divergence fix.

## Current Manager Update 2026-06-09 16:45 WEST

The older "Next Manager Actions" above are superseded by the later Step-1
source-boundary/theta/QVAPOR proof chain.

Closed since that older wave:

- `step1_rk1_source_boundary`: localized the first material mismatch to WRF
  `first_rk_step_part1` physics-state mutation of `T_STATE`.
- `step1_part1_physics_state_mutation`, `step1_pre_part1_handoff`,
  `step1_jax_loader_tstate`, and `step1_live_nest_theta_semantics`: reduced
  live-nest pre-call `T_STATE` residual from `5.490173101425171 K` to
  `0.00541785382188209 K` with WRF `USE_THETA_M=1` moist-theta semantics plus
  `adjust_tempqv`, but held production patch because same-boundary `QVAPOR`
  was missing.
- `step1_qvapor_precall_truth_schema`: proved existing QVAPOR artifacts were
  post-RK/pre-halo or different-boundary and specified the WRF savepoint.
- `step1_qvapor_precall_savepoint`: generated and validated same-boundary
  pre-call `QVAPOR` truth. Verdict:
  `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`. Filtered root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.

Latest proof facts:

- QVAPOR shape `[44,66,159]`, count `461736`, all finite.
- Prior accepted pre-call fields
  `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT/W_STATE/PH_STATE/PHB` are
  text-identical, max_abs `0.0`.
- No production `src/gpuwrf/**` source has been changed in this savepoint
  sprint.

Next active debug step:

1. Rerun `proofs/v014/step1_live_nest_theta_semantics.py` using the filtered
   same-boundary QVAPOR root.
2. Add worst-cell classification for the remaining `0.0054 K` max_abs:
   boundary band versus interior, plus field-extreme context.
3. If the theta tail is bounded/local while p99 remains near `4.5e-5 K`,
   document the gate semantics and resume the larger base-state split / V10
   driver chain. If same-boundary QVAPOR materially changes the result, decide
   on an init-only patch under a new source-changing sprint contract.
4. Still no TOST, Switzerland, FP32 source landing, or memory source work until
   grid divergence is fixed or explicitly bounded enough for long validation.

## Current Manager Update 2026-06-09 17:00 WEST

The same-boundary QVAPOR rerun is now closed. Verdict:
`STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.

Key facts:

- `proofs/v014/step1_theta_same_qvapor.{py,json,md}` used the validated
  same-boundary pre-call QVAPOR root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- The proof corrected the methodology distinction: raw child QVAPOR remains
  the input to the WRF theta/`adjust_tempqv` transcription; the same-boundary
  root is the accepted WRF QVAPOR comparator.
- Candidate QVAPOR after `adjust_tempqv` matches WRF pre-call QVAPOR closely:
  max_abs `3.838436518426372e-06`.
- Final `T_STATE` residual remains above the `1e-3 K` max_abs gate:
  all-cell/interior max_abs `0.00541785382188209 K`.
- Boundary band (`distance_to_edge <= 5`) max_abs is only
  `0.0005722015491755883 K`; worst cell is interior by the sprint rule:
  zero `{k:1,y:9,x:17}`, Fortran `{i:18,j:10,k:2}`, distance `9`.
- Therefore no production theta/`adjust_tempqv` patch is authorized yet.

Next active debug step:

1. Open a CPU-only disposable-WRF intermediate savepoint sprint for exact
   `adjust_tempqv` internals on the residual path.
2. Emit WRF pre/post `t_2`, QVAPOR, `p_old`, `p_new`, `mub`, `mub_save`,
   `c3h`, `c4h`, `p_top`, and pressure/base inputs for the worst cell or a
   compact field.
3. Use that evidence to decide whether the `0.0054 K` interior tail is a
   formula/transcription issue, pressure/base-input residual, or bounded
   rounding/source-order tail.
4. Keep TOST, Switzerland, FP32 source landing, and memory source work paused
   until this grid-parity branch is fixed or explicitly bounded.

## Current Manager Update 2026-06-09 17:16 WEST

The `adjust_tempqv` intermediate sprint is closed. Verdict:
`STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

Key facts:

- The disposable WRF hook emitted exact internals for d02 Fortran
  `{i:18,j:10,k:2}` after a manager unsandboxed MPI rerun. The earlier Codex
  PMIx blocker is no longer the sprint result.
- Proof objects are
  `proofs/v014/step1_adjust_tempqv_intermediate.{py,json,md}` and
  `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`.
- `p`, `mub_save`, `c3h`, `c4h`, and `p_top` match WRF.
- Current `mub` differs by `17.67503987130476 Pa`; `pb_new_equiv` and `p_new`
  differ by `17.49400702366256 Pa`.
- The known `t_2_post` residual remains `0.00541785382188209 K`.

Manager decision:

Do not chase a thermodynamic formula patch next. The next active debug step is
a CPU-only current-`MUB/PB` base-input split around WRF live-nest terrain/base
blending and the JAX live-nest base-init reconstruction. TOST, Switzerland,
FP32 source landing, and memory source work remain paused until this branch is
fixed or explicitly bounded.

Opened sprint:
`.agent/sprints/2026-06-09-v014-step1-current-mub-base-input-split`.
The contract is proof-only and must classify the `17.5 Pa` current-`mub` /
`pb_new` mismatch before any production source patch.

## Current Manager Update 2026-06-09 17:32 WEST

The current-`MUB/PB` split sprint is closed. Verdict:
`STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

Key facts:

- WRF `adjust_tempqv` consumes transient post-`blend_terrain` /
  pre-`start_domain` current `MUB`.
- The prior JAX theta proof used final post-`start_domain` base `MUB` for that
  earlier call.
- WRF adjust hook current `MUB`: `86812.25`.
- Proof-side direct WRF blend `MUB`: `86812.250452109511`.
- JAX final base `MUB`: `86794.574960128695`.
- WRF pre-part1 final `MUB`: `86794.5703125`.
- WRF source formula is `p_new = p + c4h + c3h*mub + p_top`, not the grouped
  `p + c3h*(mub+p_top) + c4h` form.

Next active step:

Open a narrow source-changing sprint to add a transient live-nest adjust-base
path for theta/QV adjustment only, while keeping final post-`start_domain`
BaseState unchanged. Rerun the Step-1 theta/QV proof after that change. TOST,
Switzerland, FP32 source landing, and memory source work remain paused.

Opened sprint:
`.agent/sprints/2026-06-09-v014-step1-transient-adjust-base-fix`.
This is the first source-changing sprint after the MUB split and is limited to
`src/gpuwrf/integration/d02_replay.py` plus proof artifacts.

## Current Manager Update 2026-06-09 17:48 WEST

The transient adjust-base helper sprint is closed. Verdict:
`STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

Key facts:

- `src/gpuwrf/integration/d02_replay.py` now has
  `_wrf_live_nest_transient_adjust_mub`.
- The helper exposes WRF's transient post-`blend_terrain`/pre-`start_domain`
  current `MUB` for live-nest `adjust_tempqv`.
- Final post-`start_domain` BaseState semantics are unchanged.
- Corrected theta max_abs against same-boundary WRF pre-call truth is
  `5.788684885033035e-05 K`, versus prior `0.00541785382188209 K`.
- Corrected QVAPOR max_abs is `5.970267497393267e-08`.

Manager decision:

This closes the candidate/helper proof, not full production Step-1. Next active
step is a source-changing wiring sprint for WRF theta_m conversion plus
`adjust_tempqv` in the production live-nest init consumer, using the new helper.
TOST, Switzerland, FP32 source landing, and memory source work remain paused.

Opened sprint:
`.agent/sprints/2026-06-09-v014-step1-live-nest-theta-qv-wiring`.
This sprint must wire production live-nest theta/QV initialization and then run
the Step-1 16-field comparison or name the next exact boundary.

## Current Manager Update 2026-06-09 19:59 WEST

Post-Theta/QV wiring, the debug lane moved upstream through `P/PH/MU/W`
live-nest initialization rather than TOST, Switzerland, FP32, or broad memory.

Latest committed facts:

- `ee6cbbe1` merged the parallel memory/FP32 side manager. It implemented only
  the non-conflicting WDM6 `slmsk` shape cleanup in
  `src/gpuwrf/coupling/scan_adapters.py`; CPU proof, 85 WDM6 savepoint tests,
  and WDM6 operational smoke passed. Larger FP32/memory source work remains
  locked behind grid-parity surfaces.
- `f73542c0` closed live-nest perturb-state init localization. Remaining
  residuals were reduced but not closed: `P_STATE 69.96875 -> 3.9458582235092763`
  Pa, `MU_STATE 13.256103515625 -> 0.047773029698646496` Pa, and `W_STATE`
  essentially closed to `1.2992081932505783e-07` m/s.
- `211328d2` updated the memory roadmap so WDM6 is no longer stale `Pending`.

Active sprint:

- `.agent/sprints/2026-06-09-v014-step1-start-domain-perturb-subsurface`.
- Active worker: tmux `0:4` (`gpt-start-domain`).
- WRF scratch compile and 18-second two-domain CPU replay succeeded.
- The worker emitted `112` WRF `start_domain(nest,.TRUE.)` internal truth files:
  28 tiles for each of `after_hypsometric_p_al_alt`, `before_press_adj`,
  `after_press_adj`, and `after_w_surface_branch`.
- The only repo artifact currently written by this sprint is
  `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`.
- Required proof/report/json/review files are still pending; do not close the
  sprint until they exist and validation commands pass.

Management review:

- `.agent/sprints/2026-06-09-v014-management-review-02` is due/open, but an
  Opus/Claude launch at 19:58 WEST hit the Claude plan/limit dialog before it
  could write `.agent/reviews/2026-06-09-v014-management-review-02.md`.
- The blocked Claude tmux window was closed for hygiene. Reattempt Opus when
  available; do not fake the review with a manager-authored substitute.

Manager decision:

Continue the current start-domain proof to a strict verdict. If it proves a
narrow GPU-native `d02_replay.py` fix, validate and commit it. If it only
localizes the gap, open the next minimal truth-surface sprint. TOST,
Switzerland, FP32 source work, and broad memory work remain paused until
per-cell grid parity is explained or bounded by proof.

## Current Manager Update 2026-06-09 20:05 WEST

The start-domain perturbation subsurface sprint is closed and pushed:

- Proof commit: `b659a3dc`.
- Formal closeout commit: `66c091fc`.
- Verdict:
  `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.
- Validation rerun: CPU-only proof, JSON validation, `git diff -- src/gpuwrf`
  empty, `git diff --check` clean for sprint artifacts.

Meaning:

- WRF source ordering around live-nest `start_domain(nest,.TRUE.)` is now
  proven closely enough: hypsometric `P/al/alt`, `press_adj`, and W-surface
  handling are not the remaining unknown.
- A production `P/MU` patch is still not safe: current JAX inputs leave `P`
  max_abs `3.9458582235092763 Pa` and `MU` max_abs
  `0.047773029698646496 Pa`.

Opened next sprint:

- `.agent/sprints/2026-06-09-v014-step1-jax-start-domain-input-split`.
- Objective: split current JAX live-nest `start_domain` inputs for final
  blended `HT`, `PB/MUB/PHB`, `PH_STATE`, pre-`press_adj` `MU`, and diagnosed
  `AL/ALT` against the existing WRF internal `after_hypsometric` truth.
- Optional source edit is limited to `src/gpuwrf/integration/d02_replay.py`
  only if the source/input bug is exact, narrow, and GPU-native.

TOST, Switzerland, FP32 source work, broad memory work, and GPU validation stay
paused.

## Current Manager Update 2026-06-09 20:36 WEST

The JAX start-domain input split sprint is closed locally and ready to commit:

- Sprint:
  `.agent/sprints/2026-06-09-v014-step1-jax-start-domain-input-split`.
- Verdict:
  `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.
- Manager validation passed:
  `py_compile`, CPU-only proof rerun, JSON validation, and
  `git diff -- src/gpuwrf` empty.
- No production source was changed.

Meaning:

- The dominant residual is now diagnosed `AL/ALT`, fed by base-state
  reconstruction.
- Direct WRF ALT substitution reduces pressure max_abs from
  `3.9458582235092763` to `0.07605321895971429`.
- FP32 ALT diagnosis with WRF `PHB+MUB` reduces pressure max_abs to
  `0.0859375`.
- A production patch is still not safe: the best proof-local WRF-order
  fp32/cp=1004.5 base candidate still leaves `P_STATE` max_abs `2.828125` and
  `MU_STATE` max_abs `0.011962890625`.
- Refuted as dominant: terrain/final blend, time-level selection, `PH_STATE`,
  pre-press `MU`, `PB` alone, and theta alone.

Next required sprint:

- Emit or reproduce the exact WRF `start_domain_em` base-state boundary before
  the hypsometric `AL/ALT` pass: `p_surf`, post-assignment `MUB`,
  `PB/T_INIT/ALB`, `PHB`, active hybrid coefficients, flags, and scalar
  constants.
- Once WRF-equivalent `PHB+MUB` reconstruction is closed, apply the already
  proven `P/MU/W` perturbation init path under a narrow source-changing
  `d02_replay.py` sprint.

Memory/FP32 status:

- The previous parallel memory/FP32 side manager was merged as `ee6cbbe1`.
  It closed only the non-conflicting WDM6 `slmsk` cleanup. Larger memory and
  FP32 source work remains blocked by this live-nest/grid-parity lock.
- A manual relaunch contract for a secondary memory manager now exists at
  `memory_manager_contract_260609.md`; it must respect the same source locks.

TOST, Switzerland, FP32 source work, broad memory source work, and long GPU
validation remain paused.

## Current Manager Update 2026-06-09 22:30 WEST

The Mythos kernel pass landed a validated source fix for the live-nest
`start_domain` init family:

- Proof:
  `proofs/v014/mythos_kernel_fix_260609.{py,json,md}`.
- Review:
  `.agent/reviews/2026-06-09-v014-mythos-kernel-fix.md`.
- Changed source:
  `src/gpuwrf/integration/d02_replay.py` and
  `src/gpuwrf/nesting/interp.py`.
- Verdict:
  `MYTHOS_KERNEL_FIX_START_DOMAIN_P_MU_W_CLOSED_FP32_LIBM_SINT_BLEND_BIT_EXACT`.
- Manager rerun passed:
  `py_compile`, CPU-only Mythos proof, JSON validation, and `git diff --check`.

Meaning:

- The remaining `p_surf -> MUB` gap was exact float32 libm provenance and source
  grouping: CPU-WRF/gfortran calls scalar glibc `expf/logf/powf`, and
  `(...)**0.5` compiles to `powf(x,0.5)`, not `sqrtf`.
- The previous live-nest terrain SINT/blend path also evaluated in float64 while
  WRF uses REAL(4); this is now closed in the init path.
- Init/start-domain gates versus WRF internal truth now pass:
  `P_STATE 69.96875 -> 0.0390625 Pa`, `MU_STATE 13.256103515625 ->
  4.547473508864641e-13 Pa`, and `W_STATE 0.7605466842651367 ->
  5.551115123125783e-17 m/s`.
- Base fields after patch are effectively exact against truth:
  `MUB=0`, `PB=0`, `PHB=4.547473508864641e-13`,
  `HT=4.547473508864641e-13`.

Honest remaining frontier:

- The strict Step-1 16-field one-RK-step comparison still diverges after the
  now-closed init:
  `P max_abs=975.1236470550566`, `PH=63.82327410901786`,
  `MU=14.007953430216503`, `W=2.6401070776077424`,
  first divergent field `T`.
- The Mythos attribution probe shows this is real post-init dynamics state
  divergence in the `PH/MU/P/W` acoustic/mass/vertical lane or one-step namelist
  parity, not pressure-diagnosis semantics and not the fixed init family.
- The next grid-parity debug sprint should first freeze one-step namelist
  parity (`acoustic_substeps`, `epssm`, damping) and then rerun/rebuild the RK1
  substage comparator from the now-closed init state before any dycore source
  edit.

Memory/FP32 handoff:

- The principal now wants the memory improvement/fix lane handed to Mythos as a
  parallel branch. Endpoint is not a report-only lane: all known memory issues
  should be fixed where technically safe, any newly found material memory issue
  should be fixed or proven not worth/possible, and every claim must have proof.
- Because the grid-parity branch still has a real dynamics frontier, Mythos
  memory work must run in an isolated worktree/branch and return proof plus a
  merge recommendation. Default fp64 production behavior must remain
  bit-identical unless a sprint explicitly declares and proves a semantic or
  mixed-precision mode.
- Long validation, Switzerland, and TOST remain paused until grid parity and
  memory branch interactions are reviewed.

## Current Manager Update 2026-06-09 23:05 WEST

The manager reran the focused Step-1 RK1 substage comparator on the committed
Mythos-init branch:

- Proof:
  `proofs/v014/step1_t_p_operator_localization.{py,json,md}`.
- Review:
  `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`.
- Validation:
  CPU-only rerun, JSON validation, `git diff --check` for proof artifacts.
- Verdict changed from stale pre-Mythos
  `...RK1_T_STATE` to
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_P_STATE`.

Meaning:

- The previous `T_STATE` frontier was partly stale because it was measured
  before the init/start-domain patch.
- With init now closed, the first strict substage mismatch is still `T_STATE`,
  but the first **material** T/P-family mismatch is now `P_STATE` at
  `after_rk_addtend_before_small_step_prep`, RK1.
- Top material residuals at that same boundary are tendency-family:
  `PH_TEND max_abs=794096.1875`, `RW_TEND max_abs=131390.765625`,
  `PH_TENDF max_abs=27082.453125`.
- RK1 `small_step_prep` work arrays still match for `T_WORK=0.0` and
  `P_WORK=0.0`, so the next proof target remains before small-step prep, not
  inside acoustic substeps.
- Final strict Step-1 comparison remains divergent but improved versus the
  stale pre-Mythos base: top `P max_abs=975.1236470550566`,
  `PH=67.35257977855315`, `MU=14.123722376832347`,
  `W=2.6401070783205864`.

Next active grid-parity sprint:

- `.agent/sprints/2026-06-09-v014-step1-rk1-p-state-source-split`.
- Objective: split WRF/JAX RK1 stage-entry construction after WRF
  `first_rk_step_part1/part2` and before JAX `small_step_prep`, starting with
  `P_STATE` and the huge `PH/RW` tendency-family residuals.
- Do not continue acoustic-substep debugging until this earlier boundary is
  closed.

Parallel status:

- Mythos is active in tmux `0:1` on the isolated memory/FP32 lane. It created or
  is creating `.codex/worktrees/mythos-memory-v014` and must not edit the main
  worktree.
- TOST, Switzerland, long GPU validation, and direct merge of Mythos work remain
  paused until manager review.

## Current Manager Update 2026-06-09 23:25 WEST

The RK1 `P_STATE` source-split sprint is closed:

- Sprint:
  `.agent/sprints/2026-06-09-v014-step1-rk1-p-state-source-split`.
- Proof:
  `proofs/v014/step1_rk1_p_state_source_split.{py,json,md}`.
- Review:
  `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`.
- Verdict:
  `STEP1_RK1_P_STATE_SOURCE_REFUTED_STALE_PROOF_LOADER_BYPASS_NEXT_T_TENDF`.
- Manager rerun passed:
  `py_compile`, CPU-only proof rerun, JSON validation, and `git diff --check`.

Meaning:

- The previous post-Mythos `P_STATE` material frontier was a stale proof-loader
  artifact. The proof-local live-nest Step-1 helper still bypassed Mythos'
  production `start_domain` perturbation init.
- With the production Mythos perturbation init applied in the proof capture,
  RK1 `P_STATE` at `after_rk_addtend_before_small_step_prep` drops from
  `69.96875 Pa` to `0.0390625 Pa`, below the `1.0 Pa` material gate.
- `P_STATE/MU_STATE/W_STATE/PH_STATE` are below material gates through
  `after_first_rk_step_part1`, `after_first_rk_step_part2`, and RK1
  `after_rk_addtend_before_small_step_prep` under patched-init capture.
- RK1 `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK` remain exact at
  `small_step_prep/calc_p_rho(step=0)`.

Next active grid-parity boundary:

- Split WRF `first_rk_step_part2` `T_TENDF`.
- Then split RK1 `after_rk_addtend` `T_TEND/PH_TEND/RW_TEND`.
- Compare against JAX `compute_advection_tendencies` and
  `_augment_large_step_tendencies` under patched-init capture.
- Do not enter acoustic substeps for this issue; the earlier tendency boundary
  is still open.

Parallel status:

- Mythos remains active in tmux `0:1` on the isolated memory/FP32 lane and has
  started the exact-branch memory preflight plus moisture-reuse work.
- Do not start TOST, Switzerland, or long GPU validation yet.

## Current Manager Update 2026-06-09 23:33 WEST

The Step-1 tendency contract split sprint is closed:

- Sprint:
  `.agent/sprints/2026-06-09-v014-step1-tendency-contract-split`.
- Proof:
  `proofs/v014/step1_tendency_contract_split.{py,json,md}`.
- Review:
  `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`.
- Verdict:
  `STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES`.
- Manager rerun passed:
  `py_compile`, CPU-only proof rerun, JSON validation, and `git diff --check`.

Meaning:

- The earlier `P_STATE`/`MU_STATE`/`W_STATE` material frontiers remain closed
  under patched-init capture.
- The next exact failure is WRF `first_rk_step_part2` theta source-leaf
  construction: full-domain `T_TENDF` at `after_first_rk_step_part2` differs
  from the current JAX dry source bundle by max_abs `2457.5830078125`, RMSE
  `21.20870100357482`.
- Source-save pre-addtend `T_TENDF` also differs (max_abs
  `1326.432250976562`), proving boundary/spec/acoustic explanations are too
  late for the first failure.
- Proof-local `rad_rk_tendf=1` did not move the boundary, so the simple
  radiation-cadence explanation is falsified.

Next active grid-parity boundary:

- WRF `first_rk_step_part2` internals: emit disposable truth after
  `calculate_phy_tend`, after `update_phy_ten`, and after
  `conv_t_tendf_to_moist`.
- Include raw `RTH*TEN` / `T_HIST_SRC` contributors and the current JAX dry
  physics source bundle.
- Do not patch `_augment_large_step_tendencies`, `relax_bdy_dry`,
  `rk_addtend_dry`, `spec_bdy_dry`, or acoustic code before this earlier source
  boundary is split.

Parallel status:

- Mythos remains active in tmux `0:1` on the isolated memory/FP32 lane. Its
  final merge is still pending manager review because the memory proof/closeout
  must be internally consistent before acceptance.
- Do not start TOST, Switzerland, or long GPU validation yet.

## Current Manager Update 2026-06-09 23:40 WEST

Memory/FP32 lane has been reviewed, merged, and pushed:

- `26815feb`: MYNN BouLac column tiling plus shared stage transport velocities.
- `bc847db2`: FP32 R0 default-inert acoustic precision-mode contract.
- `8f735a56`: Mythos memory/FP32 proofs, roadmaps, and sprint closeout.
- `e0091707`: Fable/Mythos token-conservation policy plus post-merge proof
  refresh.

Post-merge manager checks passed:

- `close_sprint.py` for the Mythos memory lane;
- `py_compile` and JSON validation for Mythos/FP32/Step-1 proofs;
- `tests/test_operational_namelist_cache_key.py` (5 passed);
- CPU rerun of `proofs/v014/mythos_memory_fixes_260609.py`;
- CPU rerun of `proofs/v014/fp32_acoustic_static_audit.py`;
- CPU rerun of `proofs/v014/step1_tendency_contract_split.py`, which kept the
  same verdict:
  `STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES`.

Fable/Mythos policy is now durable: conserve Fable tokens; use GPT 5.5 first for
normal validation failure collection/localization/direct fixes; escalate only
the unresolved hard core to Fable after `/compact`.

Next active grid-parity sprint:

- `.agent/sprints/2026-06-09-v014-step1-part2-source-leaves-split`.
- Objective: split WRF `first_rk_step_part2` internals after
  `calculate_phy_tend`, `update_phy_ten`, and `conv_t_tendf_to_moist`, including
  raw `RTH*TEN` / `T_HIST_SRC` contributors and the current JAX dry source
  bundle.
- This is a GPT CPU-only sprint. Do not use GPU, TOST, Switzerland, FP32/memory
  source work, Hermes, or Fable/Mythos.

## Current Manager Update 2026-06-09 21:35 WEST

The base-state boundary sprint is closed locally and ready to commit:

- Sprint:
  `.agent/sprints/2026-06-09-v014-step1-base-state-boundary`.
- Verdict:
  `STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC`.
- Manager validation passed:
  `py_compile`, CPU-only proof rerun, JSON validation, `close_sprint.py`, and
  `git diff -- src/gpuwrf` empty.
- No production source was changed.

Meaning:

- The prior base-state hypothesis is supported and narrowed further.
- The decisive remaining surface is exact WRF `p_surf -> MUB` arithmetic before
  the `AL/ALT` pass.
- Current/proof-local fp32/cp=1004.5 `p_surf` formula still leaves
  `P_STATE=2.828125 Pa` and `MU_STATE=0.011962890625 Pa`.
- Substituting WRF-emitted `MUB` into the same proof-local base/AL/ALT path
  reduces downstream `P_STATE` to `0.40625 Pa` and `MU_STATE` to
  `0.001220703125 Pa`, below the sprint gates.
- Refuted as dominant: terrain/blend input, cp constant alone,
  coefficient indexing, PH/MU time-level selection, and PHB integration order.

Memory/FP32 side-manager status:

- `proofs/v014/memory_manager_260609.*` and
  `.agent/reviews/2026-06-09-v014-memory-manager-260609.md` now exist.
- Recommendation is `REVIEW_ONLY`: no source changes, GPU not used, and all
  remaining memory/FP32 source work stays blocked by grid-parity locks until
  this live-nest/start-domain bug is fixed.

Principal-directed pause:

- `mythos_kernel_contract_260609.md` is written for the experimental model.
- The primary manager should not launch another normal debug sprint while the
  principal tests the Mythos model.
- If Mythos does not land a validated fix, resume with a sprint that emits a
  disposable WRF truth surface immediately around the `p_surf` expression and
  `grid%MUB(i,j) = p_surf - grid%p_top`, or proves a WRF-compatible fp32/libm
  helper. Gate any production `d02_replay.py` patch on `P_STATE <= 1 Pa`,
  `MU_STATE <= 0.01 Pa`, no production CPU-WRF dependency, and no timestep-loop
  host/device transfer.

TOST, Switzerland, FP32 source work, broad memory source work, and long GPU
validation remain paused.

## Current Manager Update 2026-06-09 23:59 WEST

The principal resumed the manager goal: complete and publish v0.14. The
experimental Mythos kernel lane already landed its accepted fix before this
handoff section; the Mythos memory/FP32 lane is also manager-reviewed, merged,
gated, committed, and pushed.

Current commits to preserve across compaction:

- `26815feb`: MYNN BouLac column tiling plus shared stage transport velocities.
- `bc847db2`: FP32 R0 default-inert acoustic precision-mode contract.
- `8f735a56`: Mythos memory/FP32 proofs, roadmaps, and sprint closeout.
- `e0091707`: Fable/Mythos token-conservation policy plus post-merge proof
  refresh.
- `374e8c8f`: opened the active Step-1 part2 source-leaves split sprint.
- `dc6955f4`: refreshed `PROJECT_PLAN.md` and the v0.14 release checklist after
  the memory merge.

Active work:

- GPT-5.5 xhigh in `tmux 0:3`, sprint
  `.agent/sprints/2026-06-09-v014-step1-part2-source-leaves-split`, CPU-only.
- The sprint is splitting WRF `first_rk_step_part2` internals after
  `calculate_phy_tend`, `update_phy_ten`, and `conv_t_tendf_to_moist`, including
  raw `RTH*TEN` / `T_HIST_SRC` and the current JAX dry source bundle.
- The worker compiled a disposable WRF copy and is currently clearing local run
  setup blockers caused by old hard-coded WRF instrumentation paths. This is not
  a Fable/Mythos escalation case.

Resource policy:

- Do not send Hermes updates.
- Conserve Fable/Mythos (`tmux 0:1`) for unresolved hard debug cores only. Use
  GPT first for validation failure collection/localization/direct fixes. Before
  any new Fable assignment, send `/compact`, wait, then send one endpoint-defined
  prompt with delayed repeated Enter presses.
- No TOST, Switzerland, Grid-Delta Atlas campaign, or FP32 R1/R2 implementation
  until the current fp64 grid-parity frontier is fixed or explicitly bounded.

## Current Manager Update 2026-06-10 00:17 WEST

The Step-1 part2 source-leaves split sprint is closed and manager-gated.

Artifacts:

- `proofs/v014/step1_part2_source_leaves_split.py`
- `proofs/v014/step1_part2_source_leaves_split.json`
- `proofs/v014/step1_part2_source_leaves_split.md`
- `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part2-source-leaves-split.md`

Verdict:

`STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`.

Key facts:

- WRF `update_phy_ten` closes exactly as `T_TENDF = pre + active RTH` on the
  nested interior, max_abs `0.0`.
- WRF `conv_t_tendf_to_moist` closes to roundoff, nested-interior max_abs
  `0.00016236981809925055`.
- post-conversion equals `after_first_rk_step_part2`, max_abs `0.0`.
- current patched-init JAX dry `T_TENDF` remains divergent: max_abs
  `2457.5830078125`, RMSE `21.674279301376934`.
- source-save sparse `T_TENDF` also remains divergent versus current JAX dry:
  max_abs `1326.432250976562`, RMSE `97.71886125389001`.
- active raw WRF leaves for this case are `RTHRATEN` and `RTHBLTEN`; dominant
  active raw leaf is `RTHBLTEN`.
- inactive WRF `RTH*TEN` leaves can contain uninitialized junk/NaNs and must not
  be ranked causally.

Manager validation passed:

- `python -m py_compile proofs/v014/step1_part2_source_leaves_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/manager_step1_part2_source_leaves_split.validated.json`
- `git diff --check`
- `python scripts/close_sprint.py .agent/sprints/2026-06-09-v014-step1-part2-source-leaves-split`

Next active work:

- Open an implementation sprint for true WRF dry physics source leaves for
  active `RTHRATEN`/`RTHBLTEN` before `_augment_large_step_tendencies`.
- Gate the fix on the same Step-1 proof moving near-zero, then rerun the strict
  short grid-field falsifier.
- Do not use aggregate post-physics state deltas as the fix unless a
  scheme-level raw-leaf proof closes this same gate.
- This is a normal GPT/manager implementation lane first. Do not spend
  Fable/Mythos unless the direct source-leaf implementation becomes a hard
  unresolved debug core.

## Current Manager Update 2026-06-10 00:43 WEST

The dry source-leaf implementation sprint is closed and manager-gated as a
valid blocked boundary proof, not as a fix.

Artifacts:

- `proofs/v014/step1_dry_source_leaf_fix.py`
- `proofs/v014/step1_dry_source_leaf_fix.json`
- `proofs/v014/step1_dry_source_leaf_fix.md`
- `.agent/reviews/2026-06-10-v014-dry-source-leaf-fix.md`
- `.agent/sprints/2026-06-10-v014-dry-source-leaf-fix/manager-closeout.md`

Verdict:

`DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`.

What changed:

- MYNN now exposes a scheme-local `RTHBLTEN` helper.
- `rad_rk_tendf=1` source mode now mass-couples `RTHRATEN + RTHBLTEN` into
  `DryPhysicsTendencies.t_tendf`.
- The MYNN theta delta is removed from the later non-dry state update in this
  source mode to avoid double application.
- Focused CPU regression passes.

Key proof numbers:

- Patched JAX dry `T_TENDF` is active but too small: max_abs
  `260.83156991819124`.
- WRF top active source `RTHBLTEN` remains max_abs `2522.90576171875`.
- Final WRF after-conv vs patched JAX dry residual remains max_abs
  `2457.575215120763`, RMSE `21.445918959761645`.
- Forced-radiation falsifier only moves max_abs to `2454.161554535577`, so
  held `RTHRATEN` is secondary to MYNN source fidelity.
- WRF `conv_t_tendf_to_moist` also matters: after-update vs after-conv max_abs
  `224.50967407226562`, RMSE `4.572429855170764`.

Manager validation passed:

- `python -m py_compile proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py tests/test_v014_dry_source_leaf_wiring.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool proofs/v014/step1_dry_source_leaf_fix.json`
- `python -m json.tool proofs/v014/step1_part2_source_leaves_split.json`
- `git diff --check`
- `python scripts/close_sprint.py .agent/sprints/2026-06-10-v014-dry-source-leaf-fix`

Next active work:

- Open one coherent GPT-5.5 source-fidelity sprint, not Fable/Mythos yet.
- The sprint must split MYNN PBL adapter/kernel inputs and outputs against WRF
  `RTHBLTEN/RQVBLTEN`, seed or refresh held `RTHRATEN` at the same Step-1
  boundary, implement WRF `conv_t_tendf_to_moist` before feeding
  `DryPhysicsTendencies.t_tendf`, and rerun the strict Step-1 proof.
- TOST, Switzerland, broad FP32, and broad memory remain paused until this
  source-fidelity frontier is fixed or explicitly bounded.

## Current Manager Update 2026-06-10 01:12 WEST

The Step-1 source-fidelity closure sprint is closed and manager-gated as a
successful narrowing sprint, not a parity fix.

Artifacts:

- `proofs/v014/step1_source_fidelity_closure.py`
- `proofs/v014/step1_source_fidelity_closure.json`
- `proofs/v014/step1_source_fidelity_closure.md`
- `.agent/reviews/2026-06-10-v014-step1-source-fidelity-closure.md`
- `.agent/sprints/2026-06-10-v014-step1-source-fidelity-closure/manager-closeout.md`

Verdict:

`STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.

What changed:

- MYNN source leaf now carries `rqvblten`.
- `rad_rk_tendf=1` source-leaf mode now applies WRF
  `conv_t_tendf_to_moist` before `DryPhysicsTendencies.t_tendf`.
- `rad_rk_tendf=0` remains on the existing branch.

Key proof numbers:

- Strict WRF after-conv vs current JAX dry `T_TENDF`: max_abs
  `2457.578397008898`, RMSE `21.364579991779515`.
- JAX mass-coupled MYNN `RTHBLTEN`: max_abs `260.83156991819124` vs WRF
  `2522.90576171875`.
- JAX mass-coupled qv source: max_abs `0.045505018412171354` vs WRF
  `QV_TEND` `0.4930315017700195`.
- Same-boundary scalar inputs are not the order-10 error: `T` max_abs
  `5.788684885033035e-05`, `QV` max_abs `5.969281098756885e-08`, `P` max_abs
  `0.0390625`.
- Forcing radiation only moves max_abs to `2454.113955669592`; held
  `RTHRATEN` is secondary.
- WRF oracle active sources still close the accepted formula: max_abs
  `0.00016236981809925055`, RMSE `8.089162788029723e-07`.

Manager validation passed:

- `python -m py_compile proofs/v014/step1_source_fidelity_closure.py proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py tests/test_v014_dry_source_leaf_wiring.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool` on all three proof JSONs.
- `git diff --check`
- `python scripts/close_sprint.py .agent/sprints/2026-06-10-v014-step1-source-fidelity-closure`

Next active work:

- Fable/Mythos completed the MYNN hard sprint. The MYNN driver/kernel source
  deficit was root-caused to missing WRF first-call `mym_initialize` level-2
  equilibrium QKE initialization and fixed in production via
  `mynn_coldstart_init_columns` / `mynn_coldstart_qke_from_state`.
- Open a GPT-5.5 xhigh surface-layer boundary sprint next. Endpoint: emit and
  compare exact WRF Step-1 `module_sf_mynn`/`sfclayrev` in/out hooks for
  `TSK/ZNT/UST/HFX/QFX`, first-call `flag_iter`/UST first-guess behavior, and
  roughness/skin-temperature sourcing; port the local JAX surface adapter fix if
  proven; rerun strict Step-1 proofs against deterministic rmol-pinned truth.
- TOST, Switzerland, broad FP32, and broad memory remain paused.

## Current Manager Update 2026-06-10 02:24 WEST

The Fable/Mythos MYNN source-output sprint is closed and manager-gated as a
real correctness fix plus a narrower remaining frontier.

Artifacts:

- `proofs/v014/mynn_driver_source_output_fix.py`
- `proofs/v014/mynn_driver_source_output_fix.json`
- `proofs/v014/mynn_driver_source_output_fix.md`
- `proofs/v014/mynn_driver_source_output_fix_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`
- `.agent/sprints/2026-06-10-v014-mynn-driver-source-output-fable/manager-closeout.md`
- `tests/test_v014_mynn_coldstart_init.py`

Verdict:

`MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`.

What changed:

- Added WRF-equivalent MYNN cold-start QKE initialization:
  `mynn_coldstart_init_columns` in `src/gpuwrf/physics/mynn_pbl.py`.
- Exposed production coupling through `mynn_coldstart_qke_from_state` in
  `src/gpuwrf/coupling/physics_couplers.py`.
- Wired d02 replay cold-start seeding to the WRF-equivalent initializer.
- Refreshed Step-1 source proof artifacts and same-input contract builder.

Key proof numbers:

- With WRF driver inputs and WRF-init QKE, JAX MYNN reproduces WRF raw `RTHBLTEN`
  with strong-cell ratio median `0.9982`, corr `1.0000`, RMSE `2.6e-06`.
- JAX `RQVBLTEN` strong-cell ratio median is `0.9735`, corr `0.9998`.
- Strict Step-1 after-conv residual improved from max_abs `2457.578397008898`,
  RMSE `21.364579991779515` to max_abs `1497.6112512148795`, RMSE
  `13.468453371786723`.
- The remaining blocker is the surface-layer flux/input boundary: `ustar` bias
  `-0.077` with max `0.176`, `HFX` RMSE `24.6 W/m^2`, land `TSK` differences up
  to `8.3 K`, and `ZNT` roughness differences up to `0.97 m`.

Manager validation passed:

- `python scripts/close_sprint.py .agent/sprints/2026-06-10-v014-mynn-driver-source-output-fable`
- 17 targeted MYNN/source tests passed.
- `python proofs/v014/mynn_driver_source_output_fix.py`
- strict Step-1 source proof reruns and JSON validation.
- `git diff --check`

Next active work:

- Start GPT-5.5 xhigh surface-layer boundary sprint, not Fable/Mythos yet.
- Keep TOST, Switzerland, broad FP32, and broad memory paused until this boundary
  is fixed or explicitly bounded.
