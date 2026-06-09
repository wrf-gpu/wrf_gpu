# Project Plan

Status (2026-06-10 00:43 WEST): **v0.14 is the active closure target. Grid-cell
parity gates the release; proof-backed v0.14 memory fixes are merged; TOST
remains a final validation gate, not the next blind GPU marathon.**

Current release-manager state:

| Lane | Status | Next manager action |
|---|---|---|
| Step-1/grid-cell parity | Active. Source-leaf plumbing sprint is closed as blocked, not fixed. JAX now emits nonzero dry `T_TENDF` in `rad_rk_tendf=1` mode, but the strict Step-1 proof remains red: after-conv residual max_abs `2457.575215120763`, RMSE `21.445918959761645`; WRF active `RTHBLTEN` is still much larger (max_abs `2522.90576171875`) than JAX source leaf (max_abs `260.83156991819124`). | Open one coherent GPT source-fidelity sprint: split MYNN `RTHBLTEN/RQVBLTEN` against WRF, seed/refresh held `RTHRATEN` at Step 1, implement WRF `conv_t_tendf_to_moist`/`QV_TEND` before dry source injection. Gate on strict Step-1 proof collapse or one narrower WRF-anchored blocker. No TOST/Switzerland until fixed or bounded. |
| Memory/FP32 | Memory lane accepted and merged: MYNN BouLac leading-column tiling (`26815feb`) cuts the measured whole-batch MYNN temp from 14.7 GiB to 3.2 GiB on the 641x321x50 target; shared RK-stage transport velocities landed as exact hygiene; exact-branch memory preflight is green at 8116 MiB compute peak, 378 s warm-cache. FP32 R0 default-inert acoustic precision-mode contract landed (`bc847db2`), but R1/R2 mixed precision remains blocked by the open fp64 dynamics frontier. | Do not spend Fable/Mythos on memory now. After grid parity closes, rerun exact-branch memory preflight on the final candidate and then decide whether FP32 R1 belongs in v0.14 or a post-release lane. |
| GPU validation hygiene | Runbook and wrappers exist (`docs/GPU_RUNBOOK.md`, `scripts/run_gpu_lowprio.sh`, `scripts/run_powered_tost_n15.sh`). | Keep GPU jobs serialized through the lock wrapper; long validation starts only after grid and memory branches stabilize. |
| Switzerland/Gotthard | CPU truth/cases exist, but no post-v0.14-fix GPU-vs-CPU proof yet. | Run as v0.14 validation gate after parity/memory stabilization. |
| Grid-Delta Atlas + TOST | Required final evidence pair: all-cell/all-field atlas plus ADR-029 station TOST. | Implement/run after field divergence is no longer radical; publish compact plots under `docs/assets/v014/grid_delta_atlas/`. |

Durable release checklist: `.agent/decisions/V0140-RELEASE-CHECKLIST.md`.

Earlier context (kept for audit):
The release label is secondary to correctness. The current manager directive is:

1. **Find and fix the grid-cell divergence first, across all written WRF fields.**
   The three durable powered-TOST cases show broad cell-level wind disagreement:
   `proofs/v014/v10_grid_diagnostics.json` reports V10 grid RMSE above 1.5 m/s in
   3/3 cases, while station V10 is outside the tight ADR-029 margin in 1/3 cases.
   This means station TOST cannot be the next arbiter; the model must first be made
   WRF-close on the actual fields.
2. **FP32 acoustic is the next highest-value lane after grid divergence.** The
   completed FP32 de-risk reports make mixed perturbation-authoritative acoustic
   feasible in principle, but it touches the dycore and must not mask or compound
   the current cell-level divergence root cause.
3. **Other memory work follows FP32.** The major v0.13 memory fix is already landed:
   RRTMG column tiling is merged plus GPU-proven
   (`proofs/v013/rrtmg_column_tile_vram_suite.json`: LW untiled OOM, LW tiled
   5374.84 MiB; SW untiled 10033.1 MiB, SW tiled 1619.54 MiB). The v0.14
   empirical/static memory map is now complete and says no remaining
   non-radiation memory fix should block long validation after grid parity
   (`proofs/v014/empirical_memory_map.json`).
4. **TOST resumes only after grid-field divergence is minimized or explicitly
   root-caused.** Case 3 completed and the TOST marathon was stopped before Case 4;
   continuing n=15 statistics on known-divergent fields would waste GPU time.

GPU validation launch is now standardized through `scripts/run_gpu_lowprio.sh` and
`scripts/run_powered_tost_n15.sh`; the operational runbook is `docs/GPU_RUNBOOK.md`.
The Switzerland/Gotthard suite is explicitly not a v0.13 pass: case generation and CPU truth
exist, and the v0.12 128²/150² attempt is documented as fp64 OOM/grid-ceiling evidence
(`proofs/v0120/switzerland_128_gpu_result.json`); the post-memory-fix GPU-vs-CPU-WRF
Switzerland run is v0.14 B7.
The durable handoff for this goal shift is
`.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`.
Current v0.14 grid-parity work has already fixed and proven the stale static-metric writer
payload (`C1/C2/C3/C4`, `DN/DNW/RDN/RDNW`, and `MAPFAC_*` exact in the fresh h1 smoke),
classified the remaining base/static fields, packaged the h10 same-state savepoint request,
completed the writer-only `XLAT`/`XLONG` payload fix, achieved a green CPU-WRF h10
same-state marker at step 6000, emitted the first CPU-WRF source-derived dynamic layer
around final-stage `small_step_finish`, and found the green post-RK refresh target.
The marker proves the native h10/`d02` patch/index mapping and the correct
history-backed WRF `T` source (`grid%th_phy_m_t0`). The accepted WRF compare target is
now the state immediately after `dyn_em/solve_em.F::after_all_rk_steps` and before RK
halo exchanges: it is exact/roundoff against CPU h10 for `T/P/PB/U/V/W/PH/MU/MUB`.
The first JAX wrapper sprint proved the current public runtime exposes only post-halo /
post-guard state; the follow-up hook sprint added a default-off private pre-halo capture
path and proved normal RK returns are unchanged when disabled. The missing h10 JAX
pre-step `OperationalCarry` has now been produced at completed step 5999:
`/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`
(`OperationalCarry`, paired `OperationalNamelist`, grid `159 x 66 x 44`, SHA256
`0896e4a272cbeaa85d1bb969ecae82b047e75a028df45a87ddab4f4572af8dde`). The
canonical same-surface comparison now runs and returns `JAX_MISMATCH_T`: first
mismatch is `T` max_abs `3.3545763228707983`, RMSE `1.0296598586362888`, worst
native key `[12, 17]`. The T history/source-attribution sprint then closed the
source-mapping branch: no inspected JAX theta/history candidate matches WRF
history `T_HIST_SRC` or WRF `T_THM` within the frozen tolerance
(`proofs/v014/jax_t_history_source_attribution.json` verdict
`T_EVOLUTION_MISMATCH_CONFIRMED`). The theta-evolution localization sprint then
proved the mismatch is already present at the earliest currently available
input/reference theta boundary, before current-step physics or RK
(`proofs/v014/jax_theta_evolution_localization.json` verdict
`THETA_MISMATCH_PRESTEP_OR_INPUT`; `T_OLD` max_abs `6.218735851548047`, RMSE
`4.638818160588427`; `MU_OLD` max_abs `267.01919069732367`). The explicit
step-6000 pre-RK input-boundary sprint then emitted CPU-WRF truth and compared
it to the produced JAX h10 step-5999 carry. Verdict:
`PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`
(`proofs/v014/pre_rk_input_boundary.json`): all target fields already differ
before current-step physics/RK (`T` max_abs `6.218735851548047`, `P`
`589.6789731315657`, `PB` `1047.015625`, `MU` `267.01919069732367`, `MUB`
`1050.3046875`). The prestep carry source trace then ruled out checkpoint
serialization/load corruption and classified the producer path as
`PRODUCER_WRITES_BAD_FINAL_CARRY`
(`proofs/v014/prestep_carry_source_trace.json`): raw pickle runtime state,
checkpoint API runtime state, top-level State payload, and a `/tmp` round-trip
all preserve `T/P/PB/MU/MUB` exactly, so the bad values are already in the live
nested replay `OperationalCarry` passed to
`write_checkpoint(..., runtime_state=d02_carry)`. The previous-step handoff
bisection then classified the final partial subcycle as downstream, not causal:
`proofs/v014/previous_step_handoff_bisect.json` verdict
`BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`, with final producer-shaped replay exact
against the existing bad checkpoint and `PB/MUB` already wrong at d02 completed
step 5997 before parent step 2000, `_operational_force`, and child steps
5998-5999 (`MUB` max_abs `1050.3046875`, `PB` `1047.015625`). The earlier-source
bisection then found the first wrong surface: `proofs/v014/earlier_source_bisect.json`
verdict `BASE_STATE_SPLIT_DEFINITION_MISMATCH`. The initial d02 JAX
`OperationalCarry` matches native `wrfinput_d02` `PB/MUB`, but CPU-WRF
h0/h1/h10 and h10 pre-RK truth share a stable different `PB/MUB` split
(`MUB` max_abs `1050.3046875`, `PB` `1047.015625`), so replay-time drift is not
needed to explain the bad h10 base carry. The active next sprint is a narrow
source-changing fix for
`src/gpuwrf/integration/d02_replay.py::build_replay_case` native child
base-state split construction. TOST, Switzerland, FP32, and broad memory work
remain paused until this base-state split bug is fixed or explicitly bounded.
The first source-fix attempt then correctly blocked rather than patching:
`proofs/v014/base_state_split_fix.json` verdict
`BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`. CPU-WRF h0
`PB/MUB` are base-formula values on post-nest blended h0 terrain (formula
residuals below `0.06` Pa), but the missing production input is WRF's
parent-interpolated/blended `HGT/MUB/PHB` surface from `med_interp_domain` /
`blend_terrain` before `start_domain_em` recomputes `PB/MUB/PHB/theta_base`.
Simplified bilinear+blend remains hundreds of Pa off. The active next sprint is
a WRF live-nest base hook/oracle sprint; do not patch `PB/MUB` alone and do not
use CPU-WRF `wrfout_h0` as production input.

The project completed
the 2026-05-28 reset (M8–M23 roadmap in `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md`), rebuilt
the dycore honestly, and advanced through the incremental version chain. The current system is a
**standalone, JAX-native, single-GPU WRF-compatible v4 ARW forecast system for standard regional
configurations** on one RTX 5090: **native real-init** (`wrfinput`/`wrfbdy` assembled from
met_em-stage forcing, no `real.exe`, no CPU-WRF artifact for IC/LBC — proven equivalent to
`real.exe` at t=0), a **GPU-operational physics menu** with a **fail-closed boundary** on
everything not yet ported, a **WRF-compatible namelist**, live multi-domain nesting, restart,
conservation budgets, and a standalone out-of-box CLI. **It is a WRF-compatible reimplementation
(not a Fortran-source port) and a transparent research artifact (not a full WRF replacement).**
The canonical user-facing scope statement is the top of [`README.md`](README.md).

- **Version chain:** v0.1.0 replay → v0.3.0 native metgrid → v0.4.0 native real-init → v0.6.0
  expanded physics menu → v0.9.0 standalone consolidation → v0.10.0 Thompson-sedimentation →
  v0.11.0 live nesting/restart/conservation/MYNN-EDMF/topo-slope-radiation/slope-diffusion →
  v0.12.0 standalone out-of-box CLI + persistent JIT cache + fail-closed catalog + PSFC fix +
  equivalence demo → **v0.13.0 "Validate & Accelerate"** (RRTMG VRAM-floor chunking plus
  RRTMG column tiling, GWD-on-nested default-on, compile-speed re-land GPU-validated,
  MYJ+Janjic operational, WDM5/MRF/GFS-sfclay/old-MM5/GSFC-SW coverage expansion, moisture
  flux-advection into RK3, multi-GPU fake-mesh sharding, clear-sky diagnostics, RRTM-LW
  skeptic-hardening, outsider-reproducibility + community-validation). Closeouts in
  [`.agent/decisions/`](.agent/decisions/); v0.13 roadmap = `.agent/decisions/V0130-ROADMAP.md`.
- **GPU-operational menu (scan-wired, WRF-oracle-gated):** MP {1 Kessler, 2 Lin, 3 WSM3, 4 WSM5,
  6 WSM6, 8 Thompson, 10 Morrison, **14 WDM5 (v0.13.0)**, 16 WDM6}; PBL {1 YSU,
  **2 MYJ (v0.13.0)**, 5 MYNN, 7 ACM2, 8 BouLac, **99 MRF (v0.13.0)**}; SFCLAY
  {1 revised-MM5, **2 Janjic-Eta (v0.13.0)**, **3 GFS (v0.13.0)**, 5 MYNN-SL,
  7 Pleim-Xiu, **91 old-MM5 (v0.13.0)**}; CU {1 KF, 2 BMJ, 3 Grell-Freitas, 6 Tiedtke
  operational when active flux-form moisture advection provides WRF-style RQVFTEN}; RRTMG SW+LW + Dudhia-SW/RRTM-LW +
  **GSFC-SW (v0.13.0)** + clear-sky diagnostics; GWD `gwd_opt=1` (v0.13.0 default-on nested);
  LSM {2 Noah-classic, 4 Noah-MP}.
  Source of truth: [`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py)
  + `_SCAN_WIRED_OPTIONS` in [`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py).
- **Parity-proven but fail-closed (recognized, loudly rejected):** New-Tiedtke (`cu=16`).
  (v0.13.0 promoted MYJ + Janjic to operational; Dudhia SW + RRTM LW were wired in v0.12.0.)
- **Current blockers and honest qualifiers:** **the cell-level CPU-WRF vs GPU-WRF field
  envelope is the blocking credibility gate.** The 24 h forecast-skill closure
  (T2/U10/V10) vs CPU-WRF is NOT closed (KI-9), and the equivalence demo's 24 h d02
  verdict is `NOT_EQUIVALENT`, dominated by lead-time wind divergence. The powered n=15
  TOST scoring path is unblocked (rc=2 fixed), 3/15 cases are durable, and `--resume`
  is available, but TOST is intentionally paused until the grid-cell divergence is
  explained or minimized. n=15 remains underpowered vs the n≈27 target. Multi-GPU
  throughput is fake-mesh-only (real throughput UNMEASURED → per-watt / whole-Earth
  claims stay PROJECTED). Moisture flux-advection + clear-sky diagnostics are opt-in
  (default-off, byte-identical when off).
- **Next (v0.14+ roadmap):** the path to 1.0.0 now starts with **Grid-cell parity
  and divergence attribution**: compare all written wrfout fields cell-by-cell, lead-by-lead,
  and region-by-region against CPU-WRF, then fix the responsible dycore/surface/coupling
  operators. After that: FP32/mixed-precision acoustic reformulation per
  `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`, remaining memory work, then powered
  TOST as a final gate. The scheme long-tail remains fail-closed unless oracle-proven.
  See
  [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), `.agent/decisions/V0130-ROADMAP.md`,
  [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md), and the full-port gap analysis
  under [`.agent/reviews/`](.agent/reviews/).

The sections below are the **historical M0–M7 synthesis layer** (the original backend-bakeoff and
M6 dycore-blocker planning). They are retained for provenance and are superseded by the reset
roadmap and the v0.1.0 status above. Author: manager (Opus 4.7 1M).
Inputs: `v2 ai driven from scratch plan by deepthink.txt`, `wrf to gpu gpt5.5 deep research.pdf`, `WRF GPU Porting_ Architecture & Verification.pdf`, all governance files in this repo.
Scope: this document is the **synthesis layer** that reconciles the two research briefs with the bootstrap. It does not restate the constitution, scope, spec, principles, validation strategy, precision policy, performance targets, or risk register. Those remain authoritative as-is.

## 1. Why this plan exists

The bootstrap (M0) gave us a governance system and nine-milestone skeleton. It deliberately did not pick a stack, a verification framework, or a sprint sequence. The two research briefs disagree on the most strategic question — the execution backend — and the disagreement is sharp enough that a bad call here would cost us a year. This plan freezes the things both briefs agree on, refuses to freeze the things they disagree on, and routes the disagreement into M2 as an evidence-driven decision.

## 2. What both briefs agree on (freeze now)

These are accepted as project ground truth. No further debate without ADR.

1. **No incremental OpenACC port of legacy WRF.** Both briefs are unambiguous: the architectural ceiling is ~5–7× and our target is ≥8–10×, so the path is structurally barred. Already encoded in `PROJECT_CONSTITUTION.md` (immutable) and `docs/research-inputs.md`.
2. **Whole-state device residency** for the high-frequency model state. No host/device transfers in the timestep loop without an explicit ADR. Already in `PERFORMANCE_TARGETS.md` and the constitution.
3. **Validation is non-bitwise by default.** A four-tier pyramid: (1) micro fixture parity, (2) physical invariants, (3) short-run / timestep convergence, (4) probabilistic ensemble consistency. The two briefs propose different vocabularies (PyCECT vs. probtest, Serialbox vs. icon-exclaim savepoints, TSC) — these are **all complementary** and should be combined, not chosen between. Already encoded in `VALIDATION_STRATEGY.md`; this plan operationalizes it in §6.
4. **MPI rank per GPU with GPU-aware halo exchange** is the only honest path to multi-GPU. Single-GPU comes first per `PROJECT_SCOPE.md`, but the halo interface must be designed not to preclude this.
5. **Profiler artifacts are mandatory** for every performance claim. Schema already in `PERFORMANCE_TARGETS.md`.
6. **One bounded operational target first**: Canary Islands 3 km, then 1 km. Single domain, single physics path, no nesting in v0. Already in scope.
7. **Vertical kernel fusion** at timestep granularity, not micro-kernel orgy. Both briefs cite the launch-bound failure mode of every legacy port.
8. **Explicit precision policy** with FP64 reference for mass continuity and pressure gradients; FP32/BF16 only where validated per-variable per-scheme. Already in `PRECISION_POLICY.md`.

## 3. What they disagree on (do NOT freeze — route to M2)

The deepthink brief recommends **Python + JAX/XLA + Triton** as the AI-native stack. It argues opaque C++ template errors hallucinate AI agents and DSLs lack LLM training data.

The GPT-5.5 brief recommends **Kokkos/C++** (with optional GT4Py-style DSL for stencil-heavy dycore parts) and frames CUDA Fortran as a "tactical NVIDIA-only shortcut, Plan B." It argues all successful next-gen NWP rewrites (SCREAM, HOMMEXX, ICON-exclaim, Pace) are Kokkos- or DSL-based, not JAX.

**Both arguments have weight.** Neither has decisive evidence for *our* specific constraints (single-node RTX 5090, agent-driven authorship, Python-friendly ML coupling, but also long-term portability). The constitution and `ARCHITECTURE_PRINCIPLES.md` already mandate an M2 bakeoff. This plan **enforces** that and adds the bakeoff design (see §5).

Other disputed items:
- **AI swarm topology.** Deepthink proposes α/β/γ adversarial agents. The bootstrap already has a richer cast (manager, gpt-kernel-worker, sonnet-test-engineer, opus-reviewer, profiler-bot, research-scout, human-arbiter). The bootstrap topology stays.
- **Hard speedup target.** Deepthink writes "8×–12×". GPT-5.5 cites MeteoSwiss ICON at 5.5× socket-to-socket as a real production result and is more cautious. This plan adopts the target structure already in `PROJECT_SPEC.md` ("beats the existing CPU operational baseline enough to justify the rewrite path") and refuses to pre-commit a number before the M2 bakeoff produces evidence.
- **ML emulator hybridization.** Deepthink Phase 5. Out of v0 scope per `PROJECT_SCOPE.md`. Kept as research note in `docs/research-inputs.md` only.

## 4. Strategic shape of the project

```
                 M0 ─── M1 ─── M2 ─── M3 ─── M4 ─── M5 ─── M6 ─── M7 ─── M8
              bootstrap fixt. bakeoff state  dycore phys.  couple Canary release
                 │      │      │      │      │      │      │      │      │
                 │      │      └─── ADR-001 backend lock
                 │      │                    └─── ADR-002 state layout
                 │      │                                  └─── ADR-003 dycore precision
                 │      └─── fixtures unblock M2 and every later milestone
                 └─── governance unblocks everything
```

Two things are critical and not yet emphasized enough in the bootstrap text:

1. **M1 must produce *enough* of an oracle to support M2.** A schema and one toy fixture is not enough. M2 needs at least one shared 3D stencil fixture *and* one column-physics fixture so candidate backends can be compared on both regimes. This plan tightens M1's exit criteria in §7.
2. **M2 must include agent-success rate as a first-class metric.** Both briefs argue (from opposite sides) that author productivity matters. The bakeoff measures correctness, profiler artifacts, *and* how many sprints / how many rejection cycles each candidate cost. This plan adds agent-success rate to the M2 acceptance gate in §7.

## 5. Backend bakeoff design (binds M2)

**Bakeoff candidates** (the same two kernels implemented in each, with the same fixture). The set is anchored on the candidate families already named in `ARCHITECTURE_PRINCIPLES.md` and `MILESTONES.md § M2`:

- **A:** JAX / XLA (Python). Pure functional. AOT compile via `jit`.
- **B:** OpenAI Triton (Python). For the column-physics kernel, exercises the register-pressure escape hatch deepthink emphasizes.
- **C:** GT4Py + DaCe (Python). Stencil DSL.
- **D:** Kokkos / C++. The GPT-5.5 recommended path.
- **E:** CuPy raw CUDA or Numba CUDA (Python). Low-overhead Python escape hatch for context.
- **F:** Explicit CUDA C++ ("CUDA Tile" path per `ARCHITECTURE_PRINCIPLES.md:3-4` and `PERFORMANCE_TARGETS.md`'s `cuda-tile` enum). Lower-level than D; included so the bakeoff covers the "manual tile in shared memory" paradigm independent of Kokkos abstraction cost.

**CUDA Fortran** is **not** in the v0 bakeoff *unless* a research-scout sprint produces a concrete NVIDIA-only Canary-v0 justification before M2 dispatch. The plan does not declare it disproven; it declares the v0 bakeoff bandwidth bounded at six candidates and routes the question to the human arbiter (see §11) per `.agent/rules/architecture-decision-policy.md`. This avoids pre-judging the GPT-5.5 brief's Plan B without evidence.

**Shared bakeoff problems** (defined by M1 fixtures):
- **Problem 1 (stencil):** 3D advection-diffusion update on a staggered grid, 4th-order horizontal, 2nd-order vertical, with halo exchange placeholder. Tests fusion ergonomics and memory layout.
- **Problem 2 (column):** A condensed column-physics workload exercising many local prognostics, deep conditional branching, and no horizontal coupling. Tests register pressure and SRAM tiling. We do **not** port any real WRF physics scheme in M2 — too large. We synthesize a representative analog with the same data-flow shape and branching depth as schemes like Thompson microphysics or MYNN PBL, but the analog is **not** a commitment to those schemes as the first physics suite (the M5 decision gate selects the actual operational first suite — see §8).

**Per-candidate proof object:**
- Correctness comparison vs. fixture, in the schema of `PERFORMANCE_TARGETS.md`.
- `ncu`/`nsys` JSON with: kernel launches, occupancy, registers/thread, local-memory bytes, achieved bandwidth, host/device transfer count.
- Maintainability narrative (≤300 words): build complexity, error legibility, debugger story.
- Agent-success log: how many sprints / how many reviewer rejections / how many escalation events the implementation cost.
- If the candidate fails to build, compile, or run on the target hardware: a **candidate-failure artifact** with build log, blocker category (toolchain, license, hardware, agent-success), and reviewer decision. A failed candidate is an evidence outcome, not a bakeoff deadlock — ADR-001 may still proceed with documented coverage gaps.

**ADR-001 (backend lock) requires:**
- Profiler JSON or candidate-failure artifact for both problems in all six candidates.
- Reviewer cross-model challenge per `.agent/rules/cross-model-review-policy.md`.
- Explicit human approval per `.agent/rules/architecture-decision-policy.md` (irreversible).

## 6. Validation architecture (binds M1–M6)

The four-tier pyramid in `VALIDATION_STRATEGY.md` is correct but underspecified for *tooling families*. This plan binds the **family** for each tier and the acceptance behavior, not specific named tools. Concrete tools are chosen by ADR after a research-scout sprint with implementation evidence — they may turn out to be the named candidates below or alternatives.

| Tier | Purpose | Tooling family / acceptance behavior | Candidate tools (non-binding) | First binding milestone |
|---|---|---|---|---|
| 1 | Micro fixture parity | Pre-/post-subroutine savepoints from a trusted CPU baseline; Python comparison harness; per-variable / per-tier tolerances with documented rationale | Serialbox, hand-rolled NetCDF savepoints | M1 (schema + first fixture); M2 (consumed) |
| 2 | Physical invariants | Mandatory: mass-conservation residual, tracer positivity, bounds, NaN/Inf. Optional / scenario-specific: KE spectrum slope ($k^{-5/3}$ check), water/moisture budgets, momentum budgets. | NumPy / project-owned Python | M4 (mandatory subset); M6 (optional scenario-specific) |
| 3 | Short-run / timestep convergence | GPU drift vs. CPU-baseline timestep-sensitivity envelope on a reduced/idealized case where the envelope is computable | TSC1.0 method | M4 → M6 |
| 4 | Probabilistic ensemble consistency | Either a PyCECT-style PCA projection from a perturbed CPU ensemble *or* a probtest-style per-variable tolerance derivation, or both — selection deferred to the M6 ADR, which also sets ensemble size based on actual storage/runtime budget (see risk in §10) | PyCECT, probtest, custom | M6 (small-ensemble prototype) → M7 (production ensemble after budget approval) |

Additional cross-cutting validators required from M3 onward:
- **Transfer audit** (CUPTI or backend equivalent) on every coupled-loop test run.
- **Backend-specific fault-isolation mode** allowing bit-equal compare on small cases for debugging. The exact mechanism — NVHPC `-Mnofma` / `--gpu=math_uniform` (cited by the GPT-5.5 brief for the MPAS/NVHPC stack), XLA determinism flags, Triton-level controls, or equivalent — is **documented in ADR-001 / ADR-002 for the chosen backend**. Never a production target. Not assumed universal across candidates.
- **Compute Sanitizer** (memcheck/racecheck/initcheck/synccheck) on every kernel ADR.

## 7. Proposed stricter milestone gates (pending patch into per-milestone files)

These are **proposals**. The constitution and `MILESTONES.md` remain authoritative. After human approval of this plan, the manager patches the per-milestone files via the normal repository process — this plan does not silently override them.

- **M1 exit (proposed tightening):** At minimum one analytic stencil fixture *and* one analytic column-physics fixture *and* one Canary WRF-derived fixture, plus the manifest schema, plus the storage policy, plus the comparison harness command. **All five are required to close M1.** There is no placeholder-data bypass — if the Canary fixture is missing, M1 remains open. (See §10 for the parallel-work alternative under human approval.)
- **M2 exit (proposed tightening):** Each of the six backend candidates produces either profiler JSON for both bakeoff problems *or* a candidate-failure artifact in the schema of §5. ADR-001 includes a maintainability narrative *and* an agent-success log per candidate. Human approval required.
- **M3 exit (proposed tightening):** Transfer audit shows **zero** host/device transfers inside a 1000-step dummy timestep loop. Not "low," **zero**. The single allowed exception (output) must be cited in ADR-002. `GridSpec` includes named, machine-readable fields for: map projection, terrain/geog static-field provenance, vertical coordinate metadata, halo width, and boundary-condition metadata (deferred from M7).
- **M4 exit (proposed tightening):** Tiers 1 + 2 (mandatory subset) + 3 must all pass for the reduced dycore on at least one idealized case (`em_hill2d_x` or equivalent). Tier 3 envelope is computed from the CPU baseline, not asserted by hand.
- **M5 exit (proposed addition):** Includes an M5-S0 decision-gate sprint that selects the *first physics suite* from the Canary operational target stack, with a recorded rationale. The M2 column-physics analog is **not** by itself a commitment to any specific WRF scheme.
- **M6 exit (proposed addition):** Includes the Tier-4 small-ensemble prototype; full ensemble deferred to M7. Includes a separate METplus-or-equivalent verification-tooling research-scout sprint that completes *before* M7 dispatch — see §10 risk on "METplus learning curve."
- **M7 exit (proposed addition):** Falsifiable I/O / restart compatibility — minimal `wrfinput` / `wrfbdy` / `wrfout` / `wrfrst` compatibility matrix (or an explicit deviation document) and a restart-continuity test. Adds an IC/BC mapping proof object covering source datasets, update cadence, boundary-field variables, interpolation policy, and restart interaction. Adds a surface/land/SST/static-geog proof object (earliest emergence may be M5 or M6 depending on first-suite choice).

## 8. Sprint sequence (v0 horizon)

This is the recommended ordering, not a frozen Gantt. Re-evaluated after each milestone closeout.

```
M1   ──►  S1: fixture-storage-policy + manifest-schema + extraction harness skeleton
M1   ──►  S2: analytic stencil micro fixture + comparison harness CLI
M1   ──►  S3: analytic column micro fixture + WRF variable-map seed
M1   ──►  S4: first Canary WRF-derived fixture (single timestep slice, single column subset)
[ M1 closeout + Codex review of fixtures → green-light M2 ]
M2   ──►  S5..S9: five-way backend bakeoff, one sprint per candidate, frozen interface
[ M2 ADR-001 → human approval → backend lock ]
M3   ──►  S10: GridSpec + State contract draft + ADR-002 state layout
M3   ──►  S11: device-resident state implementation + dummy loop + transfer audit
M4   ──►  S12..S14: advection, pressure/acoustic, RK coupling on reduced dycore
M5   ──►  S15..S17: microphysics subset, then PBL subset, then radiation decision
M6   ──►  S18..S19: coupling + drift envelope + ensemble prototype
M7   ──►  S20..S23: 3 km pipeline, then 1 km memory audit, then post-processing, then ops verification
M8   ──►  S24..S26: docs hygiene, packaging, release validation
```

The first sprint to execute is **S1**. Its contract lives at `.agent/sprints/<date>-m1-fixture-storage-policy/sprint-contract.md` and is the deliverable accompanying this plan.

## 9. Operating rules for the manager (delta on top of existing roles)

- Spawn every subordinate agent (Claude worker, Codex review, profiler runs) in a **named tmux window inside the user's session**. Close the window once deliverables are captured to the sprint folder. Already memoryized.
- Never dispatch a worker without: (a) a sprint contract on disk, (b) reviewer pre-approval when lifecycle requires it, (c) explicit file-ownership statement to avoid parallel edits to shared core files.
- Status reports to the user follow `docs/user-status-report-format.md`: current milestone, what changed, proof objects, blocked decisions, next sprint recommendation.

## 10. Risks added on top of `RISK_REGISTER.md`

(These extend, not replace, the official register. Items the manager believes should be patched into `RISK_REGISTER.md` are marked **[patch]**.)

| Risk | Impact | Mitigation |
|---|---|---|
| M2 bakeoff degenerates into a JAX-vs-Kokkos religious war | Project paralysis, ADR-001 escalation deadlock | Maintainability narrative + agent-success log are first-class metrics, not opinions. Cross-model review per `.agent/rules/cross-model-review-policy.md`. Human-arbiter tie-break. Candidate-failure artifact schema (§5) avoids "all-or-nothing" deadlock |
| M1 fixtures take the entire project budget (deepthink's "Phase 0: Weeks 1-3" is optimistic) | M2 starvation; bakeoff blind | Hard cap of 10 sprints applies to **analytic** fixture work only. The Canary fixture is mandatory for M1 closure. If schedule pressure requires earlier M2 motion, the manager may authorize an M2 *read-only scout sprint* (research-only, no implementation files touched) running in parallel with the remaining Canary fixture work. No M2 implementation sprint dispatches until M1 closes |
| Local RTX 5090 is unavailable or under-provisioned during bakeoff | M2 cannot collect profiler artifacts | Mitigation deferred: smoke-test on smaller GPU first, escalate to cloud H100 only if profiler artifacts are blocked. **[patch]** add to RISK_REGISTER as "RTX 5090 toolchain maturity for all bakeoff candidates" — JAX/Triton/Kokkos/GT4Py compatibility on Blackwell is not guaranteed at M2 entry |
| WRF baseline access for fixture extraction unresolved | M1 S4 blocked | Manager raises this as human decision before S1 starts (see §11) |
| **[patch]** IC/BC dataset availability and licensing for Canary | M7 blocked, possibly unrecoverable | Add explicit M3 ADR proof object naming the source (GFS / ERA5 / ECMWF AIFS / other), licensing terms, and refresh cadence. Manager raises as decision before M3 dispatch |
| **[patch]** Terrain / geog / static-field correctness | Physics on a wrong topography invalidates all Canary v0 evidence | Add M3 proof object: provenance file, projection, transform, checksum, and a sanity check (max elevation, coastline alignment) |
| **[patch]** Observation source for METplus-style verification | M7 verification not falsifiable | Research-scout sprint at the end of M6 (codex dissent on METplus timing). Identifies station network, satellite/radar feeds, license terms, and storage |
| **[patch]** Full-ensemble runtime and storage cost (100-member CPU PyCECT) | M7 ensemble can blow up storage budget, blocks the operational target | Tier 4 prototype at M6 with a *small* ensemble (e.g. 10 members) establishes the per-member cost. Full ensemble at M7 only after the cost model is approved by the human arbiter |
| **[patch]** S1→S2 ordering and parallel work | S2 implementing fixtures against a schema that is still drafting | S2 implementation does not dispatch until S1 schema review passes. A read-only S2 scout (no edits to fixture files) may run in parallel after S1 freezes the schema field list |

## 11. Operational decisions (manager-owned, recorded here)

Per the manager-autonomy directive of 2026-05-19, the user delegates all operational, design, fixture-source, tooling, sprint-sequencing, and architecture-bakeoff decisions to the manager. The user is consulted only at milestone closure or on genuine blockers. The seven items below were operational, not constitutional, and are now manager decisions recorded for audit. For ADRs that the constitution still requires human approval on (backend lock at M2, state layout at M3, public release at M8), the manager runs a Codex cross-model critical review and records the call here in lieu of pre-asking the user.

### Recorded manager decisions (2026-05-19)

1. **WRF baseline source — DECIDED.** WRF v4.7.1 binary at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` (sha `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`). Build environment: `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`. Compiler: NVIDIA HPC SDK `nvfortran` 26.3. **Source of truth for Tier-1 fixture extraction.**

2. **Fixture storage location — DECIDED.** `/mnt/data/wrf_gpu2/` (1.1 TB free as of 2026-05-19), symlinked at `./data/` in repo. Subdirectories: `data/fixtures/` (binary fixture payloads), `data/runs/` (large WRF or candidate outputs), `data/profiler_artifacts/` (ncu/nsys binary dumps), `data/cache/` (working scratch). `data` is gitignored; only manifests in `fixtures/manifests/` are tracked in git.

3. **CUDA Fortran in M2 — DECIDED: excluded from v0 bakeoff.** No scout sprint scheduled unless M2 candidates A–F all underperform or fail. The previous attempt's `../wrf_gpu/` README documents the CUDA-Fortran-with-tile-resident-restructure path as the *fallback* architecture if Kokkos/JAX/Triton/CUDA-Tile/GT4Py all fail M2. Honored as fallback, not promoted to v0 candidate.

4. **CUDA Tile / explicit CUDA C++ as M2 candidate F — DECIDED: included.** Already reflected in `PROJECT_PLAN.md §5` and `ROADMAP.md M2`.

5. **Analytic-fixture sprint cap — DECIDED: 10 sprints.** Canary WRF fixture still mandatory for M1 closure. See §10.

6. **IC/BC source dataset — DECIDED: AIFS** for M3+ production path (already adopted by Gen2 — see `/home/enric/src/canairy_meteo/Gen2/README.md` §2 "AIFS core"). For M1 fixture work where IC/BC don't bind decisions: reuse existing Gen2 WRF runs from `/mnt/data/canairy_meteo/runs/` rather than triggering new CPU WRF jobs. The S4 (Canary fixture) sprint prefers slicing from a pre-existing Gen2 run over launching a new run.

7. **M5-S0 first-physics-suite decision-gate sprint — DECIDED: approved** as a distinct sprint. Per Gen2's operational stack (Thompson microphysics, MYNN-EDMF PBL, Noah-MP land, RRTMG radiation), the leading candidate for "first physics suite" is microphysics-first (Thompson). The decision-gate sprint will confirm or revise based on M2 backend ergonomics.

### Constitutional gates that remain (manager-exercised, with Codex critical review, reported to user post-decision)

- **ADR-001 (M2 backend lock).** Manager writes; Codex cross-model review; manager finalizes; reports to user at M2 closeout.
- **ADR-002 (M3 state layout).** Same pattern.
- **ADR-003 (M4 dycore precision).** Same pattern.
- **Public release naming/legal (M8).** Reported to user before public posting (the user is the publisher of record).

## 12. What this plan does not do

- Does not pick a backend. M2 does.
- Does not freeze interface contracts. `INTERFACE_CONTRACTS.md` is explicitly placeholders until M3 ADR-002.
- Does not promise a speedup number. `PROJECT_SPEC.md` already states the bar correctly ("beats the operational baseline enough to justify the rewrite").
- Does not change governance, validation strategy, precision policy, performance targets, or scope. Those are constitutional.

## 13. Recorded operational decisions since manager handover 2026-05-23

The manager handover on 2026-05-23 triggered an M6.x dycore pivot cycle. These are operational records for a fresh reader; they do not promote any PROPOSED ADR to ACCEPTED and do not alter milestone gates.

1. **ADR-023 conservative column solver is the current proposed architecture.** It was chosen after a three-way critic, source scout, prototype, MPAS column-slice work, and diagnostic follow-up. The public scan path is unified on main, but ADR-023 remains **PROPOSED** because warm-bubble operator sanity still exposes a mass-coupling/stabilization issue and reviewer concurrence is still required.
2. **ADR-024 warm-bubble gate policy is PROPOSED.** The old `[5, 10] m/s` warm-bubble amplitude target is no longer a binding architecture gate. The critic verdict was `CHANGE-THE-GATE`: warm-bubble now reports operator sanity, finite/bounds failures, and anti-clamp evidence; M6 close remains Tier-3 convergence plus initial Tier-4 RMSE/consistency.
3. **ADR-021 carry-expansion prototype is branch evidence only, not merged architecture.** The apparent warm-bubble pass depended on target-shaped clamps and harness aids. The later clamp-strip honest test failed catastrophically (`FAIL_FINITENESS` at step 2), so ADR-021 is not a clean fallback without a new sourced stabilization plan.
4. **The HYBRID close plan is in execution.** The accepted sequencing from the close-strategy critic is: S1 diagnostic/source lock, S2 current-path d02 baseline, S3 source-backed mu/metric cleanup, S4 Tier-3 controlled convergence, S5 6h/24h Tier-4 comparison, S6 closeout or explicit architecture blocker. Current state: S1 done; S2 and S2.1 produced only synthetic fallback because real replay hangs; S2.2 is in flight to fix that; S3-narrow is done; S4-prep is running infrastructure work; S3-real/S4/S5/S6 remain queued.
5. **Source-mining is locked as a decision aid, not an ADR.** `.agent/decisions/source_mining_operator_table.md` is the canonical table for the current operator debts. S3-narrow converted the production `1.35` metric and `0.38` buoyancy-scale usage into source-backed or slice-only paths and improved the stabilizer scan from 28 to 20 experiment-backed findings and 8 to 37 source-backed findings. The `_mu_continuity_increment` limiter remains explicitly deferred until real d02 baseline evidence exists.
6. **Initial Tier-4 numerical anchors now exist.** `data/fixtures/gen2_baseline/rmse_summary.csv` records 17 same-grid Gen2 forecast-to-forecast d02 pairs: T2/U10/V10 spatial-mean RMSEs of 0.628 K, 1.456 m/s, and 1.591 m/s at 24 h; 0.255 K, 0.888 m/s, and 0.870 m/s at 72 h. These are Gen2 consistency anchors, not observation-error closure.

## 14. M6 Post-Blocker Execution Plan (2026-05-24)

Triggered by the catastrophic S2.1-redo real d02 baseline, the `NO-BUG-LOCALIZED` bug-hunt verdict, the HYBRID exit-rule firing, and the external deep-consultation response committed as `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md`. Supersedes the prior HYBRID S3-real → S4 → S5 → S6 sequence and the BLOCKER-memo manager addendum (A-as-probe-first).

### 14.1 Diagnosis (refined)

The dycore is not failing because GPU-JAX cannot run WRF-like dynamics. It is failing because the project lacks instrumentation to distinguish *which* class of error is firing: wrong recurrence, wrong staging, missing WRF scratch state, or wrong source-equation coupling. ADR-023's minimalist-carry thesis failed real d02; ADR-021's clamp-strip failure proves that a *blind* full-carry port also fails. Both architectures were guessing with different numbers of variables. The correction: **WRF compatibility must be validated at the acoustic-substep level, not at 1h.**

### 14.2 Committed sequencing (B-direct, savepoint-first)

```
M6B0 ── M6B1 ── M6B2 ── M6B3 ── M6B4 ── M6B5 ── M6B6 ── M6b ── M6c ── M7
savepoint  coef   tridiag  scratch  acoustic  full    coupled  1h    24h
harness   parity  parity   parity   parity   dycore   step    honest Gen2
                                                                      └─ public release (M8)
                                                                          ↑
                                                              optional E-lane:
                                                              GPU-WRF shadow scout
```

- **M6B0 — WRF savepoint harness.** *Decisive sprint.* CPU WRF instrumentation around `module_small_step_em`, savepoint schema (RK stage, acoustic substep, stagger, units, map factors, vertical grid, namelist, run-ID), JAX comparator with deliberate-perturbation negative test, first coefficient parity proof on one column + 16×16 patch. **No RMSE tuning, no clamps, no sanitizer acceptance.**
- **M6B1 — coefficient parity.** Reproduce WRF vertical-solve coefficients (`calc_coef_w` equivalent) on one column → 16×16 patch → full d02.
- **M6B2 — tridiagonal solve parity.** JAX `lax.scan` Thomas solver matched against WRF.
- **M6B3 — scratch-state parity.** Adopt `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, save fields as savepoints demand. WRF compatibility now wins over carry-minimalism.
- **M6B4 — acoustic recurrence parity.** One acoustic substep then all substeps inside one RK stage.
- **M6B5 — full dycore step parity.** Physics off, boundary off, sanitizer off.
- **M6B6 — coupled step parity.** Physics on, boundary on, sanitizer still off.
- **M6b — honest 1h.** Only after B6 do we re-run the 1h d02 RMSE comparison.
- **M6c — 24h Gen2 consistency.** Tier-4 statistical envelope (AceCAST-style: GPU-vs-CPU bounded by CPU-vs-CPU floating-point divergence) rather than bitwise.

### 14.3 What A-probe and Option C become

**A-probe (WRF scratch hybrid)** is **not** scheduled. The consultation's argument carries: partial-scratch may improve T2 from 137 K to 40 K and then trap the project in months of "hidden staging" whack-a-mole. The savepoint harness gives surgical attribution that A-probe cannot. If a future evidence-driven justification surfaces, A-probe may run as **a single disposable sprint with a hard kill gate** (first-nonfinite → none over 10 steps AND ≥10× T2 RMSE drop; failure means immediate B-direct continuation, never a follow-on tuning sprint).

**Option C (substrate port)** stays as fallback only: **"C-primary = JAX reimplementation of ICON4Py/MPAS/WRF-proven vertical-implicit patterns, after B proves WRF-small-step parity is too expensive or structurally unsuitable."** Not before.

### 14.4 Option E (new) — Shadow GPU-WRF lane (optional, low-priority)

A research-scout sprint may evaluate AceCAST and `FahrenheitResearch/wrf-gpu-port` as **shadow benchmarks** — operational counterfactuals that protect Canary business continuity while the JAX rewrite is held to savepoint truth. E is **not** a replacement for B. E is a parallel insurance lane; it does not consume the primary 4-core AI budget and may be deferred indefinitely.

### 14.5 Performance gating (post-correctness only)

**Principal directive 2026-05-24 (night):** *"Solutions that just bring the correct results but are massively inefficient for the GPU will bomb the project purpose by design. Pilots are OK if they can be optimized, but if incompatible with a GPU-optimized core, the solution is just wrong for this project."* This section enforces that as binding plan rule.

No optimization sprints before M6B5 passes. Then in order:
1. **Lock RTX 5090/JAX environment.** Nsight Systems trace proving no recompilation in timestep loop. `jax_enable_x64=True`. CUDA-pip wheel pinned.
2. **Whole timestep as one compiled graph.** Acoustic loop and RK loop are `lax.scan`, never Python loops. Hard rule: no `device_get`, no host callbacks, no Python diagnostics in the operational timestep loop. Diagnostics in a separate debug build.
3. **Carry size is a performance problem, not a correctness veto.** A correct large carry beats a sanitizer-dependent small carry. Use XLA buffer donation and aliasing.
4. **Profile before Triton.** Per-scheme Triton/Pallas only for measured hotspots. Aligned with ADR-001's gated fallback.
5. **Two precision modes**: validation (fp64-heavy, strict WRF parity) and operational (mixed where savepoint and Tier-4 prove safe).

### 14.5.1 Validation-mode / operational-mode separation (BINDING INVARIANTS)

**The savepoint harness is a validation-mode-only checkpoint. The operational mode is a strict GPU-optimized variant that runs in production. Conflating the two will cap max speed by design.**

| Invariant | Validation mode | Operational mode |
|---|---|---|
| Precision | fp64 strict per WRF | Per-field per ADR-007 (fp32/bf16 authorized for fields where Tier-4 proves safe) |
| State carry | Includes WRF small-step scratch (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `_save`) for bitwise comparison | Strict subset: drops any scratch field not required by Tier-4 RMSE |
| Operator boundaries | 12+ savepoint-emission points inside `module_small_step_em.F` analogues | Operators may be **fused across savepoint boundaries** into single XLA HLO graphs / `lax.scan` bodies |
| Vertical solver | Serial Thomas forward-backward (per WRF) | Thomas OR parallel cyclic reduction (PCR) OR batched-Thomas, chosen by profiler + Tier-4 envelope |
| Halo exchange | Per-WRF cadence | Per-GPU-optimal cadence — may differ if Tier-4 passes |
| H2D/D2H in timestep loop | Permitted for savepoint emission only (validation builds only) | **ZERO**. Hard constitutional rule. |
| Diagnostic snapshots | Mandatory per `[[feedback_debuggability_hooks]]` static-arg pattern | DCE-eliminated by XLA in production build |

**Binding rules that follow from these invariants:**

- A sprint that adds a field to **operational** carry must include a justification "this is required by Tier-4 envelope" with cited evidence. If no such justification, the field lives in validation mode only.
- A sprint that introduces a synchronization point inside the operational timestep loop (that is not present in WRF for numerical reasons) must include an "operational-mode ablation" sub-sprint that demonstrates the synchronization can be lifted without breaking Tier-4. Otherwise, the synchronization is rejected.
- Pilots / harness scaffolding that **cannot be optimized away** in operational mode are wrong by design and must be redesigned.
- The "passes savepoint parity" claim must be paired with "operational-mode variant passes Tier-4 RMSE" before any M6 milestone is closed.
- **(Critic Amendment #1)** Every parity sprint M6B1–B6 closeout MUST include an "Operational-compatibility" section classifying every new field / boundary / dtype / solver interface as **validation-only**, **operational-approved-with-evidence**, or **undecided**. Undecided items may not enter operational APIs.
  - **REVISED 2026-05-25:** Amendment #1 is superseded by ADR-028 for timestep composition: validation and operational both import `dynamics.core`. Validation wrappers (savepoint emission, HDF5, fp64 strict, snapshots) may not enter operational. Carry pruning, fusion, precision downcast, kernel selection, and segmentation remain operational concerns.
- **(Critic Amendment #2)** Operator fusion in operational mode: rule is "**must fuse OR carry a profiler-backed exception**." ADR-026 must include a compiled-region map with HLO/Nsight launch evidence for each RK and acoustic scan; an unfused operator boundary needs a documented profiler reason.
- **(Critic Amendment #3)** **Savepoint HDF5 layout ≠ operational in-memory layout.** ADR-026 must include operational state layout, peak device memory, XLA temporary/aliasing evidence where available, and a projected 1 km headroom calculation (not a 1 km run — a memory projection).
- **(Critic Amendment #4)** Precision authorization is **fail-closed**: ADR-026 may propose downcasts, but operational code may NOT depend on a new downcast until ADR-007 or a reviewed amendment authorizes that field/path with Tier-4 evidence attached. Sprint-local precision sub-documents do not become accidental production policy.

### 14.5.2 M6-perf-design sprint (gates M6B6 → M6b)

Between M6B6 (full coupled-step savepoint parity) and M6b (1h honest Canary d02), the project runs a deliberate **operational-mode design sprint** with the following acceptance:

- An "operational-mode design" ADR enumerates per-operator: (a) which carry fields drop, (b) which operators fuse, (c) which fields downcast and to what precision, (d) which solver variant is used, (e) what Tier-4 evidence validates each choice.
- The operational-mode build runs the same Canary d02 1h that M6B6 validated and meets: (a) Tier-4 RMSE envelope on T2/U10/V10, (b) wall-clock **less than 28-rank CPU WRF**, (c) zero H2D/D2H in timestep loop verified by Nsight Systems trace.
- If the operational mode cannot beat 28-rank CPU WRF within two perf-design sprints despite passing Tier-4, **the project re-opens whether the savepoint-first per-operator-parity framing was the right path**, with a full-state GPT critic + Gemini tiebreak (per `[[feedback_step_back_cadence]]`). This is the kill gate that prevents the project from shipping a correct-but-slow GPU dycore.

Sprint contract pre-drafted at `.agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md`; activates when M6B6 closes.

**Critic Amendments #5 + #6** also apply to the M6-perf-design sprint:

- **Amendment #5 (Tier-2 stability floors)**: M6-perf-design gates ALSO include Tier-2 invariants — finite/bounds audit, mass / dry-air continuity residual, water-budget residual where physics is active, conservation checks sensitive to solver/fusion changes. T2/U10/V10 are necessary but NOT sufficient. Speed cannot override invariant failure (precedent: ADR-007's prior failure).
- **Amendment #6 (1.2× = tripwire only)**: The 1.2× wall-clock target is a **first-pass kill switch only**, not the project's GPU value-proposition acceptance. ADR-026 must publish a **measured path** to the actual M7 speed target (project research synthesis rejected ≤5–7× as structurally insufficient against an 8–10× target), including: dominant hotspots, launch-count budget, memory-traffic budget, precision plan, whether PCR/batched-Thomas is required to reach target. A 1.2× pass without a credible path to M7's target should NOT be advertised as meeting the original GPU-native value proposition.

### 14.6 ADR status updates (this section is operational; not constitutional re-vote)

- **ADR-023 (conservative column solver)** → **SUPERSEDED-PROVISIONAL.** Scientifically sound minimalist-carry thesis, but failed real d02 evidence. Kept as reference; not the production architecture.
- **ADR-024 (warm-bubble gate = operator sanity)** → **ACCEPTED** (per consultation's "ADR-024's gate-policy idea is sound"). Warm-bubble is now permanently a diagnostic, never an architecture-acceptance target.
- **ADR-021/022 (full-carry / hybrid drafts)** → remain DRAFT, branch evidence only.
- **ADR-025 (to be drafted)**: WRF savepoint-harness + B-direct port ladder. Will be drafted during M6B0 and reviewed at M6B0 close.

### 14.7 Risk gates (sprint-level kill conditions, split per phase — critic Amendment #5)

**M6B0 / M6B0-R (savepoint harness + first real Fortran emission)**:
- Cannot produce isolated relinked WRF + fail-closed comparator in ≤2 sprints → escalate; consider AceCAST commercial-discovery sprint (reverses the E-lane defer).
- Tier-3 golden small-domain extraction infeasible at ≤5 GB even with short-pulse sub-sampling → re-examine the operator boundary set; reduce to `calc_coef_w` only and defer the rest to M6B1+.

**M6B1 / M6B2 (coefficient + tridiagonal solve parity)**:
- Schema-valid savepoints expose too many early divergences for worker-only debugging (>15 fields diverging at step 2) → trigger external WRF-expert human review before committing more sprints. Reconsider Option C evaluation as design-time alternative (still not implementation).

**M6B3 / M6B4 (scratch + acoustic recurrence parity)**:
- WRF small-step state surface keeps expanding beyond the env-audit estimate → re-scope. Reduce M6 close target to M6b only (defer M6c to a pre-M7 release gate).

**M6B5 (full dycore step parity)**:
- No measurable RTX 5090 speedup vs 28-rank CPU WRF on the validated operator → re-open performance section before M6b dispatch; profile; consider mixed-precision operational mode.

**Cross-cutting**:
- External human WRF expert review unobtainable → manager dispatches a Codex critical-review sprint as substitute and flags the gap to the user at M6B0 close.
- Operational `wrf.exe` sha256 changes at any point → STOP all dycore work, revert, escalate. The Gen2 baseline is the project's truth source.

### 14.8 Validation gates (binding)

1. **Savepoint parity (M6B0–B6)**: sanitizer-off; no caps; no nonfinites; exact shape/stagger/unit agreement; WRF/JAX deltas tracked per operator.
2. **10-step real d02 replay**: physics off then on; boundary off then on; first-bad-step remains null; no field reaches diagnostic caps; operator term budget serialized.
3. **1h d02 (M6b)**: no sanitizer in production path; theta physically bounded; wind maxima plausible; T2/U10/V10 RMSE inside pre-declared envelope.
4. **6h/24h Gen2 consistency (M6c)**: Tier-4 probabilistic — AceCAST-style envelope, not bitwise.
5. **Performance (post-M6c)**: wall-clock < 28-rank CPU WRF for Canary 3km; no host/device transfer in timestep loops; memory headroom for 1km; profiler proof of where time is spent.

### 14.9 M7 alignment

M7 (Canary operational v0) continues with CPU WRF as the operational backend until M6b passes, then migrates field-by-field as savepoint parity expands. Public messaging does not imply GPU-native operation before M6b. M7 may also be backed by an E-lane shadow GPU-WRF for business continuity if user authorizes.
