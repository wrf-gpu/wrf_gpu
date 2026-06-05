# GPT Hand-Fused Kernel Feasibility Review

Date: 2026-06-05
Reviewer: GPT-5.5 xhigh
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/perf-v0100`
Branch context: `worker/perf/v0100-kernel`, v0.10.0 head
Scope: analysis and planning only. No `src/` edits, no commits, no GPU runs.

## Decision

Recommendation: **SPIKE-FIRST**, not full GO.

The branch is architecturally feasible: Pallas, Triton, or a JAX-FFI custom CUDA path can be called from inside the existing JAX `jit`/pytree/scan runtime, so init, real-init, IO/wrfout, boundary, validation, and orchestration do not have to be rewritten. The production seam is the inner step in `src/gpuwrf/runtime/operational_mode.py`, especially `_acoustic_scan` and the physics adapter calls inside `_physics_boundary_step_with_limiter_diagnostics`.

The branch is not a small optimization pass. The current hot path is about **12.8k physical LOC** across runtime, dycore, couplers, Thompson, MYNN-EDMF, surface layer, and state contract files, versus **93.0k tracked physical LOC under `src/`** and **5,530 tracked repo files**. A hand-fused branch would likely add or replace **12k-20k LOC** for kernels, wrappers, proofs, and fallback wiring on a Pallas/Triton path, or **18k-26k LOC** if it falls back to full custom CUDA FFI.

My effort estimate is **about 30% of the from-zero WRF-in-JAX port to date**, not 10%, and not as-much-again. It is smaller than the original port because init, IO, validation, contracts, fixture generation, non-hot schemes, and the JAX oracle remain. It is much larger than 10% because it reimplements the production numerical hot path and changes shared-core architecture.

Earliest decisive viability checkpoint: **1-2 sprints / 2-3 agent-runs** on a single representative kernel, preferably the MYNN-EDMF mass-flux kernel. Stop the branch if that spike does not show real occupancy/launch-count gain with bit-exact or explicitly pre-approved tolerance-gated parity.

Expected full-branch speedup, if the spike succeeds: **1.4x-1.8x warmed coupled d02**, medium-low confidence. **2x is possible but not supported until the spike proves the backend actually beats XLA on this workload. 3x is upside, not a plan. 5x is not a decision-grade target for this branch.** Larger domains would likely improve GPU saturation without a rewrite because the current d02 grid is small: 159 x 66 x 44 = 461,736 mass cells (`proofs/perf/roofline_costonly.json`).

## Baseline And Proof Caveat

The prompt's phase split is real repo evidence, but it mixes two v0.10 timing states:

- Current v0.10.0 warmed coupled d02 step is about **64.7-64.9 ms**: `proofs/v0100/wave_b_timing_full_fp64_u1_nsed16.json` reports `64.7308378 ms`, `proofs/v0100/wave_b2_mynn_timing.json` reports `64.762187 ms`, and `proofs/v0100/v0100_release_timing_head_default16.json` reports `64.9233865 ms`.
- The MYNN/Thompson/dycore split **33.84 / 20.71 / 16.59 / 3.21 ms** is from the earlier **74.35 ms** Wave-B baseline before Thompson `NSED=16` became the fast default (`proofs/v0100/wave_b_scope.md:21-36`, `proofs/v0100/wave_b_scope.md:67-70`).
- The absolute component costs remain decision-relevant. The normalized percentages should not be blindly applied to the final 64.76 ms state. In the current final state, MYNN remains a roughly 33.9 ms block (`proofs/v0100/wave_b2_mynn_profile.json`), so it is an even larger share of the final wall.

The structural performance diagnosis is strong:

- Dycore-only roofline: **2.263 GFLOP**, **5.661 GB**, AI **0.3998 FLOP/byte**, **16.898 ms** wall, HBM floor **3.159 ms**, actual/HBM floor **5.3486x**, achieved HBM **18.7%**, achieved fp64 **8.2%** (`proofs/perf/roofline_costonly.json`).
- Kernel-level diagnosis: about **11,160 GPU ops/step** = **7,236 kernels + 3,922 memory ops**, with **6,890 tiny elementwise fusions/step** and **43%-68% GPU idle** (`proofs/perf/compute_cycle_analysis.md:93-110`).
- Precision is not the lever: fp32 was measured slower/no-op in Wave-B, **74.35 ms fp64 vs 74.57 ms gated fp32** (`proofs/v0100/wave_b_scope.md:42-51`), and the roofline explains why (`proofs/perf/compute_cycle_analysis.md:116-126`).

## 1. Scope, Files, Lines, LOC

Measured repo size:

| Scope | Count |
|---|---:|
| Tracked repo files | 5,530 |
| Tracked files under `src/` | 243 |
| Tracked physical LOC under `src/` | 93,045 |
| Current hot-path files listed below | 12,782 LOC |

Tracked `src/` LOC by major module:

| Module | LOC |
|---|---:|
| `src/gpuwrf/physics` | 40,482 |
| `src/gpuwrf/validation` | 7,538 |
| `src/gpuwrf/init` | 7,607 |
| `src/gpuwrf/dynamics` | 7,532 |
| `src/gpuwrf/coupling` | 5,615 |
| `src/gpuwrf/io` | 5,507 |
| `src/gpuwrf/runtime` | 3,676 |
| `src/gpuwrf/contracts` | 3,132 |

### Hand-Fused Candidate Surface

| Area | Current files and line refs | Current LOC counted | Fused scope | Estimated fused LOC |
|---|---|---:|---|---:|
| Runtime operational seam | `runtime/operational_mode.py` x64 enabled at `:114`; acoustic unroll hook `:120-128`; `_acoustic_scan` `:1323-1408`; RK stage scan `_rk_scan_step` `:1668-1733`; coupled physics/boundary step `:2076-2276`; chunked jit `_advance_chunk` `:2460-2494`; public entry `run_forecast_operational` `:2591-2648`; segmented/single-scan entries `:2651-2766` | 2,897 | Keep public orchestration; swap inner kernels at acoustic and physics-call seams | 500-1,200 wrappers and integration |
| Dycore/acoustic substep | `dynamics/core/acoustic.py` state/config and substep: `AcousticCoreState` `:99-219`, `advance_uv_wrf` `:384-510`, `acoustic_substep_core` `:561-814`; `advance_w.py` `advance_w_wrf` `:131-434`; `mu_t_advance.py` `advance_mu_t_wrf` `:81-389`; `tridiag_solve.py` `:13-62`; core helpers `small_step_prep.py`, `calc_p_rho.py`, `rk_addtend_dry.py`, `small_step_finish.py` | about 2,666 for the counted core files, excluding broader advection/diffusion | Fuse acoustic substep stencils, face averages, EOS, mu continuity, geopotential update, and possibly stage prep/finish. Keep XLA/cuSPARSE tridiag unless the spike proves a fused vertical solve is worth it | 3,500-6,000 |
| Physics coupler hot adapters | `coupling/physics_couplers.py` column layout helpers `_to_columns`/`_from_columns` `:269-278`; Thompson adapter and reassembly `:664-783`; MYNN state build/reassembly `:816-1002`; surface adapter/diagnostics `:1043-1089`; `scan_adapters.py` has non-hot/fail-closed scheme registry `:13-60`, maps `:991-1036` | 2,288 | Thin JAX wrappers around fused kernels. Avoid standalone coupler fusion unless attached to MYNN/Thompson because previous wrapper fusion measured <0.1% and broke bit identity | 500-1,000 |
| MYNN-EDMF / PBL / surface | `physics/mynn_pbl.py` qke solve `:804-846`, mean tendencies and 4 tridiag solves `:859-941`, EDMF call and closure `:984-1069`, jitted entries `:1111-1139`; `physics/mynn_edmf.py` DMP mass-flux `:127-421`; `surface_layer.py`; `tridiagonal_solver.py` | 2,490 | Highest-value target. First fuse `dmp_mf_columns` / EDMF mass flux. Later fuse turbulence, mean tendencies, and selected surface coupling if spike succeeds | 3,000-5,000 |
| Thompson sedimentation | `physics/thompson_column.py` precision/perf note `:125-141`; `NSED_MAX` cap `:889-923`; sedimentation unroll `:957-974`; per-species scan `_sed_one_species` `:1145-1213`; four-species sedimentation `:1291-1354`; jitted entry `:1444-1453` | 1,453 full file; about 466 for sedimentation-focused section | Fuse remaining sedimentation/fall-speed/source-sink hot loops, but do not undo the validated `NSED=16` cap | 1,500-3,000 sedimentation; 3,000-5,000 full Thompson |
| State contract | `contracts/state.py` State pytree `:354-450`; `replace` casting/sync semantics `:604-645`; pytree flatten/unflatten `:653-663` | 693 | Keep unchanged if possible. Fused wrappers should accept/return explicit leaves and call `state.replace` as today | 0-300 if ABI helpers are needed |

Current hot-path LOC counted from these exact files: **12,782**. This is not all code that would be touched. A full dycore kernel branch may also need broader large-step advection/diffusion modules outside the counted core files.

### What Stays In JAX

Unchanged or only thin-wrapper changes:

- `src/gpuwrf/init/**` and native real-init: **7,607 LOC**.
- `src/gpuwrf/io/**`, wrfout/restart writing, namelist checks, and non-hot host IO: **5,507 LOC**.
- Boundary forcing/application unless a later dycore-specific proof says otherwise.
- Public `State`/`GridSpec` contracts and SoA layout. `State` is already a pytree of named arrays (`contracts/state.py:354-450`, `contracts/state.py:653-663`).
- Validation and fixture harnesses: **7,538 LOC** under `src/gpuwrf/validation`.
- Runtime orchestration: segmented/single-scan public entries, cadence logic, radiation cadence, finite guards, and final precision enforcement should remain.
- Non-hot and fail-closed schemes in `scan_adapters.py`: alternate Kessler/Lin/WSM/Morrison/WDM6 microphysics, alternate PBL/surface/cumulus paths, and radiation diagnostics unless separately profiled and approved.
- RRTMG, Noah-MP/noah-classic, cumulus, diagnostics, and daily integration pipeline.
- Existing JAX kernels should remain as reference or fallback, not be deleted.

## 2. Percentage Of The Total Project

Basis:

- Total tracked `src/` LOC: **93,045**.
- Current hot path counted above: **12,782 LOC** = **13.7% of `src/` physical LOC**.
- The branch would not merely edit those lines. It would add backend kernels, launch wrappers, ABI declarations, shape checks, fallback paths, proof harnesses, profiler harnesses, and possibly build tooling.
- Hand-fused Pallas/Triton/CUDA kernels are usually more verbose than the JAX source they replace because indexing, tiling, memory layout, boundary masks, launch metadata, and ABI plumbing become explicit.

Estimate:

- Pallas/Triton path: **12k-20k new/changed LOC**, roughly **13%-22% of current `src/` LOC**.
- Custom CUDA FFI path: **18k-26k new/changed LOC**, roughly **19%-28% of current `src/` LOC**, plus higher maintenance cost.
- Engineering effort as a percentage of the entire from-zero WRF-in-JAX port to date: **about 30%**.

Classification:

- **Not 10%**: it reimplements the highest-risk runtime numerics, crosses shared-core file ownership, and needs ADR-grade kernel ABI decisions.
- **Not as-much-again**: init, IO, validation, fixtures, contracts, non-hot schemes, native real-init, wrfout, diagnostics, and the already-validated JAX oracle are retained.
- Best label: **major branch / about one-third of the port**, with an early stop gate.

## 3. Sprint Plan

Rough total if the spike succeeds: **12-18 agent-sprints** for Pallas/Triton. If the branch falls to custom CUDA FFI for most kernels, budget **18-26 agent-sprints**.

| Phase | Sprints | Goal | Go/no-go gate |
|---|---:|---|---|
| 0. Backend/ABI decision memo | 1 | Mini-ADR for Pallas vs Triton vs CUDA FFI, shape-static ABI, x64 policy, profiler commands, fallback wiring | Mini-ADR accepted; no in-loop host/device transfer; import/build smoke only |
| 1. Representative spike | 1-2 | Hand-fuse MYNN-EDMF mass flux only, leaving the JAX closure otherwise intact | Bit-exact or approved tolerance parity; measured block speedup; nsys/kernel-count proof |
| 2. MYNN expansion | 3-5 | Fuse remaining MYNN closure pieces that are not already optimal XLA primitives | `mynn_adapter` and full step improve materially; 24h skill/conservation unchanged |
| 3. Thompson sedimentation | 2-3 | Fuse remaining sedimentation/fall-speed loops after `NSED=16` | Precip oracle, water budget, d02 24h, no clips, no transfer regression |
| 4. Acoustic/dycore substep | 4-6 | Fuse acoustic substep stencils and selected RK-stage prep/finish | Dycore savepoints, idealized gates, conservation, full coupled step, nsys proof |
| 5. Integration/hardening | 2-3 | Runtime feature flag, fallback, profiler artifacts, compile/memory audit, docs | v0.10 regression suite plus perf proof object |
| Optional custom CUDA rescue | 2-4 extra | Replace failed Pallas/Triton kernel with CUDA FFI | Same correctness/perf gates, plus build/ABI audit |

The major advantage is that every fused kernel can be tested against an already-validated JAX op. This turns the work into independently checkable replacements:

1. Isolated kernel output parity against JAX.
2. Adapter-level parity.
3. One-step coupled parity.
4. 24h d02 skill/conservation and transfer audit.
5. Profiler proof.

This is not an "only find out at the end" rewrite.

## 4. JAX Framework Compatibility

### What Stays Compatible

Pallas/Triton/custom-call kernels can live inside the existing framework if they are pure, shape-static, device-resident functions:

- Existing `State` is already a JAX pytree of SoA leaves (`contracts/state.py:354-450`, `contracts/state.py:653-663`).
- `State.replace` already gives a typed functional update path and syncs total/legacy/perturbation fields (`contracts/state.py:604-645`).
- The production operational entry is already jitted and donated (`runtime/operational_mode.py:2460-2494`, `runtime/operational_mode.py:2591-2648`).
- The acoustic loop is a single inner call site: `_acoustic_scan` builds an `AcousticCoreState` and runs `jax.lax.scan` over `acoustic_substep_core` (`runtime/operational_mode.py:1323-1408`).
- Physics call sites are localized in the coupled step: Thompson at `runtime/operational_mode.py:2125-2130`, surface at `:2132-2165`, and MYNN/PBL at `:2182-2188`.

The practical integration seam:

- For MYNN spike: replace `mynn_edmf.dmp_mf_columns` inside `_edmf_arrays_from_state` (`mynn_pbl.py:984-1027`) with a pure fused call returning the same arrays. `mynn_adapter` and `State` update code remain.
- For Thompson: replace `_sedimentation` or `_sed_one_species` (`thompson_column.py:1145-1354`) behind the existing `step_thompson_column_with_precip` entry.
- For dycore: replace the body called from `_acoustic_scan` (`runtime/operational_mode.py:1390-1406`) with a fused acoustic-substep function, leaving RK scan and public runtime entries intact at first.

This does **not** force a broad rearchitecture on day one.

### What Breaks Or Becomes Risky

Risks that require ADR/proof, not optimism:

- **Argument count and ABI shape**: a fused dycore substep touches many State/BaseState/Grid metrics leaves. A kernel ABI with 50-100 arrays is unwieldy. Packing some static metrics may become necessary, which would be an ADR-level contract change.
- **Donation/aliasing**: JAX donation does not automatically mean a custom call aliases outputs in the intended way. Buffer assignment and D2D memcpy audits are required.
- **Dynamic control flow**: physics kernels have masks, vertical recurrences, per-column activity, and fixed scans. These must remain shape-static. No host callbacks, Python loops, or per-step data transfers.
- **Bit identity**: Pallas/Triton may not preserve JAX's exact math lowering for operations like `where`, `exp`, `pow`, or recurrence order. If bit-exactness fails, the branch needs a pre-approved tolerance ladder and a physical evidence gate. It must not silently loosen fidelity.
- **Compile/build complexity**: Pallas may stay in JAX compile flow, Triton may add kernel compilation and integration fragility, and CUDA FFI adds C++/CUDA build and ABI maintenance.

### Local Backend Authority

The accepted backend ADR selects JAX/XLA as primary and explicitly allows a gated fallback to Triton via `jax.experimental.pallas` or a thin shim only after a per-scheme mini-ADR with profile evidence (`.agent/decisions/ADR-001-backend-selection.md:14-16`). The same ADR requires real-scheme fallback evidence including registers, local memory, occupancy, launches, and correctness (`.agent/decisions/ADR-001-backend-selection.md:131-139`).

Local environment evidence:

- Installed JAX/JAXLIB: `0.10.0`.
- `triton` import is available.
- `pyproject.toml` only says `jax>=0.4`; the hard pin is in ADR-001, not pyproject (`pyproject.toml:11`, `.agent/decisions/ADR-001-backend-selection.md:14`).

### fp64 On Blackwell

The branch must preserve fp64. Blackwell fp64 arithmetic is available, and the repo already runs with `jax_enable_x64=True` (`runtime/operational_mode.py:114`). The roofline says fp64 throughput is not the bottleneck: dycore uses only **8.2% of fp64 peak** and is **5.35x above the HBM floor** (`proofs/perf/roofline_costonly.json`).

The real fp64 risk is not "can the GPU execute fp64." It can. The risks are:

- Pallas/Triton fp64 lowering and math intrinsics may not be bit-identical to XLA.
- Custom kernels can easily change recurrence order.
- Register pressure from fp64 plus many fields may reduce occupancy.
- The branch could trade one launch problem for a register-spill/local-memory problem.

Precision reduction is out of scope. bf16/fp32 would cost fidelity and has already shown no speed benefit in this workload.

## 5. Risk Per Step And Early Validability

This is **not all-or-nothing** if managed correctly.

Every replacement can be isolated behind the existing JAX reference:

- JAX operation remains the oracle.
- Fused kernel takes the same input arrays and returns the same output arrays.
- The adapter compares arrays before the fused call becomes production default.
- Full-step and 24h gates are later confirmation, not first discovery.

### Best First Spike

Spike target: **MYNN-EDMF mass flux**, specifically the `mynn_edmf.dmp_mf_columns` path called by `_edmf_arrays_from_state`.

Why this target:

- It is the largest measured MYNN internal phase: **15.6605 ms** (`proofs/v0100/wave_b2_mynn_profile.json`).
- It is about **48.6% of the MYNN closure kernel** (`15.6605 / 32.2097 ms`) and about **24% of the final 64.76 ms coupled step**.
- It is isolated in code: `physics/mynn_edmf.py:127-421` and the caller `physics/mynn_pbl.py:984-1027`.
- The proof says the structure is `vmap(columns) o vmap(plumes=8) o lax.scan(levels~42): dependent vertical recurrence` (`proofs/v0100/wave_b2_mynn_profile.json`). That is exactly the kind of pattern where XLA may be launch/occupancy-limited, and exactly the kind of pattern a hand-fused kernel must prove it can handle.
- Acoustic unroll is a worse first spike now: Wave-B found unroll=2 only **0.67%** faster and kept default at 1 (`proofs/v0100/wave_b_scope.md:25-27`). The dycore still has theoretical roofline headroom, but current repo evidence says MYNN is the higher-value first proof.

Spike scope:

1. Freeze the output ABI of `dmp_mf_columns`.
2. Implement one fused backend version only. Prefer Pallas first if the JAX integration is clean; fall back to Triton/custom CUDA only through the mini-ADR.
3. Compare all EDMF output arrays against JAX for representative d02 columns and MYNN fixtures.
4. Insert behind an env flag or static config switch in `_edmf_arrays_from_state`.
5. Benchmark `edmf_massflux`, `mynn_closure_kernel_only`, `mynn_adapter`, and full warmed step.
6. Produce nsys/kernel-count/transfer proof. No performance claim without profiler artifacts.

Spike pass gate:

- Correctness: bit-exact output against JAX for all returned arrays, or a pre-approved tolerance gate if bit-exactness is impossible for a documented math-lowering reason.
- Performance: at least **2x faster on the EDMF mass-flux block** (`15.66 ms -> <=7.8 ms`) and at least **8%-10% full-step improvement** or a clearly equivalent measured reduction in `mynn_closure_kernel_only`.
- Transfer audit: zero host/device transfers inside the timestep loop and no D2D memcpy explosion.
- Occupancy/register proof: no local-memory spill pattern that makes the result non-scalable.

Spike fail gate:

- If the fused EDMF block is only marginally faster (<25% block speedup), not bit-validable, or reduces one kernel count while increasing D2D/register-spill cost, stop the major branch.
- If Pallas cannot express the kernel cleanly but the correctness/perf signal is otherwise promising, commission a narrower Triton/CUDA FFI rescue spike, not the full branch.

Agent-runs at risk before decisive checkpoint:

- **Minimum**: 2 agent-runs: one implementer spike plus one verifier/profiler review.
- **Practical maximum before stop/go**: 3 agent-runs if the first backend needs a small fallback attempt.
- Do not commit to the 12-18 sprint branch before this checkpoint.

### Risk Ladder

| Phase | Risk | Why | Early validation |
|---|---|---|---|
| EDMF spike | High but bounded | Dependent vertical recurrence, 8 plumes, fp64 math-order sensitivity | Direct JAX array oracle, block timer, nsys kernel-count proof |
| Full MYNN | High | Closure is genuine compute/control-flow, not wrapper overhead; previous mechanical fusions measured <0.1% and broke bit identity | MYNN adapter parity, 24h skill/conservation |
| Thompson sedimentation | Medium-high | Sedimentation has fixed scans and precip conservation; `NSED=16` already captured the easy 9.62 ms win, so remaining upside is smaller | Precip oracle, water budget, wet-column histograms |
| Dycore acoustic | High | Many fields, C-grid staggering, boundary terms, and pressure/geopotential stability. Tridiag is already cuSPARSE PCR, so only stencil fusion is the lever | Savepoints, idealized warm bubble/Straka, full coupled regression |
| Coupler-only fusion | Low upside, medium fidelity risk | Existing surface+MYNN fusion was 0.0946% and not bit-identical | Only fuse as part of scheme kernels, not standalone |

## 6. Honest Ceiling And Recommendation

### Speedup Ceiling

Observed final step: **about 64.76 ms**.

Practical expectation if the spike succeeds:

- MYNN-EDMF mass flux: 15.66 ms block. If fused to 5-8 ms, full-step gain is about 7-11 ms.
- Additional MYNN closure pieces: possible but uncertain. Tridiag solve is already an XLA primitive and not the obvious target (`proofs/v0100/wave_b2_mynn_profile.json`).
- Thompson: previous `NSED=16` already removed **9.62 ms** (`proofs/v0100/wave_b_scope.md:67-70`); remaining sedimentation/source-sink fusion may still help but should not be counted as another easy 10 ms.
- Dycore: roofline says large theoretical headroom (**16.898 ms vs 3.159 ms HBM floor**), but current acoustic unroll evidence says XLA-level unroll is not a strong current lever. Hand-fused dycore could work, but it is a later high-risk phase.

Expected full-branch result after successful spike:

- Conservative: **1.25x-1.4x** if MYNN spike works but Thompson/dycore provide only modest gains.
- Realistic target: **1.4x-1.8x**.
- Upside: **2x-2.5x** only if MYNN, Thompson, and dycore all show real fused-kernel occupancy gains.
- Not credible as a plan: **5x** on current d02. That is a theoretical hardware-floor direction, not a commissioned-branch acceptance target.

### Biggest Unknowns

1. Whether Pallas/Triton can produce a materially better fp64, recurrence-heavy kernel than XLA without register spills.
2. Whether bit-exactness can be preserved. If not, whether the principal accepts a tolerance ladder for a performance branch.
3. Whether d02 is too small to reward the rewrite. Larger domains may saturate the 5090 better without touching code.
4. Whether custom CUDA FFI becomes necessary. That would increase effort, maintenance, and ABI risk.
5. Whether the current XLA graph already has hidden optimizations for the vertical/tridiag pieces, leaving less unfused headroom than roofline suggests.

### Recommendation

Commission only the **1-2 sprint MYNN-EDMF mass-flux spike** first.

Full branch GO criteria after spike:

- Bit-exact or explicitly approved tolerance-gated parity.
- At least 2x speedup on the EDMF block and about 8%-10% full-step improvement.
- Profiler proof that kernel count/occupancy improved rather than shifting cost to D2D copies or spills.
- No in-loop host/device transfers.

If the spike fails, recommendation becomes **NO-GO** for the hand-fused hot-path branch on current d02. Keep the JAX/XLA path, exploit larger-domain saturation where operationally relevant, and reserve custom kernels only for a later, narrower scheme-specific bottleneck with stronger proof.

## Handoff

Objective:

- Produce a decision-grade feasibility analysis and phased plan for a possible hand-fused GPU-kernel branch replacing the JAX/XLA hot path.

Files changed:

- Added `.agent/reviews/2026-06-05-gpt-hand-fused-kernel-feasibility.md`.
- Will add `/tmp/v0100_fused_feasibility.done` after this review is written.

Commands run:

- `taskset -c 0-3 git status --short --branch`
- `taskset -c 0-3 sed -n ... PROJECT_CONSTITUTION.md AGENTS.md .agent/README.md`
- `taskset -c 0-3 sed -n ... .agent/skills/.../SKILL.md`
- `taskset -c 0-3 git ls-files | wc -l`
- `taskset -c 0-3 git ls-files src | wc -l`
- `taskset -c 0-3 git ls-files src | xargs wc -l`
- `taskset -c 0-3 find src/gpuwrf/...`
- `taskset -c 0-3 nl -ba src/gpuwrf/runtime/operational_mode.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/dynamics/core/acoustic.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/physics/thompson_column.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/physics/mynn_pbl.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/physics/mynn_edmf.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/coupling/physics_couplers.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/coupling/scan_adapters.py ...`
- `taskset -c 0-3 nl -ba src/gpuwrf/contracts/state.py ...`
- `taskset -c 0-3 jq '.' proofs/perf/roofline_costonly.json`
- `taskset -c 0-3 jq '.' proofs/v0100/wave_b2_mynn_profile.json`
- `taskset -c 0-3 nl -ba proofs/perf/compute_cycle_analysis.md ...`
- `taskset -c 0-3 nl -ba proofs/v0100/wave_b_scope.md ...`
- `taskset -c 0-3 python -c "import jax; print(jax.__version__)"`
- `taskset -c 0-3 python -c "import importlib.util; ..."`

Proof objects used:

- `proofs/v0100/wave_b_timing_full_fp64_u1_nsed16.json`
- `proofs/v0100/wave_b2_mynn_timing.json`
- `proofs/v0100/v0100_release_timing_head_default16.json`
- `proofs/v0100/wave_b_timing_full_fp64_u1.json`
- `proofs/v0100/wave_b_scope.md`
- `proofs/v0100/wave_b2_mynn_profile.json`
- `proofs/v0100/wave_b2_fusion_ab.json`
- `proofs/v0100/wave_b2_edmf_unroll_ab.json`
- `proofs/perf/roofline_costonly.json`
- `proofs/perf/compute_cycle_analysis.md`

Proof objects produced:

- This feasibility review.
- `/tmp/v0100_fused_feasibility.done`.

Unresolved risks:

- No new GPU benchmarks were run in this analysis turn by request.
- Pallas/Triton/custom CUDA fp64 exactness and register behavior are unproven for the actual EDMF recurrence.
- Existing phase percentages come from multiple timing baselines; the review uses absolute measured milliseconds where possible.
- A full branch requires mini-ADR approval before any non-JAX backend kernel proceeds.

Next decision needed:

- Decide whether to commission the **MYNN-EDMF mass-flux spike** with a hard 1-2 sprint go/no-go gate.
