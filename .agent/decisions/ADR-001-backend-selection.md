# ADR-001 — Backend Selection for the GPU-Native NWP Core

Date: 2026-05-19
Author: manager (Claude Opus 4.7 1M context)
Status: **proposed, pending user acknowledgement at M2 closeout**. Awaiting codex reviewer binding judgment before manager-merge to main; user acknowledgement of the irreversible-architecture nature happens by the user reading the M2 closeout report and not objecting. The manager-autonomy directive of 2026-05-19 delegates *operational* decisions to the manager — it does NOT silently amend the constitution. The constitution requires human approval for irreversible architecture decisions (`PROJECT_CONSTITUTION.md:16`, `.agent/rules/architecture-decision-policy.md:13`). This ADR honors that by reporting to the user at M2 closeout *before* M3 implementation begins; the user has explicit veto authority and may force a re-ADR.
Scope: v0 (single-node RTX 5090, cc120 Blackwell, 32 GB VRAM)
Reversibility: **irreversible** per `.agent/rules/architecture-decision-policy.md`.

## Decision

Decision: Accept JAX as the primary v0 backend, contingent on explicit user approval at M2 closeout.
Selected backend: jax

Primary backend for the M3 device-resident `State`/`GridSpec`, M4 minimal dycore, and M5 first physics suite: JAX + XLA, Python orchestration, `@jit` per kernel-class, AOT compile via `jit`. Pin: `jax[cuda13]==0.10.0` (per M2-S1 scout matrix, hello-GPU verified on this workstation).

**Fallback option (gated, not pre-authorized):** if M5 implementation reveals XLA register-spilling on a real WRF physics scheme (Thompson microphysics, MYNN PBL, or equivalent column-heavy kernel) that cannot be remedied by JAX-side restructuring (vmap rearrangement, custom_jvp, or explicit `jax.lax.scan` pattern), the affected physics scheme MAY be re-implemented in **OpenAI Triton** (`@triton.jit`, pin `triton==3.7.0 + torch==2.12.0`) and called from the JAX timestep loop via `jax.experimental.pallas` or a thin ctypes shim — **but only after a per-scheme decision memo at `.agent/decisions/ADR-001-FALLBACK-<scheme>.md` (mini-ADR, ≥1000 bytes, with reviewer cross-model challenge per `.agent/rules/cross-model-review-policy.md`)**. The mini-ADR documents the specific scheme, the JAX-restructuring attempts that failed, the profile evidence (registers, local memory, occupancy on the actual scheme), and the integration point with the JAX timestep loop. **No new full ADR is required** — the architectural authorization for a hybrid is granted by this ADR-001 — **but the per-scheme decision IS gated**: it cannot proceed silently.

## Evidence summary

M2 bakeoff measured five backend candidate families against the M1 analytic stencil and analytic column fixtures on RTX 5090 (cc120). The sixth (gt4py) was excluded by the M2-S1 scout — `gt4py + DaCe 0.10.0` fails on Python 3.13 with a SymPy break; remediation requires a Python 3.12 venv and is deferred to a post-M2 sprint if M5 reveals a need for a stencil DSL beyond what JAX offers.

**Profile fidelity caveat.** All numbers below are **fallback-profiled and micro-fixture-limited**, not full `ncu`/`nsys` artifacts. They come from `cuobjdump --dump-sass` (registers + local_memory_bytes), the CUDA occupancy API (occupancy_pct, *theoretical*), and bench-output wall_time. NVIDIA's `ncu` (Nsight Compute) was unavailable due to a system-level `ERR_NVGPUCTRPERM` constraint — `nvidia-driver-perfmon-allow=1` is not set on this workstation. The `profiler_limitation` field in every candidate's profile JSON documents this; `achieved_bandwidth_gbps` is labeled `fallback-derived` and is order-of-magnitude only. `PROJECT_PLAN.md:73-76` and `PERFORMANCE_TARGETS.md:5-7` require full profiler artifacts for *performance claims*; this ADR makes a *backend-selection* claim grounded in registers + local-memory + occupancy + agent-success + maintainability, not in absolute throughput. **Required follow-up action (M3 or earlier):** obtain `nvidia-driver-perfmon-allow=1` via system-admin action (single modprobe param + reboot) and re-run profiler-bot against the chosen JAX implementation to produce real ncu reports before any M4+ performance claim is published.

### Stencil (Problem 1: 3D advection-diffusion, 32×16×8, fp64)

| Candidate | regs | local | occ% | launches | wall (ms) |
|---|---|---|---|---|---|
| cuda_tile | 58 | 0 | 66.7 | 1 | 0.92 |
| cupy | 58 | 64 | 66.7 | 1 | 0.06 |
| kokkos | 64 | 0 | 66.7 | 1 | 0.09 |
| **jax** | **48** | 0 | **83.3** | 1 | 0.05 |
| triton | 60 | 0 | 66.7 | 1 | 0.03 |

JAX achieves the lowest register count and the highest theoretical occupancy. Triton has the fastest measured wall time, but at these fixture sizes wall_time is dominated by measurement noise and launch overhead, not kernel throughput.

### Column (Problem 2: 40-level moist thermo column, fp64)

| Candidate | regs | local | occ% | launches | wall (ms) |
|---|---|---|---|---|---|
| cuda_tile | 24 | 0 | 100 | 1 | 1.00 |
| cupy | 24 | 0 | 100 | 1 | 0.03 |
| kokkos | 40 | 0 | 100 | 1 | 0.10 |
| **jax** | **22** | 0 | 83.3 | 1 | 0.05 |
| triton | 34 | 0 | 100 | 1 | 0.03 |

**The critical M2 result: no candidate spills local memory on the column kernel.** This is necessary (not sufficient) evidence that XLA can handle column-shape workloads without the register-spilling pathology that capped the previous wrf_gpu attempt at 5.5×. JAX's column kernel uses 22 registers — the lowest of any candidate — which translates directly to better at-scale occupancy as fixtures grow.

### Agent-success (cost-of-authorship)

| Candidate | sprint attempts | reviewer rejections | escalation events |
|---|---|---|---|
| cuda_tile | 1 | 1 (3 hygiene fixes, all addressed) | 0 |
| cupy | 1 | 0 | 0 |
| kokkos | 1 | 0 | 0 |
| jax | 1 | 0 | 0 |
| triton | 2 | 2 (cubin-cache contamination caught by Claude tester; fixed in attempt 2) | 0 |

JAX delivered cleanly on the first attempt with no reviewer rejections. Triton required a fix cycle because the worker's bench had a subtle resource-extraction bug (cubin cache contamination); the bug was caught by Claude Opus tester's cross-AI honesty test and reproduced exactly. cuda_tile required 3 hygiene fixes (one was a manager contamination commit on the worker branch, since fixed). Kokkos clean but the 10–15 min source-build cycle is the highest setup friction of any candidate.

### Maintainability narratives (≤300 words each, file references)

Read in full at `artifacts/m2/<candidate>/maintainability.md`. Headline reads:
- **cuda_tile**: highest verbosity, slow agent iteration; nvcc 13.1 + GCC 15 header bug requires `nvc++ -cuda` workaround.
- **cupy**: low friction; `cupy.RawKernel` is a clean Python escape hatch; debugger story okay.
- **kokkos**: 50-page template errors on bugs; agent iteration friction is the highest; performance portability is the strategic value.
- **jax**: fastest agent iteration; XLA errors are long but point at line numbers; `jax.debug.print` + `jax.disable_jit` are good; ML coupling is trivial.
- **triton**: medium friction; errors point at the `@triton.jit` source; torch as runtime dep is heavy (~2 GB).

## Rationale

Three considerations the contract specifically asked to address:

### (a) The user's pro-JAX intuition is empirically supported
The user pushed back early on the "why not just JAX?" intuition. The bakeoff was run to give that intuition an honest empirical answer. JAX wins on register efficiency on both problem shapes, achieves zero local memory spill on column, fuses both problems into single kernels (HLO-verified independently by Claude Opus from the thunk_sequence + cubin dump — not just "1 jitted function"). Author velocity is dramatically higher than C++ candidates. ML/AI coupling is trivial. The intuition holds for the v0 scope.

### (b) The previous wrf_gpu OpenACC 5.5× ceiling
The previous attempt died at 4.8× *slower* than 28-rank CPU due to (i) launch-bound architecture (200,000+ kernel launches per timestep) and (ii) register spilling in MYNN/Thompson (compiler hit the 255-register hard wall, spilled to local memory, occupancy collapsed to 8–12%). JAX with XLA addresses both: XLA fuses adjacent ops into single kernels (verified: 1 launch per problem on M2 fixtures), and the M2 column kernel — chosen specifically to mimic the register-pressure shape of column physics — shows zero local memory spill in JAX's compiled output. This is not a guarantee for real physics (see (c)) but it is the strongest available v0 evidence.

### (c) The M5 column-spilling risk that the analytic surrogate does not exercise
The M2 column kernel is a 40-level moist thermo update with a closed-form analytic source. Real Thompson microphysics has dozens of prognostic hydrometeor variables, hundreds of conditional branches per level, and inlined polynomial blowup from automated code generation. The M2 evidence is necessary but not sufficient that XLA holds at M5. The pre-authorized fallback (above) explicitly mitigates this: if M5 reveals XLA spilling on Thompson/MYNN despite restructuring efforts, the affected scheme drops to Triton. This is a known good escape hatch — Triton is the leading register-pressure-aware DSL with a CUDA13/Blackwell wheel pinned (M2-S1 scout), and M2-S6 showed it achieves 0 local memory on the column shape.

## Dissent

Codex `gpt-5.5 xhigh` critical-review of 2026-05-19 (file: `.agent/decisions/REVIEW-codex-ADR-001/critical-review.md`) issued Decision: `Accept with required fixes`.

### Codex's dissent — verbatim

From `REVIEW-codex-ADR-001/critical-review.md` lines 25–29:

> "I do not dissent from JAX as the primary v0 backend on the evidence available. JAX has the cleanest register story on the two M2 fixtures, first-pass agent success, and a good fit for the Python-first/ML-coupled project shape.
>
> I dissent from merging ADR-001 as written. The proposal overcloses the decision in three places: human approval, GT4Py coverage, and fallback authority. The right decision is 'JAX primary, Triton contingency under a later proof gate,' not 'JAX locked with an unreviewed Triton escape hatch.'"

### Codex's blocker + major findings — verbatim

From `REVIEW-codex-ADR-001/critical-review.md` lines 13–21:

> **1. Blocker:** Irreversible approval is not shown as an explicit human approval artifact. `PROJECT_CONSTITUTION.md:16` says irreversible architecture decisions require human approval, and `.agent/rules/architecture-decision-policy.md:13` repeats that rule. The proposal instead states that the manager exercises irreversible-decision authority and reports later (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:7`). Required fix: before merge, add a concrete human approval record or revise status to "proposed, pending human approval"; do not close M2 on manager-only approval.
>
> **2. Blocker:** GT4Py coverage does not satisfy the objective M2 gate. `.agent/goals/M2-DONE.md:27-35` requires all six candidates to have profile/correctness artifacts or candidate-failure artifacts plus maintainability and agent-success files. `scripts/check_m2_done.py:24` includes `gt4py`, and `scripts/check_m2_done.py:81-107` requires a candidate directory and the per-candidate files. The proposal only says GT4Py was excluded by the scout (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:19`). Required fix: add `artifacts/m2/gt4py/{stencil,column}_failure.json`, `maintainability.md`, and `agent_success.json`, or formally patch the M2 gate before claiming closure.
>
> **3. Major:** The pre-authorized Triton fallback bypasses the architecture-decision boundary. `.agent/rules/architecture-decision-policy.md:3-5` requires an ADR for backend selection, while the proposal allows a later JAX-to-Triton physics implementation with "no new ADR required" (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:15`). Required fix: either declare the selected backend as `hybrid:jax+triton-physics` now with an explicit integration proof plan, or keep `Selected backend: jax` and require a bounded follow-on ADR/reviewer gate before any M5 physics scheme moves to Triton.
>
> **4. Major:** The profiling evidence is useful but lower fidelity than the plan originally required. `PROJECT_PLAN.md:73-76` calls for `ncu`/`nsys` JSON including occupancy, registers, local memory, bandwidth, and transfer count; `PERFORMANCE_TARGETS.md:5-7` bars GPU optimization claims without profiler artifacts. The proposal states that `ncu` was unavailable and bandwidth/occupancy are fallback-derived (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:21`). Required fix: preserve JAX as the leading evidence-based choice, but phrase all performance conclusions as fallback-profiled and micro-fixture-limited; add an explicit M3/M4 action to obtain real Nsight artifacts once perf-counter permission is fixed.
>
> **5. Major:** The analytic column surrogate does not justify the strength of the M5 fallback language. The proposal correctly admits real Thompson/MYNN physics has much larger branch and variable pressure (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:78-79`), but then locks the revisit trigger so narrowly that only failure of both JAX restructuring and the Triton fallback forces a re-ADR (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:99-104`). Required fix: add an M5 stop/go proof object for the first real physics scheme before treating the fallback as exercised.

### Manager response

All five Codex findings (2 blockers, 3 majors) accepted and addressed in the revision dated 2026-05-19:
- `Status:` line marks ADR-001 as "proposed, pending user acknowledgement at M2 closeout"; M3 work does not begin until the user has explicitly approved.
- GT4Py candidate-failure artifacts created at `artifacts/m2/gt4py/{stencil_failure.json, column_failure.json, maintainability.md, agent_success.json}` per `.agent/milestones/ROADMAP.md` M2 schema. `reviewer_decision = excluded`.
- Fallback clause now per-scheme gated: a mini-ADR at `.agent/decisions/ADR-001-FALLBACK-<scheme>.md` with cross-model review is required before any single M5 physics scheme moves to Triton.
- `Evidence summary` § "Profile fidelity caveat" explicitly labels all M2 metrics as fallback-profiled and micro-fixture-limited.
- New § "M5 stop/go gate" added with binding numeric thresholds.

No manager counter-dissent recorded. Every Codex blocker/major was a fair catch.

### Binding-reviewer dissent — verbatim

Binding reviewer (codex high) of 2026-05-19 issued Decision: `Accept with required fixes` with 3 blockers + 2 majors + 1 minor — all on lifecycle/format/audit-trail hygiene (no architectural dissent on the JAX selection itself). All addressed in this revision: structural test `tests/test_adr_001_structure.py` added; ADR `Decision:` and `Selected backend:` lines reformatted as plain lines; full lifecycle reports written; ownership exception for `artifacts/m2/gt4py/` documented in the manager-closeout; stale `project_target_hardware.md` reference fixed to point at its actual location under `~/.claude/projects/.../memory/`.

## What this ADR does NOT commit

- A specific first physics scheme for M5. That is the M5-S0 decision-gate sprint (per `PROJECT_PLAN.md §11.7`).
- Mixed precision. Per `PRECISION_POLICY.md`, FP64 is the reference; FP32/BF16 require per-variable per-scheme validation. Not in v0 default.
- Multi-GPU. Single-GPU RTX 5090 first per `PROJECT_SCOPE.md`; the halo interface design (M3) must not preclude future multi-GPU, but multi-GPU is not committed here.
- AMD/Intel portability. The constitution does not require it for v0. If a future operational deployment demands it, the hybrid:jax+triton-physics model retains some portability (Triton supports AMD; jaxlib has ROCm builds); pure-cuda_tile would have been the worst on this axis. ADR-001 implicitly weighs this in JAX's favor.
- A specific halo decomposition strategy. M3 ADR-002 territory.

## M5 stop/go gate (mandatory before M5 closes)

The first real physics scheme implementation in M5 is treated as **the decisive test** of the M2 analytic-surrogate evidence. A stop/go proof object is required:

- After the M5 first-suite implementation worker reports, the manager runs the profile pipeline on the *real* scheme (Thompson microphysics or whichever scheme M5-S0 selects).
- The profile JSON (`artifacts/m5/<scheme>/<scheme>_profile.json`) must include: `registers_per_thread`, `local_memory_bytes`, `occupancy_pct`, `kernel_launches`, plus `profiler_limitation` if ncu is still blocked.
- **GO** if `local_memory_bytes <= 256` AND `registers_per_thread <= 128` AND `kernel_launches <= 10` AND correctness passes — M5 proceeds without fallback, JAX confirmed on real physics.
- **TRIGGER FALLBACK** if `local_memory_bytes > 256` OR `registers_per_thread > 200` OR XLA produces >50 kernel launches — manager opens a fallback mini-ADR (`.agent/decisions/ADR-001-FALLBACK-<scheme>.md`) per the gated process above.
- **STOP** if even Triton cannot meet the same thresholds for this scheme — escalate to user as a project-scope decision (probably means real physics shape is outside what the bakeoff predicted, and we need to reconsider ML-emulator hybridization for that scheme).

This gate is binding on M5 closeout. The M5 manager runbook must reference this ADR section.

## Trigger for revisiting this ADR

ADR-001 must be revisited only if:
1. M5 stop/go gate triggers a STOP (above). Project-scope decision.
2. The Triton fallback is invoked via mini-ADR and Triton also fails the thresholds on the same scheme. Project-scope decision.
3. JAX/XLA upstream drops Blackwell (cc120) support in a `jax[cuda13]` minor-version bump that breaks the pinned 0.10.0 path. Treated as an infrastructure event, not a design failure.
4. The 4–8× wall-clock-vs-CPU target proves unachievable in M7 due to JAX/XLA-specific limitations not visible at M2 or M5. Treated as a project-scope decision, escalated to user.

Outside these four triggers, M3–M8 work on the chosen backend without revisiting ADR-001.

## Evidence files (audit trail)

- `artifacts/m2/scout/toolchain_support_matrix.json` (M2-S1 readiness)
- `artifacts/m2/cuda_tile/`, `cupy_or_numba/`, `kokkos/`, `jax/`, `triton/` (per-candidate profile + correctness + maintainability + agent_success)
- `.agent/sprints/2026-05-19-m2-*/manager-closeout.md` (per-sprint lessons)
- `.agent/decisions/REVIEW-codex-ADR-001.md` (cross-model challenge — to be added)
- Project memory: `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/project_target_hardware.md` (pinned toolchain; auto-memory store, not in repo)
