# ADR-001 — Backend Selection for the GPU-Native NWP Core

Date: 2026-05-19
Author: manager (Claude Opus 4.7 1M context)
Status: **proposed**; awaiting Codex `gpt-5.5 xhigh` critical-review and reviewer binding judgment before merge into main
Scope: v0 (single-node RTX 5090, cc120 Blackwell, 32 GB VRAM)
Reversibility: **irreversible** per `.agent/rules/architecture-decision-policy.md`; per the manager-autonomy directive of 2026-05-19 the manager exercises the irreversible-decision authority and reports to the user at M2 closeout.

## Decision

**Selected backend: jax**

Primary backend for the M3 device-resident `State`/`GridSpec`, M4 minimal dycore, and M5 first physics suite: **JAX + XLA**, Python orchestration, `@jit` per kernel-class, AOT compile via `jit`. Pin: `jax[cuda13]==0.10.0` (per M2-S1 scout matrix, hello-GPU verified on this workstation).

**Pre-authorized fallback (no new ADR required if triggered):** if M5 implementation reveals XLA register-spilling on a real WRF physics scheme (Thompson microphysics, MYNN PBL, or equivalent column-heavy kernel) that cannot be remedied by JAX-side restructuring (vmap rearrangement, custom_jvp, or explicit `jax.lax.scan` pattern), the affected physics scheme MAY be re-implemented in **OpenAI Triton** (`@triton.jit`, pin `triton==3.7.0 + torch==2.12.0`) and called from the JAX timestep loop via `jax.experimental.pallas` or a thin ctypes shim. This forms a `hybrid:jax+triton-physics` model. The constitution's "scope expansion requires human approval" rule does NOT trigger here — this fallback is the explicit M2 conclusion that the user is consulted on now via M2 closeout, not later.

## Evidence summary

M2 bakeoff measured five backend candidate families against the M1 analytic stencil and analytic column fixtures on RTX 5090 (cc120). The sixth (gt4py) was excluded by the M2-S1 scout — `gt4py + DaCe 0.10.0` fails on Python 3.13 with a SymPy break; remediation requires a Python 3.12 venv and is deferred to a post-M2 sprint if M5 reveals a need for a stencil DSL beyond what JAX offers.

All numbers below are from `cuobjdump --dump-sass` (registers + local_memory_bytes), the CUDA occupancy API (occupancy_pct, theoretical), and bench-output wall_time. NVIDIA's `ncu` (Nsight Compute) was unavailable due to a system-level `ERR_NVGPUCTRPERM` constraint — `nvidia-driver-perfmon-allow=1` is not set on this workstation. The `profiler_limitation` field in every candidate's profile JSON documents this; `achieved_bandwidth_gbps` is labeled `fallback-derived` across all candidates and is order-of-magnitude only.

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

To be populated from `REVIEW-codex-ADR-001/critical-review.md` after the Codex `gpt-5.5 xhigh` critical-review runs. Per `.agent/rules/cross-model-review-policy.md`, dissent is preserved verbatim — not paraphrased, not omitted.

If the critical-review proposes a different selected backend (e.g. hybrid up-front, or pure Triton), the manager will either revise the Decision section to match OR record the disagreement here with explicit rationale for not adopting it.

**Placeholder until critical-review returns:** none recorded.

## What this ADR does NOT commit

- A specific first physics scheme for M5. That is the M5-S0 decision-gate sprint (per `PROJECT_PLAN.md §11.7`).
- Mixed precision. Per `PRECISION_POLICY.md`, FP64 is the reference; FP32/BF16 require per-variable per-scheme validation. Not in v0 default.
- Multi-GPU. Single-GPU RTX 5090 first per `PROJECT_SCOPE.md`; the halo interface design (M3) must not preclude future multi-GPU, but multi-GPU is not committed here.
- AMD/Intel portability. The constitution does not require it for v0. If a future operational deployment demands it, the hybrid:jax+triton-physics model retains some portability (Triton supports AMD; jaxlib has ROCm builds); pure-cuda_tile would have been the worst on this axis. ADR-001 implicitly weighs this in JAX's favor.
- A specific halo decomposition strategy. M3 ADR-002 territory.

## Trigger for revisiting this ADR

ADR-001 must be revisited only if:
1. M5 implementation reveals XLA register-spilling on a real physics scheme AND JAX-side restructuring cannot remedy it AND the Triton-physics fallback cannot be made to fit the M3 state/halo contracts. The pre-authorized fallback handles the first two; only the third would force a re-ADR.
2. JAX/XLA upstream drops Blackwell (cc120) support in a `jax[cuda13]` minor-version bump that breaks the pinned 0.10.0 path. Treated as an infrastructure event, not a design failure.
3. The 4–8× wall-clock-vs-CPU target proves unachievable in M7 due to JAX/XLA-specific limitations not visible at M2 or M5. Treated as a project-scope decision, escalated to user.

Outside these three triggers, M3–M8 work on the chosen backend without revisiting ADR-001.

## Evidence files (audit trail)

- `artifacts/m2/scout/toolchain_support_matrix.json` (M2-S1 readiness)
- `artifacts/m2/cuda_tile/`, `cupy_or_numba/`, `kokkos/`, `jax/`, `triton/` (per-candidate profile + correctness + maintainability + agent_success)
- `.agent/sprints/2026-05-19-m2-*/manager-closeout.md` (per-sprint lessons)
- `.agent/decisions/REVIEW-codex-ADR-001.md` (cross-model challenge — to be added)
- `project memory: project_target_hardware.md` (pinned toolchain)
