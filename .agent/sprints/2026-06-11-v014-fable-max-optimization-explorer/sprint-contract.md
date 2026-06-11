# Sprint Contract: v0.14 → v0.15 Kernel Optimization Explorer (Fable 5 max)

Date: 2026-06-11 WEST
Owner: manager (Opus 4.8)
Assignee: Fable 5 max, dedicated worktree
Status: PREPARED. Dispatch ONLY after the venting fix has landed and the
Switzerland 72h gate has PASSED (principal directive: venting → 72h pass →
THEN this explorer). Supersedes and merges the two prepared prep sprints
`2026-06-11-v014-fable-performance-regression-audit/` and
`2026-06-10-v015-fable-kernel-efficiency-review/`.

## Why this sprint exists (the principal's worry, stated plainly)

Before this project began, the premise was that on THIS workstation (RTX 5090
Blackwell, 28-rank CPU-WRF baseline) a GPU port of WRF v4 should run the same
computation roughly **~10× faster** than the CPU, because regional NWP is a
large, finite, highly parallel stencil/physics computation and the GPU is built
for exactly that. Reality so far:

- an early, incomplete-dycore build measured ~3× (partly small-grid kernel-init
  overhead eating the advantage);
- after fixing memory bugs and runtime issues, the latest measured Canary L2
  d02 72h gate is only **1.059×–1.069×** vs an approximate 28-rank CPU
  denominator — i.e. roughly parity with the CPU.

Some of that may be residual debug code (which should not count), small-grid
overhead, or unfair measurement — but the principal is rightly worried the port
is currently too slow to be interesting. This sprint must explain the gap with
evidence and chart the realistic route back toward the ~10× premise for the
asymptotic large-grid regime.

## Objective

A complete computation + runtime efficiency analysis of the kernel and core
runtime, with TWO deliverable streams:

**Stream A — immediate safe fixes (commit now).** Fix, in this sprint, ONLY
changes that are *clearly and provably identity-preserving* against WRF v4 / the
current validated build. Allowed without further proof beyond an fp64
bit-identity / focused regression check:
- removal of debug/status/logging work from production hot paths (must stay
  available behind the M4 `debug: bool` static-arg convention, just not
  unconditional);
- removal of unnecessary host/device copies, redundant `.copy()`/materializations,
  and unnecessary synchronizations;
- removal of dead code, dead loops, and end-point calculations with no
  downstream consumer;
- hoisting stage-invariant or step-invariant recomputation out of the hot loop
  (e.g. quantities WRF treats as stage-constant);
- mathematically identical algebraic simplifications that XLA cannot do itself
  and that are bit-identical in fp64.
Each Stream-A change must carry a one-line identity justification and pass a
focused fp64 bit-identity check (or the relevant existing regression gate).
Keep Stream-A commits SMALL and separately reviewable.

**Stream B — ranked optimization document (no source edits).** Everything whose
gain depends on a change that could perturb WRF identity, alter numerics,
change precision, restructure kernels, or carries any non-trivial risk goes into
a complete ranked document for v0.15, NOT into code this sprint.

## Required analysis

1. **Runtime localization (where the time actually goes).** Profile a real
   forecast (use the GPU when the manager confirms it is free) and/or run
   component-disable experiments. Attribute wall-clock across: dycore/acoustic
   small steps, RK3, physics drivers (radiation/PBL/surface/microphysics),
   boundary/nesting/interp, halo/state copies, EOS/diagnostics, writer/IO,
   host↔device transfers, compile/cold-start, and orchestration/Python overhead.
   Separate **kernel-init/compile/small-grid overhead** from **steady-state
   per-step compute** so the asymptotic large-grid picture is visible. Profiler
   artifacts are MANDATORY for any compute claim (GPU kernel rules).
2. **Full code review** of the kernel stack and kernel-adjacent runtime for the
   inefficiencies feeding Streams A and B.
3. **Memory footprint audit vs CPU-WRF v4.** The principal believes peak memory
   is similar to original CPU-WRF v4 — verify with peak-VRAM / compiled-memory
   artifacts + a transfer audit. If GPU memory is much less efficient than
   CPU-WRF for the same grid, explain why. Do identity-certain memory reductions
   in Stream A; note significant-but-uncertain ones in Stream B.
4. **WHY-NOT-10×.** The headline question: if the sum of all found
   inefficiencies does NOT add up to ~10× on this system in the asymptotic
   large-grid / amortized-init regime, explain why, in evidence plus a simple
   intuitive explanation a non-specialist can follow. Cover the candidate
   ceilings: memory-bandwidth vs compute bound, kernel-launch/fusion overhead,
   scan/carry latency, Python/dispatch overhead, transfer overhead, occupancy,
   precision, algorithmic serial bottlenecks. State whether a kernel-architecture
   change (larger fusion boundaries, custom Triton/Pallas/CUDA kernels, persistent
   kernels, data-layout rewrite, physics batching, graph capture) could move the
   system toward computational near-optimum, with rough ceilings.
5. **Per-inefficiency estimates.** For each Stream-B item: expected speed gain
   (separately for (i) current small RTX 5090 cases, (ii) optimal RTX 5090
   in-VRAM grids, (iii) asymptotic H200/GB300 large-grid / cluster regime),
   implementation complexity, and risk (numerical, performance, implementation,
   validation).
6. **Minimum-sprint v0.15 implementation plan.** Order the Stream-B work to
   reach the most speedup in the fewest agent sprints with maximum safety for
   WRF identity. Specify, per group, exactly how to PROVE identity most
   efficiently — and plan the minimal set of identity-proof tools/harnesses to
   build (savepoint/comparator/profiler-diff/microbench) so the proof loops are
   cheap. Minimize the number of agent sprints; prefer a few large, well-gated
   sprints over many small ones.

## Priority and policy

- **Compute speed > memory.** When a high-value optimization trades memory for
  compute speed, prefer compute speed. If a caching/precompute/residency option
  has a significant memory footprint, design it as an OPTIONAL feature (default
  off, opt-in), not a forced cost.
- Start with the largest-impact items; do not spend the analysis on micro-gains.
- Known starting candidates (from `.agent/notes/2026-06-11-efficiency-notes-advance-w-lane.md`):
  per-substep `w_damp` placement, hot-path `safe_*` `jnp.where` floors,
  `t_2ave`/`mass_*` denominator hoist, Thomas-scan `unroll`, and the already-fixed
  per-stage diagnostics recompute. Confirm/extend these; do not stop there.

## Constraints

- Stream A only touches clearly identity-preserving code; ANY doubt → Stream B.
- No precision/mixed-fp changes in Stream A (fp64 default must stay bit-identical).
- No host/device transfer added inside timestep loops; document any found.
- No performance claim without a profiler/wall-clock artifact; no memory claim
  without a peak-VRAM/compiled-memory artifact + transfer audit.
- Work in the assigned worktree; commit Stream-A fixes + all artifacts there.
- One GPU job at a time; coordinate the GPU with the manager.

## Deliverables

- `.agent/reviews/2026-06-11-v014-fable-max-optimization-explorer.md`:
  verdict paragraph (is there major remaining headroom? what is the realistic
  asymptotic speed ceiling and why), the WHY-NOT-10× section, the ranked
  Stream-B table (rank, component/files, issue, gain class XL/L/M/S/XS per the 3
  machine regimes, complexity, risk, proof gates, v0.15 priority), top-5 memory
  + top-5 compute opportunities, the memory-vs-CPU-WRF audit, and the
  minimum-sprint v0.15 implementation + identity-proof-tooling plan.
- Profiler/benchmark artifacts under `proofs/v014/optimization_explorer/`.
- Stream-A commits on the worktree branch, each with its identity justification
  and focused-gate result.

## Acceptance gate (manager)

- Report exists, ranked by gain AND complexity across the 3 machine regimes,
  with the WHY-NOT-10× evidence-backed answer.
- Every Stream-A commit has an identity justification + a passing focused gate;
  no Stream-A change touches precision or dynamics semantics.
- Profiler/peak-VRAM artifacts present for all quantitative claims.
- No source change outside the Stream-A allowlist.

## Completion marker

Print exactly:
`FABLE V014_OPTIMIZATION_EXPLORER DONE - see .agent/reviews/2026-06-11-v014-fable-max-optimization-explorer.md`
