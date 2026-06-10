# Sprint Contract: V0.15 Fable Kernel Efficiency Review

Date: 2026-06-10
Owner: manager
Assignee: Fable/Mythos xhigh, tmux `0:1`
Status: PREPARED; dispatch only after the Canary h24 intermediate field compare
shows no unexpected divergence, or after v0.14 validation otherwise no longer
needs Fable for correctness debugging.

## Objective

Perform a complete read-only memory and compute efficiency review of the
wrf_gpu2 kernel stack and kernel-adjacent components, producing a ranked v0.15
action list. Do not modify source code.

The project goal is a WRF-faithful-enough, GPU-optimized, near compute- and
memory-optimal, scalable GPU rewrite. The review must find real efficiency
opportunities without weakening WRF fidelity, validation, or GPU-native design.

## Scope

Review at least:

- dycore and acoustic small-step kernels;
- RK3 integration, state/carry layout, scan boundaries, donation/liveness;
- pressure/thermodynamic diagnostics and base/perturbation handling;
- radiation and RRTMG tiling/chunking;
- PBL/surface-layer/NoahMP/coupling memory flow;
- microphysics and moisture advection/limiters;
- boundary/live-nesting, interpolation, feedback, halo/state copies;
- wrfout writer and validation-runner IO where it affects memory/compute;
- precision contracts, especially the deferred FP32 acoustic lane;
- multi-GPU/sharding and AOT/precompile opportunities.

## Inputs

Read only the files needed, starting with:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/decisions/V0150-ROADMAP-DRAFT.md`
- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `proofs/v014/exact_branch_memory_preflight.md`
- `proofs/v014/mythos_memory_fixes_260609.md` if present
- `src/gpuwrf/**` as needed
- relevant `proofs/v013` and `proofs/v014` memory/performance artifacts

## Required Output

Write exactly one primary report:

`/.agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`

The report must be concise enough for a manager context window and include:

1. Verdict paragraph: is there likely major remaining efficiency headroom?
2. Ranked table of candidates. Columns:
   - rank;
   - component/files;
   - issue/opportunity;
   - gain class: `XL`, `L`, `M`, `S`, or `XS`;
   - complexity class: `XL`, `L`, `M`, `S`, or `XS`;
   - risk: numerical, performance, implementation, validation;
   - required proof gates;
   - recommended v0.15 priority.
3. Separate top-5 memory opportunities.
4. Separate top-5 compute opportunities.
5. FP32 acoustic feasibility/update: whether it remains the highest-value v0.15
   lane, and what exact first implementation sprint should do.
6. Items explicitly rejected as low value or too risky for v0.15.
7. Context-sparing manager handoff, max 12 bullets.

## Constraints

- No source edits, no formatting edits, no generated cleanup.
- No GPU use unless the manager explicitly says the GPU is free.
- CPU-only probes are allowed if quick and readonly; use `JAX_PLATFORMS=cpu`
  and `CUDA_VISIBLE_DEVICES=`.
- Do not spend tokens on routine status. Produce the report and print:
  `FABLE V015_KERNEL_EFFICIENCY_REVIEW DONE - see .agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`
- If the task is too broad, still produce the best ranked action list; do not
  ask for a micro-prompt.

## Acceptance Gate

Manager acceptance requires:

- report exists at the required path;
- recommendations are ranked by both gain and complexity;
- every high-gain recommendation includes concrete files/components and proof
  gates;
- no source files changed.
