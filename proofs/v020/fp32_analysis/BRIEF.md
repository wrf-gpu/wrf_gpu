# DEEP ANALYSIS BRIEF — v0.20.0 fp32/mixed-precision kernel rewrite (POINT 1)

You are ONE of TWO independent analysts (the other is the other model). Produce an
INDEPENDENT, in-depth strategic analysis + step-by-step implementation & validation
plan for the **fp32 / mixed-precision kernel rewrite** that finally makes the GPU
dycore **MASSIVELY faster** — the proven ~4x "compact-explicit-fp64-island" path —
NOT a minimal speedup where parts stay fp64 because of tiny numeric differences.
This is a PLANNING task (GPU unavailable tonight): no GPU runs, deep analysis + plan.

## THE MISSION (the user, exact)
"This time we really end up with a massively faster kernel and not just minimal
speedups because parts remain in fp64 due to tiny differences." Your plan must show,
step by step, HOW to keep the fp64 islands TINY so the bulk runs fp32 (→ ~4x), and
HOW to validate each step. HARD CONSTRAINT: the **nested all-7 max_dom=9 case must
stay FULLY FUNCTIONAL** with every kernel change — it is the benchmark + test case.

## CURRENT STATE
v0.19.1 shipped: fused nested all-7 max_dom=9 = 720 s/fc-h vs 12-rank CPU 1020 =
1.41x faster, 24h-VRAM-stable, all-fields tolerance-green. Bottleneck for further
speed = the fp64 dycore on consumer Blackwell (RTX 5090, fp64 = 1/64 of fp32).
Validation tolerance = the FROZEN grid-delta manifest (proofs/v014/grid_delta_atlas/
tolerance_manifest_candidate.json); nested grid-compare via scripts/compare_wrfout_grid.py.

## READ FIRST (historic work — do NOT repeat past mistakes; build on proven findings)
- .agent/decisions/KERNEL-OPTIMIZATION-FINDINGS-FINAL.md
- .agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md
- .agent/decisions/ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md
- .agent/decisions/ADR-003-dycore-precision.md, ADR-007-precision-policy.md
- .agent/decisions/2026-06-04-v0100-kernel-optimization-superplan.md
- PRECISION_POLICY.md  and  FP32_GRADIENT_PROBLEM_OPEN_CHALLENGE.md  (the core open problem)
- proofs/perf/fp32_downcast_plan.md, proofs/perf/fp32_downcast_spec.md
- proofs/v018/k1_fp32_stability_scoping.md
- proofs/thompson_perf/kernel_lever_summary.json, proofs/thompson_perf/fp32_vs_fp64_oracle_diff.json
- .agent/dispatches/2026-06-08-gpt-fp32-*.prompt.md (past fp32 probe/ROI dispatches + outcomes)
- scripts/precision_bench.py, tests/test_fp32_divergence_growth_metric.py, tests/test_m6_precision_matrix.py
KNOWN PRIOR VERDICT (confirm/challenge it with the docs): naive global-fp32 is INVALID;
auto-promote ceiling ~1.1x; the COMPACT-EXPLICIT-fp64-ISLAND rewrite is the proven ~4x
(+~2x VRAM) path; 4.3x measured was invalid global-fp32; ~6x exceeds the roofline; kernel
fusion gave ~0%. WHY did prior attempts stall at minimal speedup? Diagnose that root cause.

## DELIVER (your report)
a. STRATEGIC ANALYSIS: the core reason naive fp32 yields only ~1.1x (the "tiny differences
   cascade -> instability -> parts forced back to fp64" trap). Which dycore/physics parts
   GENUINELY require fp64 (and why — conditioning, cancellation, hydrostatic balance, long
   time integration) vs which can be fp32. Define the fp64-ISLAND boundaries precisely.
b. STEP-BY-STEP IMPLEMENTATION PLAN: an incremental, each-step-validated sequence. Per
   component (RK3 dynamics, acoustic substeps, advection, pressure-gradient/hydrostatic,
   diffusion, microphysics, radiation, PBL/surface, the nested boundary/feedback path):
   the precision decision, the conversion boundaries (where to cast), and WHY. Order the
   steps so each is independently testable and reversible. Make the fp64 islands MINIMAL.
c. VALIDATION / PROBES / TOOLS to MAXIMIZE success: how to test each step keeps nested
   all-7 tolerance-green (grid-compare + the frozen manifest); the fp32-divergence-growth
   metric; savepoint/oracle diffs; early-divergence localization probes; per-variable
   precision A/B; a roofline/speedup probe to confirm each step actually moves the needle.
   Specify the exact gates (tolerance, divergence rate, speed delta) per step.
d. ROADBLOCKS requiring MATHEMATICAL / COMPUTATIONAL CREATIVITY -> concrete IDEAS for each.
   Especially the tiny-difference problem: e.g. perturbation-form fp32 (subtract a fp64
   reference/base state so fp32 carries only the small perturbation), compensated/Kahan or
   error-feedback summation, fp64 residual-correction islands, mixed-precision RK with fp64
   slow-tendency accumulation, deterministic/stable reductions, stochastic rounding,
   rescaling/non-dimensionalization for conditioning, per-column vs global precision. Rank
   the ideas by expected payoff vs risk.
e. THE MASSIVE-SPEEDUP GUARANTEE: explicitly argue how this plan avoids the "minimal
   speedup" failure mode (keep fp64 islands tiny; quantify the expected speedup per step;
   identify the few mandatory-fp64 ops and bound their cost).
f. RISKS + MITIGATIONS, and how every change preserves NESTED all-7 functionality.

## OUTPUT
Write your full report to your assigned file (you will be told OPUS_FP32_PLAN.md or
GPT_FP32_PLAN.md) under proofs/v020/fp32_analysis/. Be concrete, cite file:line / specific
kernels / specific docs. This feeds the manager's synthesis -> the v0.20.0 roadmap, which a
GPT critic will then review. NO GPU runs; NO source edits; analysis + plan only.

## THE USER PRECISION-ACCEPTANCE POLICY (2026-06-21 — overrides any tighter prior policy)

This is the key that unlocks the massive speedup: the prior ~1.1x fp32 ceiling came from
demanding tight tolerance vs WRFv4. **We now ACCEPT LESS-TIGHT margins vs WRFv4.**
- If fp32 brings the huge gains (it SHOULD) — good. **If NOT, find the EXACT PROVABLE
  reason (no hand-waving) — analyze until a provable computational/mathematical cause is
  pinned.**
- The exact tolerance tightness is the analysts'/manager's call — choose a DEFENSIBLE level.
- HARD GATES (non-negotiable):
  1. **No variable explodes / blows up** (NaN / unphysical) — EXCEPT known cumulative
     variables and ones like `QVAPOR` we already have documented issues with.
  2. **Wind + cloud-related layers (and temperature) stay STABLE** at a reasonable margin
     that will NOT degrade the **24-120h forecast skill** of wind / temp / cloud.
- **Prioritize the massive speedup**: do NOT keep parts in fp64 merely for tiny
  differences. Keep fp64 ONLY where genuinely required for stability or for wind/cloud/temp
  24-120h skill.
- Therefore the PLAN must: (i) define concrete per-variable fp32 acceptance gates (with the
  cumulative/QVAPOR carve-outs + the wind/cloud/temp 24-120h-skill priority); (ii) design
  VALIDATION around 24-120h forecast skill of wind/temp/cloud + stability (no blow-up),
  NOT bit-tightness vs WRFv4; (iii) keep nested all-7 max_dom=9 the functional benchmark.
