# v0.22 Roadmap Critic Report

Date: 2026-06-26
Mode: read-only, CPU-only. No repo files edited. No GPU/profiler commands run.

## Verdict

**REVISE before using this roadmap as an execution contract.**

I do **not** find broad ruled-out contamination in K1-K6: the roadmap correctly rejects whole-step CUDA graph/count-only wins, scan-unroll-as-bankable, conditional-WHILE early exit, hoisting, radiation-beyond-radt, and 6-10x large-grid saturation. The binding DEVICE-bound / count-cuts-wall-neutral framing is mostly applied.

The blocking problems are narrower but material: R0's gate does not match the user's zero-regression question, the roadmap overstates a shared compile-pathology root, lossless/tiered gates omit explicit conservation/transfer/restart guards, and the operational-relaxed tier is not yet frozen tightly enough to prevent silent fidelity loss.

**Single most important correction:** split the "shared compile-pathology crack" into separately falsifiable hypotheses for (a) O(nz) BouLac data-dependent search, (b) fp32-BouLac mixed fp32/fp64 stall, and (c) fused 9-nest cold-compile RAM; do not claim one fix or fp32-operational state unblocks all three until measured.

## Blocking Findings

### B1 - R0 zero-runtime-regression gate is too loose and overclaims "risk = 0"

Evidence:
- `.agent/decisions/V022-ROADMAP.md` R0 asks for low-RAM compile + AOT + "runtime == fused ~1.376 s/step" and says flag sweep "does not change the runtime executable" / "perf risk = 0 by construction" (lines 13-14, 36-49).
- The same gate accepts `s/step == fused 1.376 +/- 5%` (lines 56-58).

Why this blocks:
- A 5% slowdown violates the stated "0% perf loss" requirement.
- Memory-fitting, optimization-level, remat, scheduler, and XLA flags can change HLO, fusion, rematerialization, binary layout, runtime, and last-bit numerics. They are not zero-risk by construction; they are safe only if the gate proves safety.

Concrete fix:
- Replace the R0 pass condition with "no statistically significant regression vs same-day fused baseline; hard fail if slower by >1-2% or outside predeclared measurement noise."
- Change "does not change the runtime executable" to "intended to preserve the fused one-executable execution model; must prove same/no-worse runtime, valid AOT blob, and Tier-S bitwise or Tier-P field identity depending on whether compiler flags alter numerics."

### B2 - "Shared compile pathology" is an unproven root-cause claim

Evidence:
- V022 says K1's O(nz) BouLac blocker is the "same root as fp32-BouLac" and that K4/fp32-operational is the "single fix that unblocks BOTH K1's compile pathology AND the 1 km capability/VRAM-arena story" (lines 82-88, 101-107, 159-162).
- `proofs/perf/v015/boulac_nz_optimization.json` describes O(nz) BouLac as a data-dependent first-crossing search / scan-or-unrolled stateful iteration that triggers slow compile in the full pipeline.
- `proofs/perf/v015/fp32_definitive_verdict.json` describes fp32-BouLac as mixed fp32/fp64 BouLac fusion that compiles slowly then stalls.

Why this blocks:
- The artifacts support a shared symptom/class ("XLA Very slow compile in full operational jit"), not one proven root.
- ADR-031/fp32-operational plausibly removes the mixed-precision fp32-BouLac pathology and fp64 arena pressure, but it does not by itself prove the O(nz) data-dependent search pathology disappears.

Concrete fix:
- Rewrite sequence step 1 as a compile-pathology diagnostic suite with three independent kill gates: O(nz) BouLac full-pipeline compile/run, fp32-BouLac compile/run, and fused-cascade cold-RSS/R0.
- Keep "stays opt-in" language for K1 and fp32-BouLac, but strengthen it to "may remain permanently opt-in/closed unless its own full-pipeline gate passes."
- Do not describe fp32-operational as unblocking K1 until a measured O(nz)+fp32-operational full-pipeline compile/run proves it.

### B3 - Several gates omit mandatory conservation / transfer / restart guards

Evidence:
- V022 globally requires canary s/step + VRAM/RAM and no clamps (lines 3-6), but per-lever gates vary.
- K1 gate has compile, tiered identity, 24h stability, d02 skill, and ms reprofiling, but no explicit conservation/transfer/restart check (lines 82-88).
- K2 gate has CFL ladder, finite detector, canary s/step, multi-day stability, and tiered identity, but no explicit conservation, boundary-ring growth, restart/output cadence, or CPU-WRF same-config comparator (lines 89-93).
- K5 gate has corpus oracle, tiered identity, warmed A/B, but no explicit conservation or no-clip/limiter hit accounting (lines 108-113).

Why this blocks:
- The prompt explicitly requires canary s/step plus identity/skill plus conservation.
- K2 changes timestep/substep cadence, so "tiered identity" alone is not a sufficient physical gate over multi-day forecasts; it needs conservation and phase/restart/output checks.

Concrete fix:
- Add a standard gate footer to every K1/K2/K5 ship candidate: canary s/step, peak VRAM/RSS, warmed transfer audit with no in-loop H2D/D2H regression, finite detector, dry-mass/water/energy conservation budgets, restart/output equivalence, and no masking-clamp/limiter hit increase.
- For K2 specifically, require CPU-WRF or strict-GPU same-namelist comparator where applicable, plus boundary-ring and steep-terrain diagnostics before any default change.

### B4 - Operational-relaxed tier is directionally right but not yet release-safe

Evidence:
- V022 defines operational-relaxed as 24/72/120h skill bands, hard stability/conservation guards, no silent clamps/fallbacks, the user sign-off, and opt-in (lines 132-143).

Why this blocks:
- "A stated band" is not enough for an execution contract. Without frozen cases, variables, baselines, pass/fail thresholds, and proof schema before implementation, K3/K6 can silently become goalpost-moving.
- The tier says no silent clamps/fallbacks, but does not require static/dynamic clamp scans, limiter-hit counters, or explicit comparison to strict-WRF and observations/AEMET/AIFS where available.

Concrete fix:
- Make operational-relaxed work item 0 produce a frozen ADR/contract before K3/K6 implementation: cases, lead times, variables, station/grid/object metrics, baselines, thresholds, conservation budgets, clamp/fallback audits, profiler proof, and exact the user sign-off checkpoint.
- Require "opt-in only" in code/config/docs and forbid using operational-relaxed evidence to mutate default WRF-faithful behavior.

## Nice-To-Have Findings

### N1 - Split WRF-faithful cadence semantics from non-lossless cadence

Evidence:
- GPT deep-dive recommends "first WRF-faithful positive `bldt`/`cudt` semantics, then a separate non-lossless PBL/surface/Noah cadence ladder."
- V022 treats K3 as non-lossless PBL `bldt` cadence and skips cumulus `cudt` because default cumulus is off (lines 94-100, 118-119).

Concrete fix:
- Add a small K3a item: if positive `bldt`/`cudt` namelists are supported, match WRF cadence exactly with CPU-WRF fixtures. Keep K3b for fast-mode cadence beyond default/strict WRF under operational-relaxed.

### N2 - Reconcile carried `#101 async history output`

Evidence:
- V022 carries `#101 async history output` as likely independently shippable (line 151).
- The ruled-out briefing also lists async history output among v0.20 low-hanging levers already shipped (lines 36-39), while later listing it as open.

Concrete fix:
- Mark #101 as one of: shipped/default, shipped opt-in needing docs, or genuinely pending. Do not let it reappear as a fresh speed lever without current proof.

### N3 - Add a low-risk profiler hygiene audit bucket, not a roadmap promise

Evidence:
- GPT deep-dive flagged possible `RedzoneAllocatorKernelImpl` / debug allocator residue and layout/slice cleanup as low-single-digit or zero-risk audits.

Concrete fix:
- Add a "cheap hygiene audit" appendix with strict exit criteria: if production-env profile shows no such family or no wall delta, close immediately. Do not promote these to K-level levers.

### N4 - K4 is correctly out of v0.22; only scope/ADR should land

Evidence:
- V022 calls fp32-operational a separate major milestone and not a v0.22 add-on (lines 101-107, 170-171).

Concrete fix:
- Keep K4 out of v0.22 implementation. At most, v0.22 may produce an ADR-031 execution plan and proof inventory; no partial default precision change without its own milestone gates.

## Sequencing Assessment

- **K2 early is correct.** It is independent of compile work and is the cleanest lossless runtime ladder, but its gate must include conservation/restart/boundary diagnostics.
- **Compile pathology first is reasonable but overcoupled.** Run it first as a diagnostic suite, not as a presumed shared crack.
- **K3 does need the operational-relaxed tier first if it is non-lossless.** The WRF-faithful positive-`bldt`/`cudt` semantics, if missing, can be split out and done without waiting for the non-lossless tier.
- **K4 should remain its own milestone.** No v0.22 piece should land except planning/ADR/proof inventory.

## Ruled-Out Contamination Check

No direct blocking contamination found in the main K1-K6 list. The roadmap does not re-propose these as wins:
- whole-step CUDA graph / command-buffer launch-count cuts;
- scan-unroll knobs as a banked acoustic win;
- conditional-WHILE/data-dependent early exit;
- stage-constant/denominator hoist;
- radiation cadence beyond `radt`;
- global fp32 / precision-only nest speedup;
- 6-10x large-grid saturation.

The nearest contamination risks are wording-level, not lever-level:
- R0 implies compiler flags preserve the same runtime executable by construction.
- The compile-pathology wording implies one root/fix across O(nz), fp32-BouLac, and R0 without proof.

## Handoff

- Objective: adversarial critic of `.agent/decisions/V022-ROADMAP.md` against authoritative kernel findings and v0.22 deep-dives.
- Files changed: `/tmp/v022_roadmap_critic_REPORT.md` only; repo untouched.
- Commands run: read-only `sed`, `nl`, `rg`, `find`, `python -m json.tool`, `git status --short`, `tmux list-panes`, `date`.
- Proof objects produced: this report.
- Unresolved risks: no new GPU measurements because task was CPU-only and GPU busy; critique is based on existing artifacts.
- Next decision needed: manager should revise V022 before phase-5 execution, especially B1/B2 gates and compile-pathology wording.
