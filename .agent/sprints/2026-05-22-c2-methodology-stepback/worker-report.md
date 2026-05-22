# Methodology Step-Back Review Report

**Reviewer**: Gemini 3.5 Flash (Methodology Step-Back Reviewer)  
**Date**: 2026-05-22  
**Scope**: Meta-review of `c1` iterations A1-A11, the `c1` Numerical Stability Spike, and the proposed `c2-A1`/`c2-A2` architecture and implementation.

---

## 1. Bottom Line
The 11-iteration `c1` struggle and the immediate `c2-A2` step 1 blowup prove that trying to build a "simplified/reduced" dycore and then adding terms reactively is a methodology failure. We must pivot from operator-bisection and simplified test cases to a **rigorous formulation-driven implementation of the full WRF-canonical split-explicit acoustic and vertical solver equations**. The dynamics solver must be verified against 1D hydrostatic rest and 2D Schär mountain wave benchmarks before attempting any coupled real-data forecasting.

---

## 2. Q1 — Warm-Bubble Discriminator
* **Verdict**: The flat warm-bubble test is a **necessary but insufficient** discriminator. It correctly identified that `c1`'s buoyancy was missing and that `c2-A2`'s vertical acoustic solver is incomplete (causing immediate step 1 NaN). However, it is blind to terrain-following coordinate errors (e.g., PGF slope truncation errors) and lateral boundary condition (LBC) mismatches.
* **Alternative**: Retain the warm-bubble test as the Tier-1 gate for flat dry dynamics. Add a **2D/3D Schär mountain wave test case** (Tier-2 gate) to validate the terrain metrics and the well-balanced PGF before running coupled forecast probes.
* **Cost Delta**: Implementing the Schär wave setup costs ~2 hours of scripting. It saves 12-24 hours of developer time spent chasing coupled domain blowups caused by steep volcanic slopes (Mount Teide) when the terrain PGF terms are mathematically incorrect.

---

## 3. Q2 — Bisection Methodology
* **Verdict**: Operator-level bisection (disabling terms one-by-one) is **ineffective and dangerous** at this stage. Disabling components in a tightly coupled non-hydrostatic solver breaks the physical-numerical balance, creating artificial instabilities and masking the true formulation flaws. The `c2-A2` step 1 blowup is a structural omission (missing vertical acoustic terms), which bisection cannot solve.
* **Alternative**: Transition to **physical invariant/conservation auditing** and **decoupled sub-component validation**. Run a 1D hydrostatic column test to verify that base-state/perturbation cancellation works perfectly at rest, and implement conservation checks (total mass, total dry static energy).
* **Cost Delta**: Writing a 1D hydrostatic rest test and mass/energy conservation checks costs ~3 hours. It saves 1-2 days of worker agents performing blind trial-and-error bisections on a broken codebase.

---

## 4. Q3 — WRF-Port Direction
* **Verdict**: The WRF-port split-explicit direction is the **fastest and only viable path** to a 3km operational forecast on a single RTX 5090. Since the regional boundaries are pre-determined by WRF Gen2 files, any deviation from WRF's grid nesting, hybrid-eta coordinate, or split-explicit acoustic formulation introduces boundary/coordinate interpolation mismatches that will cause immediate gravity-wave blowup at the boundaries.
* **Alternative**: Stop the "minimalist reduction" approach. Treat WRF's `dyn_em/module_small_step_em.F` and `module_advect_em.F` as the literal mathematical spec. Do not skip terms (such as vertical acoustic momentum/geopotential transport or the fourth non-hydrostatic PGF term) to save time, as these reductions are load-bearing for stability.
* **Cost Delta**: Direct full implementation of WRF-canonical acoustics in `c2-A2.x` avoids the "each fix exposes the next" loop, saving 3-5 days of sequential refactoring.

---

## 5. Q4 — Tool Gaps
* **Top 3 Missing Tools**:
  1. **Acoustic Substep Spatial Profile Plotter / NaN Locator**: An automated diagnostic that runs a single step, catches the NaN, and prints the 3D index, variable name, and vertical slice of the blowing-up field. (Estimated build cost: 1 hour).
  2. **1D Hydrostatic column rest test**: A script that runs the dycore in a 1D column with zero initial wind and verifies that vertical velocities remain at machine epsilon ($<10^{-14}\text{ m/s}$) for 1000s. (Estimated build cost: 2 hours).
  3. **Saved-state operator cross-validator**: A script that reads a saved WRF state and compares JAX-derived tendencies (PGF, advection) against WRF-derived tendencies for a single step. (Estimated build cost: 4 hours).
* **Total Build Cost**: ~7 hours.

---

## 6. What the Manager is Doing RIGHT
1. **Numerical Stability Spike Execution**: Postponing the `c2` implementation to run the flat vs. mountain and damping tests was an excellent decision that saved weeks of blind coding.
2. **ADR-020 Pytree Design**: Separating `State`, `BaseState`, `BoundaryState`, and `DycoreMetrics` is a superb JAX-native architecture pattern that prevents static recompilation storms.
3. **Strict Versioning and Proof Objects**: Demanding JSON proof artifacts for every sprint ensures that physics/performance claims are backed by code execution rather than model hallucination.

---

## 7. Open Questions for Full Council
* *Question 1*: How will the boundary relaxation (LBC) zone be implemented in JAX without triggering expensive out-of-place array allocations?
* *Question 2*: Should the precision policy (ADR-003) be modified to use mixed precision (FP64 for pressure/mass, FP32 for advection/physics) to utilize the RTX 5090 Tensor Cores, or does the stability risk of FP32 pressure solvers outweigh the performance gains?
