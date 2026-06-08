# Sprint Contract: v0.14 FP32 Acoustic De-Risk

Date: 2026-06-08
Manager branch: `worker/gpt/v013-close-manager`
Base release-candidate commit for workers: `b4901a907948623a7781b6cfc10fbbf42f64dd73`
Priority: highest v0.14 memory/performance lane; possible v0.13 only if evidence shows low risk.

## Objective

Determine whether the mixed perturbation-authoritative fp32 acoustic path can be moved from
roadmap to implementation with a small, proofable sequence of sprints. Do not block the active
fp64 TOST n=15 run. Produce evidence that lets the manager decide one of:

- promote a bounded default-inert subset into v0.13,
- keep it as v0.14 P1,
- or kill/reshape the lane because risk is too high.

## Lanes And File Ownership

1. `worker/gpt/v014-fp32-r0r1`
   - Owns: ADR/proposal files under `.agent/decisions/`, focused tests/proofs under
     `proofs/v014/`, and a narrow default-inert source prototype if needed.
   - Source prototype may touch only acoustic precision/base plumbing files needed for R0/R1.
   - Must not change default fp64 output or production CLI behavior.

2. `worker/gpt/v014-fp32-probes`
   - Owns: `proofs/v014/fp32_acoustic_*` and its report.
   - CPU-only numerical probes unless the manager explicitly authorizes a short GPU smoke.
   - No source code changes.

3. `worker/gpt/v014-fp32-roi`
   - Owns: report-only memory/ROI and 0.13-vs-v0.14 recommendation.
   - No source code changes.

## Hard Constraints

- The active fp64 TOST run is not delayed for FP32 GPU experiments.
- GPU may be used only by manager decision and only for tiny, short probes with low VRAM.
- No global fp32 dtype flip.
- No tolerance changes after results.
- No JAX-vs-JAX-only equivalence claim.
- No source merge into v0.13 without manager review, proof objects, and default-fp64 bit-inertness.

## Required Proof Standard

Minimum proof to consider any v0.13 pull-in:

- default production path unchanged and tested,
- no hidden host/device transfer inside timestep loops,
- source audit showing no mixed-mode perturbation is recovered by fp32 total-minus-base subtraction,
- CPU scalar/one-column proof for cancellation behavior,
- focused acoustic prep/finish or carry test,
- explicit list of fp64 islands retained.

## Deliverables

Each lane writes a handoff under `.agent/reviews/` with:

- objective
- files changed
- commands run
- proof objects produced
- v0.13 pull-in recommendation
- unresolved risks
- next decision needed

Final marker lines:

- `GPT FP32 R0R1 DONE`
- `GPT FP32 PROBES DONE`
- `GPT FP32 ROI DONE`
