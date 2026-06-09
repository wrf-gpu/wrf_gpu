# V0.14 Early-Step Same-Input Discriminator Plan

Date: 2026-06-09
Status: executed 2026-06-09; closed fail-closed, see
`proofs/v014/early_step_discriminator.*`

Update 2026-06-09 12:45 WEST: execution verdict is
`EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.
The sprint covered steps `1`, `60`, `600`, `3000`, and `5999`, but no strict
same-input comparison ran. Next plan is not another discriminator attempt; build
the missing comparison contract/tooling first.

## Why

The v0.14 grid-parity investigation is correctly grid-first, but the h10 /
step-6000 same-input path has become a blocked instrumentation ladder. Opus
management review `2026-06-09-v014-management-review-01.md` recommends
bisecting from the clean end where instrumentation is cheaper.

This plan has now been executed. Keep it as the record of the intended
discriminator and use the closeout proof to seed the contract-builder sprint.

## Objective

Run a consolidated early-step same-input discriminator from shared `wrfinput`
state and locate the first strict divergence window. The sprint must execute at
least one strict same-input comparison, or name all remaining blockers in one
proof pass.

## Scope

- CPU first. GPU only later for symptom confirmation, not inside the first
  discriminator unless the manager explicitly records why CPU is insufficient.
- No production `src/gpuwrf/**` edits.
- No TOST, Switzerland, FP32, or memory source work.
- Use existing replay loaders and WRF scratch hooks where possible.

## Candidate Steps

Evaluate the smallest useful sequence, adapting if the first strict comparison
already diverges:

- step 1
- step 60
- step 600
- step 3000
- step 5999

The step selector must use dynamic/perturbation fields for headline decisions;
exclude static writer/base artifacts from the dominant-field selector.

## Required Proof Surface

For each executed candidate step, capture or construct:

- WRF-controlled initial state for `T/P/PB/PH/PHB/MU/MUB/U/V/W` plus active
  moisture needed by the selected JAX entry point.
- WRF-controlled tendencies or a proof that the selected entry point computes
  them from the same state.
- Matching WRF post-RK/pre-halo truth.
- JAX post-RK/pre-halo result from the same input.
- Per-field count, max_abs, RMSE, bias, p95, p99, and ranked residuals.

## Verdicts

The next sprint should emit exactly one of:

- `EARLY_STEP_DYNAMICS_CLEAN_THROUGH_<step>`
- `FIRST_DIVERGENT_STEP_<step>_<field_or_operator>`
- `EARLY_STEP_DISCRIMINATOR_BLOCKED_<all_blockers_named>`

No more one-blocker micro-sprint verdicts are acceptable for this stage.

## Acceptance Gate

- At least one strict same-input comparison executes and writes JSON/Markdown,
  or one proof names all blockers preventing execution across the early-step
  sequence.
- `git diff -- src/gpuwrf` is empty.
- JSON validates with `python -m json.tool`.
- The top-level Markdown is context-sparing; detailed field tables are in JSON
  or CSV.

## Follow-On

If the discriminator finds a mismatch, open one source-edit sprint scoped only to
the named operator/step. If it stays clean through a useful early window, open a
producer/handoff trace sprint to explain how the trajectory drifts before h10.
