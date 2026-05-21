# M6-S6 Worker Report - Tier-3 TSC1.0 Drift Envelope

## Objective

Implement the M6-S6 Tier-3 controlled timestep-sensitivity envelope for coupled surface variables and close the two M6-S4 tautology follow-ups that were assigned to this sprint:

- F-min-1: Thompson water-budget residual must use an independent tendency side-channel instead of precipitation/update self-accounting.
- F-min-2: boundary closure must have an independent wrfbdy decoder path instead of comparing `boundary_apply` output to its own reconstructed tendency.

The sprint contract required `U10`, `V10`, `T2`, `qv2`, and `precip` at +6/+12/+24h, controlled dt refinement at 18s/9s/4.5s, per-variable/per-lead status, and `artifacts/m6/tier3/tsc_envelope.json`.

## Outcome

Status: **BLOCKED for AC3**, implemented for the code/schema/oracle portions.

The reduced controlled TSC1.0 envelope was generated at the required +6/+12/+24h leads for dt=18/9/4.5. The generated artifact is:

- `artifacts/m6/tier3/tsc_envelope.json`

The pinned d02 drift comparison at dt=18s was attempted, but the run hit CUDA OOM during the pinned d02 segment after the reduced TSC work had completed. The artifact records `status=BLOCKED` and carries an explicit `d02_drift_blocker` message. The Tier-3 gate correctly fails rather than accepting an incomplete AC3 proof.

## Files Changed

- `src/gpuwrf/validation/tier3_coupled.py`
  - New TSC1.0 module.
  - Builds the reduced idealized coupled case.
  - Runs dt=18/9/4.5 controlled refinement.
  - Computes `max(|F18-F9|, |F9-F4.5|)` envelopes per variable and lead.
  - Computes per-variable/per-lead GREEN/PARTIAL/FAIL/BLOCKED status.
  - Contains Thompson and wrfbdy oracle probes used in the artifact.
- `src/gpuwrf/coupling/physics_couplers.py`
  - Added `ThompsonTendencySideChannel`.
  - Extended `thompson_adapter(..., return_tendencies=True)` without changing existing driver call behavior.
  - Added `thompson_adapter_with_tendencies` for validation callers.
- `src/gpuwrf/validation/tier2_coupled.py`
  - `water_budget_residual` now accepts a Thompson side-channel oracle and compares observed water delta against oracle tendency.
  - Existing accumulator/delta behavior remains available for existing callers.
- `src/gpuwrf/io/boundary_replay.py`
  - Added wrfbdy decoder for `_B*` and `_BT*` lateral-boundary variables.
  - Added `compare_boundary_tendency_to_wrfbdy` for independent GPU tendency vs wrfbdy tendency comparison.
- `src/gpuwrf/io/proof_schemas.py`
  - Expanded `Tier3DriftEnvelope` schema to require per-variable/per-lead envelope, drift, forcing, boundary, regridding, and status fields.
- `scripts/m6_run_tsc.py`
  - New runner for generating the Tier-3 artifact.
  - Supports full run, `--skip-d02`, lead override for smoke testing, radiation cadence, and explicit blocked reason.
- `scripts/m6_gate_tier3.py`
  - New gate script. It fails on the produced artifact because status is BLOCKED.
- `tests/test_m6_tier3_tsc.py`
  - New focused tests for raw envelope calculation, per-lead status, Thompson side-channel residual, wrfbdy decoder comparison, and schema requirements.
- `artifacts/m6/tier3/tsc_envelope.json`
  - Generated proof object. This path is ignored by the repo's global `artifacts/*` ignore rule, but the file exists in the worktree.

## Commands Run

- `pytest -q tests/test_m6_tier3_tsc.py`
  - Result: `5 passed in 4.33s`
- `python scripts/m6_run_tsc.py --skip-d02 --lead-hours 0.005 --output /tmp/tsc_skip.json`
  - Result: `status=BLOCKED`, smoke artifact generated.
- `python scripts/m6_run_tsc.py --radiation-cadence-s 86400 --output artifacts/m6/tier3/tsc_envelope.json`
  - Result: stopped after pinned d02 segment hit repeated `CUDA_ERROR_OUT_OF_MEMORY` allocations up to 8 GiB.
- `python scripts/m6_run_tsc.py --skip-d02 --radiation-cadence-s 86400 --blocked-reason 'Pinned d02 dt=18s +6/+12/+24 run was attempted on 2026-05-21 but hit CUDA OOM during the d02 segment after reduced TSC completed; raw stderr showed repeated CUDA_ERROR_OUT_OF_MEMORY allocations up to 8 GiB.' --output artifacts/m6/tier3/tsc_envelope.json`
  - Result: `status=BLOCKED`, full reduced +6/+12/+24 envelope artifact written.
- `PYTHONPATH=src python - <<'PY' ... Tier3DriftEnvelope.validate_file(...) ... PY`
  - Result: `schema ok`
- `python scripts/m6_gate_tier3.py --artifact artifacts/m6/tier3/tsc_envelope.json`
  - Result: expected failure, `Tier-3 gate failed: status=BLOCKED`, all d02 drift leads BLOCKED.
- `pytest -q tests/test_m6_tier3_tsc.py tests/test_m6_proof_schemas.py`
  - Result: `8 passed in 4.69s`
- `pytest -q tests/test_m6_boundary_replay.py tests/test_m6_tier2_coupled.py`
  - Result: `6 passed in 20.51s`

## Proof Objects Produced

- `artifacts/m6/tier3/tsc_envelope.json`
  - Contains the required reduced TSC dt triplet and +6/+12/+24 leads.
  - Contains per-variable/per-lead envelopes for `U10`, `V10`, `T2`, `qv2`, and `precip`.
  - Contains `status=BLOCKED` because the pinned d02 GPU-vs-Gen2 drift was not completed.
  - Contains Thompson oracle evidence:
    - `residual_max_abs = 2.404476617812179e-10`
    - `corrupted_residual_max_abs = 9.989196314563742e-08`
    - `load_bearing = true`
  - Contains wrfbdy oracle evidence:
    - wrfbdy decoder source: pinned `wrfbdy_d01`
    - comparison status: `PARTIAL`
    - raw `max_abs_all_variables = 0.9984259605407715`
    - note records that d02 uses replay zarr and the wrfbdy proof is on d01 because no native `wrfbdy_d02` exists.

## Acceptance Criteria Accounting

- AC1: **Implemented / proof produced.** Reduced controlled dt refinement at 18/9/4.5 was run for +6/+12/+24.
- AC2: **Implemented with analytic-fixture reference.** No reviewed Gen2 multi-dt d02 campaign was available in the pinned run inventory; artifact documents analytic reduced fixture derivation and explicitly avoids l2-vs-l3 config-noise comparison.
- AC3: **BLOCKED.** Pinned d02 dt=18 drift run hit GPU OOM before per-lead GPU-vs-Gen2 drift could be computed.
- AC4: **Implemented.** Artifact is per-variable/per-lead and does not contain aggregate-only pass.
- AC5: **Produced.** `artifacts/m6/tier3/tsc_envelope.json` exists and validates against the updated schema.
- AC6: **Implemented / load-bearing.** Thompson side-channel residual stays small and a deliberately corrupted qv field causes the residual to rise by over two orders of magnitude.
- AC7: **Implemented / partial proof.** wrfbdy decoder is real and compares GPU boundary tendencies to wrfbdy `_BT*` terms. The d01 proof is partial due FP32 boundary storage and because the operational d02 path uses replay zarr rather than native `wrfbdy_d02`.
- AC8: **Implemented.** `Tier3DriftEnvelope` schema expanded.

## Unresolved Risks

- The sprint does not have a GREEN Tier-3 gate because AC3 is blocked by GPU memory on the pinned d02 dt=18 run.
- The d02 drift table is BLOCKED, not FAIL or GREEN. No conclusion should be drawn about GPU drift staying inside the TSC envelope.
- The wrfbdy independent oracle is substantive but only partial for this sprint because the pinned nested d02 forecast does not have a native `wrfbdy_d02`; the proof uses `wrfbdy_d01`.
- The inherited M6-S5 dycore cap remains in `coupling.driver` and is recorded in the artifact forcing mode.
- `artifacts/m6/tier3/` is ignored by the repo's global artifact ignore rule, so the proof file is present locally but would require force-add or ignore-rule adjustment if it must be committed.

## Next Decision Needed

Decide whether to:

1. Rerun pinned d02 drift with a smaller memory profile or segmented output strategy so AC3 can complete.
2. Accept this sprint as code/oracle scaffolding plus a BLOCKED Tier-3 proof, and dispatch a follow-up specifically for memory-safe d02 drift generation.
3. Move d02 drift comparison to an offline NPZ-output workflow that reuses existing forecast outputs, if manager approves that as equivalent evidence for AC3.
