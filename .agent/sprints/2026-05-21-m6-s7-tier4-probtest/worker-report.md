# M6-S7 Worker Report — Tier-4 Probtest Prototype

Worker: codex gpt-5.5 xhigh
Branch: `worker/codex/m6-s7-tier4-probtest`
Worktree: `/tmp/wrf_gpu2_m6s7`
Outcome: **BLOCKED-PARTIAL**

## Objective

Implement the M6-S7 Tier-4 probtest prototype for `U10`, `V10`, `T2`, `qv2`, and `precip` at `+6h`, `+12h`, and `+24h`. The sprint contract required a 10-member deterministic historical Gen2 `wrf_l3` sample ending at pinned `20260520_18z`, land/sea/elevation stratification via M6-S2a `domain_mask`, a frozen tolerance artifact before candidate evaluation, a storage/runtime cost model, schema validation, and held-out validation of the M6-S2/S3 GPU forecast.

## Outcome Summary

The scaffold and proof-object machinery are complete, but acceptance is blocked by local Gen2 data availability. I found only four `wrf_l3` run directories with real `wrfout_d02_*` history through `+24h`; one of those four has a non-pinned d02 grid shape `(66, 120)` instead of the pinned `(66, 159)`. The usable pinned-grid complete sample is therefore **3/10 members**. The M6-S2 held-out cycle `20260519_18z_l3_24h_20260520T025228Z` has no real `wrfout_d02_*` files for `+6/+12/+24h`; `Gen2Run.history_files("d02")` falls back to `wrfinput_d02`, so AC8 cannot be scored honestly.

The generated artifacts are intentionally `status: "BLOCKED"` and labeled `M6 prototype; full ensemble at M7`. The tolerance table is a pre-candidate diagnostic n=3 table, not an accepted Tier-4 tolerance freeze.

## AC Status

| AC | Status | Evidence / Notes |
| --- | --- | --- |
| AC1 — 10-member historical sample | **BLOCKED** | `ensemble_member_manifest.json` records `sample_size: 3`, `sample_size_required: 10`. Only four complete-history local `wrf_l3` d02 runs were available; `20260509_18z_l3_24h_20260511T190519Z` was excluded for shape `(66, 120)` vs pinned `(66, 159)`. |
| AC2 — Per-variable per-lead tolerance derivation | **PARTIAL / BLOCKED FOR ACCEPTANCE** | `probtest_tolerances.json` contains diagnostic tolerances for all requested variables/leads using `k=1.96`, `ddof=1`, and no `min(raw, cap)` fudge. Because n=3 not n=10, status remains `BLOCKED`. |
| AC3 — Stratification | **PARTIAL / BLOCKED FOR ACCEPTANCE** | Implemented via shared `domain_mask`; artifact includes `land`, `sea`, and `elevation_band_0..5` plus `canary`. Same data blocker as AC1. |
| AC4 — Storage/runtime cost model | **PARTIAL** | `cost_model.json` produced. Runtime uses provisional M6-S2 `spacetime_budget_d02.json` because no M6-S5 lifted-cap verdict artifact was visible in this worktree. Cost model explicitly gates M7 dispatch. |
| AC5 — Tolerance freeze report | **PARTIAL / BLOCKED** | `tolerance_freeze_report.md` written before held-out validation attempt. It documents frozen choices and blockers. It is not an accepted tolerance freeze because sample size is insufficient. |
| AC6 — `Tier4ProbtestTolerances` schema validated | **PASS** | `proof_schemas.py` now requires prototype label, domain, sample size, variables, leads, strata, method, tolerance factor, and held-out policy. `Tier4ProbtestTolerances.validate_file(...)` passed on the artifact. |
| AC7 — Prototype label | **PASS** | All Tier-4 artifacts explicitly contain `M6 prototype; full ensemble at M7`. The report and cost model state that M7 ensemble dispatch is gated. |
| AC8 — Held-out candidate validation | **BLOCKED** | `heldout_candidate_validation.json` records that held-out `20260519_18z_l3_24h_20260520T025228Z` lacks real `wrfout_d02` history for `[6, 12, 24]`. Candidate validation was attempted only after frozen artifacts existed, preserving the no-after-failure rule. |

## Scaffold Completed

- `src/gpuwrf/validation/tier4_probtest.py`
  - deterministic member selection with held-out cycle exclusion;
  - complete-history validation for init/+6/+12/+24 files;
  - pinned-grid filtering against the ending-cycle d02 shape;
  - per-member SHA manifest generation;
  - variable loading through `Gen2Run` and shared validation I/O;
  - precipitation definition as `(RAINC + RAINNC + optional RAINSH at lead) - same components at lead 0`;
  - land/sea/elevation masks from `domain_mask`;
  - tolerance derivation as `k * RMS(per-grid-cell member std)`, `k=1.96`, `ddof=1`;
  - held-out candidate validation kept separate from tolerance freeze;
  - storage/runtime cost model;
  - freeze-report writer.
- `src/gpuwrf/io/proof_schemas.py`
  - tightened `Tier4ProbtestTolerances` required fields.
- `scripts/m6_run_tier4.py`
  - writes manifest, tolerances, freeze report, cost model, then held-out validation artifact;
  - records blockers rather than silently substituting invalid data.
- `scripts/m6_gate_tier4.py`
  - validates schema/table completeness/prototype label/no forbidden cap language/sample-size gate/cost-model recommendation.
- `tests/test_m6_tier4_probtest.py`
  - covers tolerance math, duplicate/held-out member selection, and schema-required freeze fields.

## Proof Objects Produced

- `artifacts/m6/tier4/ensemble_member_manifest.json`
  - `sample_size: 3`;
  - per-member SHA-256 records for `namelist.input`, `wrfinput_d02`, and `wrfout_d02` at init/+6/+12/+24;
  - blockers for insufficient complete histories and one non-pinned-grid complete member.
- `artifacts/m6/tier4/probtest_tolerances.json`
  - `status: "BLOCKED"`;
  - diagnostic n=3 per-variable/per-lead/per-stratum tolerance table;
  - `sample_size_required: 10`;
  - no post-candidate tolerance mutation.
- `artifacts/m6/tier4/heldout_candidate_validation.json`
  - `status: "BLOCKED"`;
  - explicit held-out missing-history blocker.
- `artifacts/m6/tier4/cost_model.json`
  - `status: "BLOCKED"`;
  - per-member referenced history/static mean about 68.85 MB;
  - requested-lead GPU NPZ sample about 39.29 MB/member;
  - compact surface-only NPZ estimate 629,640 bytes/member;
  - provisional 24h runtime from M6-S2 spacetime budget: 29.075786 s/member, 0 post-init transfer bytes;
  - recommended first M7 full ensemble size: 100 members, pending data and M6-S5 cost gates.
- `artifacts/m6/tier4/tolerance_freeze_report.md`
  - records frozen choices and blockers before held-out evaluation.

## Commands Run

- `pytest -q tests/test_m6_tier4_probtest.py tests/test_m6_proof_schemas.py`
- `python -m py_compile src/gpuwrf/validation/tier4_probtest.py scripts/m6_run_tier4.py scripts/m6_gate_tier4.py`
- `python scripts/m6_run_tier4.py`
- `python - <<'PY' ... Tier4ProbtestTolerances.validate_file('artifacts/m6/tier4/probtest_tolerances.json') ... PY`
- `python scripts/m6_gate_tier4.py --allow-heldout-fail`
- `pytest -q tests/test_m6_tier4_probtest.py tests/test_m6_proof_schemas.py tests/test_m6_validation_io.py`

## Validation Results

- Unit/schema/shared-I/O tests: `8 passed in 0.66s`.
- `Tier4ProbtestTolerances.validate_file('artifacts/m6/tier4/probtest_tolerances.json')`: PASS.
- `python scripts/m6_run_tier4.py`: completed and emitted artifacts with `status: "BLOCKED"` and `heldout_status: "BLOCKED"`.
- `python scripts/m6_gate_tier4.py --allow-heldout-fail`: correctly failed with `Tier-4 sample size must be 10 for M6-S7, got 3`.

The gate failure is the correct outcome under the sprint contract.

## Honest Blocker Analysis

The sprint contract required deterministic historical Gen2 `wrf_l3` day-members, not an alternate sample. I did not substitute `wrf_l2`, mixed grids, or `wrfinput`-only directories because that would make the tolerance table look accepted while violating the data premise. The local data can prove the method, schema, and artifact behavior, but it cannot prove the requested statistical tolerance. The held-out validation blocker is separate and equally hard: without CPU `wrfout_d02` truth for the pinned M6-S2 day, a GPU-vs-Gen2 RMSE check would be fabricated.

Precipitation is also explicitly limited. The CPU-side precip loader uses accumulated WRF precipitation components. The candidate validator can represent current M6 GPU precipitation as zero-accumulator behavior if candidate NPZ lacks precip accumulators, but that path was not reached because held-out CPU truth is absent.

## Recommended Next Sprint / Manager Decision

Recommended manager decision: close M6-S7 as **BLOCKED-PARTIAL pending data decision**.

Next-sprint options:

1. **Preferred:** restore or generate 10 complete pinned-grid Gen2 `wrf_l3` d02 histories ending at `20260520_18z`, including init/+6/+12/+24 for each member, and restore/generate held-out `20260519_18z` d02 history. Then rerun `python scripts/m6_run_tier4.py` and expect the gate to evaluate normally.
2. **Alternative requiring explicit approval:** amend the sprint contract to allow a surrogate sample such as `wrf_l2` or mixed `wrf_l3`/`wrf_l2`. This must be labeled as a surrogate prototype and should not be used as a production Tier-4 tolerance freeze.
3. **Do not do silently:** do not count `wrfinput_d02` fallback directories as day-members, do not mix the `(66, 120)` d02 grid with the pinned `(66, 159)` grid without an approved regridding rule, and do not tune tolerances after looking at candidate failures.
