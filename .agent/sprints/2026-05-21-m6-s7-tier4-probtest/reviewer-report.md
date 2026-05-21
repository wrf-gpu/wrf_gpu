# M6-S7 Reviewer Report — Tier-4 Probtest Prototype

Reviewer: Claude Opus 4.7 xhigh
Worker: codex gpt-5.5 xhigh
Worktree: `/tmp/wrf_gpu2_m6s7/`
Branch: `worker/codex/m6-s7-tier4-probtest`
Worker outcome claim: **BLOCKED-PARTIAL**
Binding decision: **ACCEPT-AS-SCAFFOLD-DEFER-TO-M7**

## 1. R-Findings

### 1.1 Worker honesty — VERIFIED

The data-availability blocker is structurally real and was independently reproduced.

Probe of `/mnt/data/canairy_meteo/runs/wrf_l3` (26 run directories total) yielded only **4** directories with the complete +24h wrfout_d02 history:

| Run | wrfout_d02 count | d02 T2 shape | Pinned (66,159)? |
| --- | --- | --- | --- |
| 20260430_18z_l3_24h_20260520T191306Z | 25 | (1, 66, 159) | YES |
| 20260502_18z_l3_24h_20260520T103946Z | 25 | (1, 66, 159) | YES |
| 20260509_18z_l3_24h_20260511T190519Z | 25 | (1, 66, 120) | **NO** (excluded) |
| 20260520_18z_l3_24h_20260521T045847Z | 25 | (1, 66, 159) | YES (ending cycle) |

All other 22 directories have 0 wrfout_d02 files (the Gen2 storage policy keeps only `wrfinput_d02` for older runs); one has 3 files (`20260509_18z..._T154354Z`) — far short of `+24h`. The held-out cycle `20260519_18z_..._T025228Z` has **0** wrfout_d02 files, so AC8 truth data is genuinely absent. The non-pinned (66,120) grid for the 9 May run reflects an earlier domain configuration; mixing it with the pinned grid would silently break per-cell variance — worker correctly excluded it.

Conclusion: **3/10 complete pinned-grid d02 members is the actual on-disk ceiling**, not a worker shortcut. No period in the available archive yields ten complete pinned-grid members ending at any cycle ≤ 20260520_18z.

### 1.2 Scaffold quality — HIGH

1,257 LOC across `validation/tier4_probtest.py` (793), `scripts/m6_run_tier4.py` (267), `scripts/m6_gate_tier4.py` (101), and `tests/test_m6_tier4_probtest.py` (96). Real probtest math, real stratification, real cost model, no stubs.

Spot-checked:
- **Variance estimator** (`derive_stratified_tolerance_records`): per-grid-cell `np.nanvar(..., ddof=1)` then `sqrt`; RMS reduction per stratum; tolerance = `k * sigma_rms_member_std`. No `min(raw, cap)` anywhere in the source. Grep confirms zero occurrences in tier4 code or artifacts.
- **No-peek policy** is enforced by the *call order* in `scripts/m6_run_tier4.py`: tolerances + freeze report written *before* `validate_heldout_candidate` is invoked, and held-out failure cannot rewrite the tolerance artifact (separate file, separate call).
- **Stratification** uses the shared `domain_mask` helper from M6-S2a (land/sea + 500 m elevation bands + canary), so M6-S6 and M6-S8 share the same mask definitions — no parallel-stratum drift risk.
- **Precip definition** is the correct accumulated-difference `(RAINC + RAINNC + optional RAINSH) at lead − at init`. Candidate-side falls back to zero only when NPZ accumulators are absent and *records the limitation in the artifact* rather than fabricating a value.
- **Schema** (`Tier4ProbtestTolerances`) requires prototype label, sample size, variables, leads, strata, method, tolerance factor, heldout policy. `validate_file` passes against the BLOCKED artifact (BLOCKED status is allowed; sample-size policing is in the gate, not the schema — correct separation).
- **No wrf_l2 substitution.** Grep over tier4 sources and artifacts: 0 hits for `wrf_l2`. No `wrfinput`-only fallback in member selection — `has_required_history_files` checks that every required lead resolves to a `wrfout_{domain}_` file, not just any history entry.
- **Cost model arithmetic** checks out: 29.075786 s/member × 100 = 2907.58 s ≈ 0.808 h (matches artifact). `sqrt(1/(2·99)) = 0.07106690…` (matches `n100_relative_sigma_estimator_uncertainty`).

Tests pass on re-execution (`6 passed in 0.20s`); gate correctly fails (`Tier-4 sample size must be 10 for M6-S7, got 3`).

### 1.3 Artifact honesty — VERIFIED

All four artifacts carry `status: "BLOCKED"` and identical `blockers` arrays naming the exact data deficiency. Prototype label `"M6 prototype; full ensemble at M7"` present on every artifact. Nothing is camouflaged as PASS.

| Artifact | status | blockers populated |
| --- | --- | --- |
| `ensemble_member_manifest.json` | BLOCKED | yes (count, grid mismatch) |
| `probtest_tolerances.json` | BLOCKED | yes + `sample_size_required: 10` |
| `heldout_candidate_validation.json` | BLOCKED | yes + held-out missing-history line |
| `cost_model.json` | BLOCKED | yes |
| `tolerance_freeze_report.md` | BLOCKED | yes (markdown ## Blockers section) |

## 2. Adversarial probes — outcome

- **Different historic period with more members?** No. The on-disk archive policy strips `wrfout_d02_*` from older runs; the four complete runs are spread across a 21-day window and one is the wrong grid. The blocker is not a side-effect of the chosen ending cycle.
- **Real probtest math?** Yes — the variance/stratum reduction is mathematically correct, ddof=1 is the unbiased sample estimator, k=1.96 ≈ 2σ.
- **Cost model formulas defensible?** Yes — 29 s/member from M6-S2 spacetime budget × 100 ≈ 49 minutes serial single-GPU for the recommended first M7 cohort. Relative σ-estimator uncertainty drops from ~24 % at n=10 to ~7 % at n=100, which is the correct justification for choosing 100 over 1000.

## 3. AC verification (sprint contract)

| AC | Decision basis | Verdict under SCAFFOLD-DEFER |
| --- | --- | --- |
| AC1 — 10 members | Only 3 pinned-grid d02 available; honestly recorded | SCAFFOLD-OK / DATA-BLOCKED |
| AC2 — Per-variable per-lead tolerance derivation | Code correct, n=3 diagnostic only | SCAFFOLD-OK |
| AC3 — Stratification | Implemented via shared `domain_mask`, 9 strata present | SCAFFOLD-OK |
| AC4 — Cost model | Provisional runtime from M6-S2 budget, full scaling + recommendation | PASS (provisional) |
| AC5 — Tolerance freeze report | Written before held-out validation; blocker section honest | PASS-as-prototype |
| AC6 — `Tier4ProbtestTolerances` schema | `validate_file` passes; required fields all present | PASS |
| AC7 — Prototype label honest | All artifacts carry the label | PASS |
| AC8 — Held-out validation | Held-out cycle lacks wrfout_d02; cannot score honestly | DATA-BLOCKED (deferred to M7) |

## 4. Manager decision recommendation — HYBRID (Option 1 deferred, Option 2 declined)

The worker's three options were:
- **Option 1** (preferred by worker): regenerate 10 complete pinned-grid d02 histories + held-out.
- **Option 2**: amend contract to allow surrogate sample (wrf_l2 / mixed grids) labeled prototype.
- **Option 3** (rejected by worker): silent substitution.

**My recommendation: HYBRID — ACCEPT-AS-SCAFFOLD-DEFER-TO-M7.**

Rationale:
- **Option 3** is correctly rejected. Mixing grids or `wrfinput`-only fallbacks would fabricate variance.
- **Option 1** is the right *eventual* path but it is an out-of-band data sprint (re-running 7+ days of CPU WRF backfill on the workstation), not an M6 sprint. Blocking M6 close on regeneration would delay milestone gate by days for a deliverable (Tier-4 production ensemble) that the critic amendment already pushes to M7.
- **Option 2** is unnecessary because the *scaffold itself* — not a surrogate-sample tolerance number — is the M6 deliverable per critic §7 (M6 Tier-4 is a prototype; M7 owns full ensemble). Computing tolerances against a surrogate sample would invite future readers to *use* those numbers as if they were a real tolerance freeze, when the cost model + scaffold + n=3 diagnostic are themselves the prototype.

**Concretely:**
1. Accept M6-S7 with status `ACCEPT-AS-SCAFFOLD-DEFER-TO-M7`. Worker has fulfilled the methodology contract (probtest math, stratification, schema, freeze policy, cost model, prototype labeling). Status `BLOCKED` in each artifact must remain.
2. Spawn a small follow-up data sprint **M6.5-D1** (data backfill) only if M7 dispatch demands it. Scope: regenerate CPU `wrfout_d02_*` for ≥ 10 pinned-grid 18z days plus held-out 20260519_18z, then rerun `scripts/m6_run_tier4.py` and `scripts/m6_gate_tier4.py` without any code change. The scaffold is already wired for this — it will just produce the PASS path.
3. Do **not** amend the M6-S7 sprint contract retroactively. Treat the BLOCKED gate as the truthful outcome and document the SCAFFOLD-DEFER decision in the M6 close report.

## 5. M6-S8 dispatch impact

M6-S8 is **not blocked** by this decision.

Per critic amendment §7 + S8 amendment §6, M6-S8's binding gate is CPU-vs-observation for observed U10/V10/T2 (operational philosophy), with Tier-4 reported as a **separate statistical sanity check** rather than a loosening factor. With Tier-4 frozen at status BLOCKED, M6-S8 should:
- Treat `probtest_tolerances.json` as informational (sigma diagnostic table is present; n=3 is documented).
- Bind the M6-S8 PASS criterion to CPU-vs-observation only (no `max(CPU_vs_obs, S7 tolerance)` widening).
- Record Tier-4 status `BLOCKED-deferred-to-M7` in the S8 comparison artifact.

The Tier-4 cost model (29 s/member, n=100 recommendation, full storage scaling) is the deliverable M6-S8/M7 actually needs from S7 to plan M7-S2/S3 ensemble dispatch. That deliverable is complete and provisional but defensible.

## 6. M7 deferral path

When M7 begins:
- **Prerequisite M6.5-D1**: backfill ≥ 10 + held-out wrfout_d02 days at the pinned grid. Until that lands, the scaffold remains structurally complete but BLOCKED.
- **M7-S2 (ensemble dispatch)** consumes `cost_model.json` directly for the 100-member sizing decision. Worker's recommendation of 100 over 1000 is sound (~7 % σ-estimator uncertainty, ~49 min single-GPU serial; parallelism trivially shortens that).
- **M7-Sx (Tier-4 production)**: re-run `m6_run_tier4.py` against the backfilled archive; expect `status: PASS` with `sample_size: 10`; held-out validation against M6-S2/S3 GPU NPZ outputs becomes scoreable. No code change should be needed — the gate's `--allow-heldout-fail` flag is the only knob.
- M7 should also revisit `k=1.96` — the prototype rationale is a normal-approximation 95 % band; M7 may prefer empirical quantile bounds once n ≥ 100.

## 7. Open follow-ups for the manager

- F-S7-1: schedule M6.5-D1 data backfill (CPU WRF re-runs for missing pinned-grid days). Out of scope for M6 close.
- F-S7-2: confirm M6-S8 binding gate ignores Tier-4 BLOCKED status (treat as informational sigma diagnostic, not pass-gate input).
- F-S7-3: M7-S2 dispatch must consume `cost_model.json`'s `recommended_m7_ensemble_size: 100`, not invent a new number.
- F-S7-4 (non-blocking): once M6-S5 lifted-cap verdict artifact exists, swap the provisional 29 s/member runtime in `cost_model.json` for the S5 number and rerun the cost model — the script's `--spacetime-budget` argument already handles this.

## 8. Binding decision

**ACCEPT-AS-SCAFFOLD-DEFER-TO-M7.**

The worker has fulfilled the M6-S7 prototype methodology contract in every line of code, schema, artifact, and gate that does not require physically-absent data. The BLOCKED status on the four artifacts is the *honest* outcome and is the right answer under the critic amendment's "no tolerance after seeing candidate" rule plus the project's validation philosophy. The scaffold is M7-ready: when the data backfill lands, the same scripts produce the PASS path without modification.

Files reviewed: `src/gpuwrf/validation/tier4_probtest.py`, `src/gpuwrf/io/proof_schemas.py:198-222`, `scripts/m6_run_tier4.py`, `scripts/m6_gate_tier4.py`, `tests/test_m6_tier4_probtest.py`, all four artifacts under `artifacts/m6/tier4/`, plus on-disk audit of `/mnt/data/canairy_meteo/runs/wrf_l3/`.
