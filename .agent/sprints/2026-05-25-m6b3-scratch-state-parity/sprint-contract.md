# Sprint Contract — M6B3: Scratch-State Parity (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, save fields)

## Objective

Fourth rung of the B-direct ladder. With calc_coef_w (M6B0-R), advance_mu_t (M6B1), and Thomas tridiag (M6B2) all parity-achieved, M6B3 adopts WRF's small-step scratch state into the JAX validation harness and proves per-field parity. These are the running-average and stage-carry fields that WRF uses across acoustic substeps within one RK stage; without them the small-step recurrence drifts.

Per the consultation's diagnosis: ADR-023's minimalist-carry thesis failed because these scratch fields *are* required for WRF-equivalent small-step semantics. M6B3 brings them in **for validation mode**. Whether they remain in operational mode is a separate decision (M6-perf-design + ADR-026 + Critic Amendment #1 classification).

## Non-Goals

- NO RMSE tuning.
- NO modifications to operational `wrf.exe`. Pre/post sha256 inherited.
- NO modifications to operational runtime carry shape (only validation harness adds scratch).
- NO multi-RK-stage coupling in this sprint (M6B4 is the acoustic recurrence ladder rung).
- NO 1h forecast.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_m6b3` on branch `worker/gpt/m6b3-scratch-state-parity`.

Write-only:
- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` — extend with `sp_t_2ave_update_pre/post`, `sp_ww_update_pre/post`, `sp_muave_update_pre/post`, `sp_ph_tend_accumulate_pre/post`, plus a `sp_substep_save_state` that snapshots `_save` family fields between substeps
- `external/wrf_savepoint_patch/solve_em.F.patch` — extend with new call sites
- `scripts/m6b3_scratch_state_compare.py`
- `src/gpuwrf/dynamics/small_step_scratch.py` (NEW) — validation-only callable that produces WRF-shaped scratch from prognostic state; do not wire into operational runtime
- `src/gpuwrf/validation/savepoint_schema.py` — add scratch-family enum + tolerance ladder entries (MUAVE accumulation, t_2ave running average per WRF: `(t_old + t_new)/2`)
- `tests/test_m6b3_scratch_state_parity.py`
- `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/` — proofs + worker-report

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-25-m6b1-advance-mu-t-parity/worker-report.md` (pattern)
3. `.agent/sprints/2026-05-25-m6b2-tridiagonal-solve-parity/worker-report.md` (pattern)
4. `.agent/decisions/ADR-025-wrf-savepoint-bdirect-port-PROPOSED.md`
5. `PROJECT_PLAN.md §14.5.1` (operational-compatibility invariants — Critic Amendment #1; THESE FIELDS ARE THE CANARY for carry-creep, classify carefully)
6. WRF source line ranges (per env-audit table):
   - `module_small_step_em.F:969-1175` (advance_mu_t — t_2ave, ww, muave, muts, ph_tend updates here)
   - `module_small_step_em.F:1399-1581` (advance_w — `_save` family stage transitions)
7. `src/gpuwrf/validation/tolerance_ladder.json` (extend)

## Acceptance Criteria

### Stage 1 — Wrapper extension + rebuild

Add 5 new operator pairs (10 hooks total) for the scratch families. Pre/post operational sha256 check.

### Stage 2 — Synthetic dry-run extension

Verify each new field's fail-closed comparator semantics.

### Stage 3 — Real WRF scratch-state extraction

Tier-1 column / Tier-2 16×16 / Tier-3 golden small-domain (pinned run-ID inherited). 10 acoustic substeps each.

### Stage 4 — First REAL JAX-vs-WRF scratch-state parity

For each scratch field (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, save fields):
- Per-tier max-abs delta vs ladder tolerance
- Sanitizer-OFF

Outcome: `FOURTH-OPERATOR-FAMILY-PARITY-ACHIEVED` or `PARITY-DEFECT-LOCALIZED-IN-<field>`.

### Stage 5 — Kill gate (Critic Amendment #5)

Count diverging fields across the 3 tiers at substep 1. If >15 → STOP, escalate.

### Stage 6 — Operational-compatibility section (CRITICAL — Critic Amendment #1)

This sprint is the highest-risk sprint for **carry creep**. The 6 scratch families WILL be classified, and the principal directive says they should be **operational-undecided** by default — only operational-approved-with-evidence if a Tier-4 ablation proves they cannot drop. Default classifications expected:

| Field family | Validation classification | Operational classification (default) |
|---|---|---|
| `t_2ave` (theta running average) | Required | **Undecided** — defer to M6-perf-design ablation |
| `ww` (omega→w metric working state) | Required | **Undecided** |
| `muave` (column-mass running average) | Required | **Undecided** |
| `muts` (column-mass at substep) | Required | **Undecided** |
| `ph_tend` (geopotential tendency accumulator) | Required | **Undecided** |
| `_save` family (RK stage transition state) | Required | **Undecided** |

**Undecided fields may not enter operational state APIs.** A follow-up M6-perf-design ablation sprint decides per-field whether it lives in operational mode.

If the worker classifies any of these as **operational-approved-with-evidence**, the worker MUST cite a specific Tier-4 ablation result (which does not yet exist), so this is effectively gating future evidence.

### Stage 7 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py -v
```

### Stage 8 — Worker report

`worker-report.md`: all stages, operational-compatibility classification table (per-field), parity result, kill-gate decision, files changed, handoff to M6B4 (acoustic recurrence parity).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6b3
bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_build_rebuild.txt
python scripts/m6b3_scratch_state_compare.py --synthetic-dryrun 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_synthetic_dryrun_m6b3.txt
python scripts/m6b3_scratch_state_compare.py --tier column --steps 10
python scripts/m6b3_scratch_state_compare.py --tier patch16 --steps 10
python scripts/m6b3_scratch_state_compare.py --tier golden --steps 10
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_no_regression.txt
```

## Performance Metrics

N/A.

## Kill Gates

- >15 fields diverge at substep 1 → STOP, escalate.
- Operational sha changes → STOP, revert.
- Any field classified `operational-approved-with-evidence` without a Tier-4 citation → **REJECT** (classification must be Undecided).

## Risks

- This sprint is where the M6 dycore literally **becomes WRF-shaped at the carry level**. Without Critic Amendment #1 discipline, this is exactly where the project ships CPU-shape into production by accident. Worker must default to Undecided + defer to M6-perf-design ablation.
- The `_save` family is large; storage may pressure /tmp. Use Tier-3 short-pulse (10 substeps) only.

## Handoff Requirements

When all proofs + worker-report.md committed on branch `worker/gpt/m6b3-scratch-state-parity`: `/exit`. Manager dispatches M6B4 (acoustic recurrence parity).

## Failure modes the manager will reject

- Operational-approved without Tier-4 citation.
- Skipping operational-compatibility section.
- Modifying operational runtime carry shape.
- Multi-RK-stage coupling.
- Post-sanitize finiteness as acceptance.
