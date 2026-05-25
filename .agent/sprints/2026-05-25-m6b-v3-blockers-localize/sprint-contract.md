# Sprint Contract — M6b V3 Blockers Localization (operational-vs-validation parity on real Gen2 IDs)

## Objective

M6b V3 (post-reframe operational 1h × 3 Gen2 IDs) returned PARTIAL:
- **`20260521_18z_l3_24h_20260522T072630Z`**: bounded for 45 steps, then v_abs_max = 103.72 m/s at step 46 (~460s, 4% over the 100 m/s bound)
- **`20260509_18z_l3_24h_20260511T190519Z`**: theta explosion at some earlier step (per worker note)
- All other gates clean (no NaN/Inf, theta lower-30 bounded, D2H=0)

**The decisive question**: are these failures **real math defects** (operational diverging from validation) or **physically valid** outputs that exceed our tight bounds (operational matches validation but both exceed bound)?

Per the bisect (`worker/gpt/m6b-standalone-vs-comparator-bisect`), multi-step parity at 2/5/10 was 0.0 bitwise on 20260521. This sprint extends that comparator to step 46 (and beyond) on **both** Gen2 IDs to answer the question.

## Non-Goals

- NO modifications to `dynamics/core/`, `validation_wrappers.py`, or `operational_mode.py` body.
- NO modifications to operational `wrf.exe`.
- NO sanitizer.
- NO new physics.
- NO 1h full forecast on additional IDs in this sprint.
- NO bound revision in this sprint (recommend it; don't apply).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_v3localize` on branch `worker/gpt/m6b-v3-blockers-localize`.

Write-only:
- `scripts/m6b_real_ic_operational_compare.py` — extend with longer-step capability if needed
- `tests/test_m6b_v3_localization.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-v3-blockers-localize/` — proofs + worker-report

Read-only everywhere else.

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/worker-report.md` (the PARTIAL verdict)
3. `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_bounds.json` (the per-step bounds detail)
4. `.agent/sprints/2026-05-25-m6b-standalone-vs-comparator-bisect/worker-report.md` (the 10-step bitwise PASS)
5. `scripts/m6b_real_ic_operational_compare.py` (the multi-step comparator)

## Acceptance Criteria

### Stage 1 — Comparator on 20260521 out to step 50 (MANDATORY)

Run operational vs validation side-by-side on `20260521_18z_l3_24h_20260522T072630Z` for 50 steps. Per-step max-abs delta.

**If max-abs delta = 0.0 at step 46 (and all other steps)**: 20260521's v=103 is physically real (validation produces the same). Recommend bound revision.

**If max-abs delta > 0 at step N < 46**: operational diverges from validation; localize the operator.

Capture: `proof_20260521_step50_compare.json`.

### Stage 2 — Comparator on 20260509 out to step 50 (MANDATORY)

Same comparator, 20260509 IC, 50 steps.

**If both paths produce the theta explosion**: it's a real numerical instability for that IC (Gen2 reference may also blow up; or the operational IS the reference and explosion is real). Need different escalation.

**If only operational explodes**: the operational has an IC-specific bug. Localize.

Capture: `proof_20260509_step50_compare.json`.

### Stage 3 — Diagnostic memo (MANDATORY)

`localization_memo.md`:
1. Per-ID per-step deltas summary
2. For each ID: is operational mathematically equivalent to validation (yes/no)? If yes, the V3 bounds failures are physical. If no, the operator-level localization.
3. Recommended next sprint:
   - **`BOUND-REVISION`**: relax v bound from 100 to 120 m/s tropospheric (physics-justified for jet stream); fold into Stage 1 of next M6b RETRY; estimate impact on 20260509
   - **`NAMED-FIX-<operator>`**: operational composition has remaining defect; localized to operator X at step Y
   - **`IC-SPECIFIC-DEFECT`**: 20260509 has an IC-reader or initialization defect

### Stage 4 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m6b_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py -v
```

### Stage 5 — Worker report

`worker-report.md`: per-stage status, deltas table, named recommendation.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_v3localize
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 50 2>&1 | tee .agent/sprints/2026-05-25-m6b-v3-blockers-localize/proof_20260521_step50.txt
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260509_18z_l3_24h_20260511T190519Z --steps 50 2>&1 | tee .agent/sprints/2026-05-25-m6b-v3-blockers-localize/proof_20260509_step50.txt
pytest <test list> -v
```

## Kill Gates

- Operational sha256 changes → STOP.
- Validation regression → REJECT.

## Risks

- 50-step comparator takes 5-15min per ID due to JAX compile + run.
- If both paths explode on 20260509, we don't know which is right (validation may also be wrong for that IC); flag honestly.

## Handoff Requirements

When localization memo committed: `/exit`. Manager dispatches the named next sprint (BOUND-REVISION / NAMED-FIX / IC-SPECIFIC).

Time budget: **30-60 min**.
