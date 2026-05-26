# Sprint Contract — M6 Acceptance Attempt 2 (post 3-layer fix)

## Objective

Three deep root-cause fixes landed today:
1. HPG dpxy/mass-coupling
2. Acoustic theta mass-coupling pattern
3. RK save-family / MUTS-basis fix

The 120-step uncapped guard-disabled probe (per the RK-save-family worker) showed **NO 10× envelope breach** with **worst theta ratio = 0.93** (under 1.0 envelope). This is strong evidence the dycore is now well-formed.

**This sprint runs the actual M6 acceptance gate:**

1. Remove the `min(args.n_steps, 75)` cap in `scripts/m6_guard_disabled_debug.py` line 630.
2. Run full 360-step (1h) probe on all 3 V3 ICs with default `disable_guards=False`.
3. Run Tier-4 RMSE on all 3 ICs.
4. If GREEN: write `.agent/decisions/MILESTONE-M6-CLOSEOUT.md`.

## Non-Goals

- NO new bug fix. If the 360-step probe fails, REPORT — don't fix.
- NO modification to dycore source.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_m6acc2` on branch `worker/gpt/m6-acceptance-attempt-2`.
FIRST: `cd /tmp/wrf_gpu2_m6acc2`.

Write-only:
- `scripts/m6_guard_disabled_debug.py` (REMOVE 75-cap on line 630 only — `min(int(args.n_steps), 75)` → `int(args.n_steps)`)
- `.agent/sprints/2026-05-26-m6-acceptance-attempt-2/` — proofs + worker-report
- `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` (ONLY if all gates pass)

Read-only:
- Everything else.

## Acceptance Criteria

### Stage 1 — Remove cap + 360-step guard-disabled probe on 20260521

Patch line 630, then run `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-acceptance-attempt-2/probe_521/`. Confirm 360 steps complete with no 10× envelope breach.

If breach occurs: write `proof_first_explosive_step.json` + report blocker.

### Stage 2 — 360-step probe on 20260509 and 20260429

Same as Stage 1 for the other 2 V3 ICs.

### Stage 3 — Tier-4 RMSE acceptance (default guards mode)

Run `taskset -c 0-3 python scripts/m6_acceptance_tier4_all3.py --output .agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/`.

Acceptance criteria (per prior contract):
- All 3 ICs: |u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s, theta in [200,700]K all 360 steps.
- T2 RMSE ≤ 3 K (per IC + aggregate).
- U10 RMSE ≤ 7.5 m/s, V10 RMSE ≤ 7.5 m/s.

### Stage 4 — Parity preservation

```bash
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all  # B6 must remain 0.0
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10  # multi-step 0.0
taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v  # 12/12
```

### Stage 5 — M6 closeout (if Stage 1-4 all GREEN)

Write `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` with:
- **Status**: `M6-CLOSED`
- **Acceptance evidence**: Stage 1-4 proof paths
- **Today's 3 deep fixes** as separate commits + summary
- **Outstanding caveats**: any unresolved (e.g., guards-as-defense-in-depth justification, microphysics guard removal post-close)
- **Recommended M7 entry**: profiling sprint, GPU optimization

### Stage 6 — Worker report

`worker-report.md` with `Summary:`, status, proofs, risks, handoff. >=400 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_m6acc2
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Patch line 630 in scripts/m6_guard_disabled_debug.py (replace min(int(args.n_steps),75) with int(args.n_steps))

# Stage 1-2: 360-step probes
for IC in 20260521_18z_l3_24h_20260522T072630Z 20260509_18z_l3_24h_20260511T190519Z 20260429_18z_l3_24h_20260524T204451Z; do
    taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id "$IC" --n-steps 360 --output .agent/sprints/2026-05-26-m6-acceptance-attempt-2/probe_${IC%%_*}/
done

# Stage 3: Tier-4
taskset -c 0-3 python scripts/m6_acceptance_tier4_all3.py --output .agent/sprints/2026-05-26-m6-acceptance-attempt-2/tier4/

# Stage 4: parity
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10
taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v

git add -A && git commit -m "[M6 acceptance attempt 2] $(date -u +%FT%TZ)"
```
