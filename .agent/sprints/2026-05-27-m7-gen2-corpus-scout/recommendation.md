# Recommendation — M7 Gen2 Corpus Scout

**Sprint**: `2026-05-27-m7-gen2-corpus-scout` (AC4)
**Inputs**: `full_gen2_inventory.json`, `pinning_analysis.md`, `recovery_candidates.md`
**State of disk (read-only)**: 2 pinned-grid-complete L3 24h members (`20260521`, `20260524`), 1 live (`20260525`) likely to land as a 3rd, 1 L2 72h sibling of `20260524` on the pinned grid, 51 cycle dirs stripped of wrfouts.

The M7-S0 harness requires 10 pinned-grid-complete L3 24h members with cycle ≤ `DEFAULT_ENDING_CYCLE` (`20260520_18z`). On the current disk, the count is **0**. Even after lifting the cycle window, the count is **2 → 3** after `20260525` finishes.

## Options

### Option A — Lower the M7-S0 corpus floor

**Action**: change `count = 10` (`select_historical_members` default in `src/gpuwrf/validation/tier4_probtest.py:108`) to a smaller value, with matching update of the M6-S7 reviewer-report threshold cited at `.agent/sprints/2026-05-21-m6-s7-tier4-probtest/reviewer-report.md:50,107`. Also bump `DEFAULT_ENDING_CYCLE` to `20260524_18z` (or `20260525_18z` once live run lands) so existing pinned-grid members fall in-window.

**Statistical cost**:
- The tolerance freeze derives per-stratum sigma from a single-cell sample-variance estimator with `ddof=1` (`tier4_probtest.derive_probtest_tolerances:349-365`). With `N=2`, sample variance has 1 d.o.f. and the 95% upper-bound multiplier for σ from a chi-squared distribution is ~31× (vs. ~1.32× at N=10). A 31× inflation in tolerances guts the M7 RMSE claim — any GPU forecast within plausible operational range would pass trivially, and the gate stops discriminating.
- Empirical bisection minimum: `N≥5` keeps the upper-tail σ multiplier under ~3×, which is still pessimistic but recoverable. `N≥7` brings it under ~2×.
- A held-out validation member must be excluded *before* the freeze; with `N=5` plus 1 held-out, the corpus needs ≥6 pinned-grid-complete runs.

**Trade-off**: cheap to implement (a constant change + ADR), but rewriting the M6-S7 acceptance gate is governance work and the claim "Tier-4 RMSE validated against Gen2" weakens commensurately. Acceptable as a **bridge**, not as the final M7 close.

### Option B — Relax the pinned-grid requirement

**Action**: allow the `(66, 120)` legacy `d02` to participate by adding a regrid step (bilinear/conservative interp from old to new grid) inside `data_quality.compute_rmse_against_gen2`. Would lift the `20260509` L3+L2 runs back into the eligible set, bringing total candidates to 4-5 pinned-equivalent members.

**Comparator complexity cost**:
- The adapter is currently strict (`shape mismatch → ValueError`, `data_quality.py:374`); adding a regrid path means a new interpolation operator, lat/lon source registration for the old grid, and validation that the regrid error itself is sub-tolerance. That's a new sub-module plus a new fixture and a new ADR.
- Worse: the regridded field is no longer "CPU WRF baseline" — it's "CPU WRF baseline through a regrid filter", which inflates the noise floor and contaminates the operational RMSE claim. The whole point of the pinned grid was to avoid this confound.
- Counts: only +1 unique cycle gained (`20260509_18z`), since the L2 sibling is the same cycle.

**Trade-off**: expensive to implement, weakens the validation, and yields only one extra cycle. **Reject** for the M7 close path.

### Option C — Relabel WRONG_GRID runs as a separate grid family

**Action**: treat `(66, 120)` as its own stratum `legacy_d02_v0`; build a parallel inventory + tolerance freeze for that family; report two independent Tier-4 numbers.

**Spirit of acceptance gate**: the M7 Tier-4 gate is about "GPU is consistent with what the operational CPU WRF would have produced *for the same configuration*". Two strata answers a different question: "GPU is consistent with each of two historical CPU WRF configurations". That's a research result, not a release gate. Also, the `(66, 120)` family has only 2 members in total (1 L3, 1 L2 — same cycle), so its own N is even worse than the pinned family.

**Trade-off**: **Reject**. Doesn't help the gate; adds bookkeeping.

### Option D — Trigger fresh CPU WRF runs

**Action**: ask the Canairy Gen2 operator to (i) flip the retention policy so `wrfout_d02_*` is preserved going forward, (ii) re-run a set of past cycles on the pinned grid using the surviving WPS staging dirs (`runs/wps_cases/20260428_18z_72h`, `20260429_18z_72h`, `20260521..20260525_18z_72h`) plus rebuilt met_em for cycles whose WPS is gone.

**Day count + estimated wall-time**:
- 24-h L3 single-domain forecast wall-time on the 28-rank CPU baseline: **~3 h elapsed** per cycle (extrapolated from `nightly_scale_up.log` cadence ~22 Z start → 02 Z complete in observed live runs).
- To reach `N=10` pinned-grid-complete L3 24h members starting from the 2 already on disk + the live `20260525`:
  - 7 more cycles needed → **21 wall-hours of CPU WRF** (sequential), or **2-3 wall-days** of nightly runs on the existing once-per-night cadence.
  - To reach `N=14` (M7-S0a-plan preferred margin): 11 more → ~33 wall-hours / 3-4 nightly runs.
- WPS regeneration for cycles whose met_em is gone: AIFS month files are on disk for `202604`/`202605`; WPS itself takes ~15 min/cycle on 4 cores. Negligible vs. WRF.
- Disk impact: each L3 24h run with 25 hourly d02 wrfouts is ~3-4 GB based on `total_bytes` per the inventory; 8 new full retentions ≈ 24-32 GB net new on `/mnt/data/canairy_meteo/runs/wrf_l3/`. The existing wrf_l3 tree already carries ~20 GB across the stripped dirs; budget impact is modest.

**Trade-off**: highest correctness, lowest engineering work in this repo, but requires:
- An operator action on a tree this project explicitly treats read-only.
- Calendar time (2-4 nights of live runs, or 1-2 operator-days of replays).
- Coordination with the M7 schedule (forecast-vs-obs scaffold already merged — see git log; daily pipeline already integrated).

**This is what the M7-S0a backfill plan already prescribes** (`.agent/sprints/2026-05-22-m7-s0a-ops-data-prologue/gen2_corpus_backfill_plan.md:29-37`); the present sprint independently confirms it is still the right call.

## Recommendation — **Option D + bridge from Option A**

**Primary**: execute Option D (operator-side retention flip + targeted CPU WRF re-runs). This restores the pinned-grid Tier-4 claim without weakening it, and the wall-time is bounded (≤4 nights or ≤2 operator-days).

**Bridge while D runs**: take Option A in a *bounded* form — temporarily run M7-S0 with `count=5`, `--ending-cycle 20260525_18z`, marked as a **probationary tolerance freeze** with a written stop-condition "supersede on first `N≥10` rerun". This unblocks the M7-S0 dispatch on this week's timescale rather than next-week's. The probationary RMSE numbers must be tagged `PROBATIONARY_N5` in any artifact and not cited as the operational M7 Tier-4 evidence.

**Reject**: Options B and C — both weaken the gate semantics for too little corpus gain.

## Concrete next-step asks (for the manager)

1. **Operator action**: ask the Canairy Gen2 owner to set `KEEP_D02_WRFOUT=1` (or equivalent retention flag) for the next 7-10 successful 18Z cycles, starting tonight `20260527_18z`. Without this, the corpus will not grow at all.
2. **Retention re-run** of `20260520_18z` (the *original* M6-S2 frozen reference cycle whose own wrfouts were stripped) if it's needed for held-out validation lineage; otherwise pick `20260519_18z` as held-out and exclude `20260520_18z` from the freeze pool.
3. **GPU project**: dispatch a follow-up sprint after the corpus reaches `N≥6` to bump `DEFAULT_ENDING_CYCLE` + run `m6_run_tier4.py` cleanly. No code change to this sprint's scope.
4. **Do NOT** dispatch a writer sprint to relabel/regrid existing data — Options B and C are net negatives.

## Stop-go criteria for next dispatch

- **Go to follow-up sprint** when `tier4_eligible_pinned_complete_runs ≥ 6` in a fresh inventory snapshot.
- **Hold** below that count; re-scout once retention has been live ≥ 4 nights.
- **Escalate to user** if no operator response within 48 h after the retention flip is requested — at that point, only Option A probationary path remains, and that needs human sign-off because it weakens the M7 acceptance gate.
