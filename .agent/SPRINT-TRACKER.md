# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence. **Update on every dispatch per user 12:35.**

## Currently in flight (3 codex — all Working confirmed)

| Window | Sprint | AI | Started | Status |
|---|---|---|---|---|
| `worker-s3zz` | M5-S3.zz RRTMG SW closeout | codex gpt-5.5 xhigh | 12:48 | finalizing — sfluxzen+setcoef ROOT CAUSES CLOSED; new broadband-transfer/cloud-optics root cause emerged (sprint outcome: PARTIAL); writing report + committing |
| `worker-m6s2` | M6-S2 coupled forecast driver | codex gpt-5.5 xhigh | 12:50 | 56m — running 1h smoke proof after fixing finite-state (capped internal reduced-dycore step at 1s); long compile + iterating |
| `critic-m7plan` | M7 plan critical-review | codex gpt-5.5 xhigh | 13:05 | re-paste required at 13:46 (first prompt didn't fire); now Working |

## Tick observations

**M5-S3.zz partial outcome**: worker closed the M5-S3.z reviewer's named root causes (sfluxzen band/g-point allocation + setcoef precision) but Tier-1 SW flux STILL fails — root cause shifted to broader transfer-solver and cloud-optics closure (cldprmc_sw/spcvmc_sw layer optical inputs). Worker recommends next sprint extract those as intermediate oracles before more production code edits. **M5-S3.zzz scope may need amendment** (LW closeout was advance-bound, but the SW transfer-solver/cloud-optics issue is newly discovered).

**M6-S2 reality**: 1h smoke compile + run is slower than expected on real 16×16×30 → 160×67×45 d02 domain. Worker iterating; this is normal first-real-coupled-forecast pain. Watching.

**M7 critic dispatch failure**: first prompt didn't fire (sandbox + paste timing); re-pasted at 13:46. Now Working. Wall budget still 30-60min from re-paste.

## Closed this tick

(none — all 3 still working; M7 critic re-pasted)

## M5 sprint table

| Sprint | Status | Notes |
|---|---|---|
| M5-S0 through M5-S3.z (11 sprints) | ✓ all CLOSED | M5 prologue closure tracked in MILESTONE-M5-CLOSEOUT.md |
| **M5-S3.zz RRTMG SW closeout** | 🟡 worker finalizing | Outcome: sfluxzen+setcoef CLOSED; new SW broadband/cloud-optics root cause |
| M5-S3.zzz LW closeout | ⚪ queued (may need scope amendment per S3.zz finding) | Advance-bound but new SW issue may need parallel/prior sprint |
| M5-S1.z Thompson collision tables | ⚪ optional | Only if M6 RMSE flags |

## M6 sprint table

| Sprint | Status |
|---|---|
| M6 plan + S1 + S2a | ✓ all closed |
| **M6-S2 coupled forecast driver** | 🟡 worker iterating (56m+, 1h smoke compile) |
| M6-S3 surface + Noah-MP | ⚪ queued (after M6-S2 smoke) |
| M6-S4..S7 (Tier-2/3/4 + 4× verdict) | ⚪ queued (parallel after M6-S3) |
| M6-S8 operational + closeout | ⚪ queued (serial final) |

## M7 sprint table

| Sprint | Status |
|---|---|
| M7 plan scout | ✓ CLOSED (34KB plan, 9 sprints) |
| **M7 plan critic** | 🟡 codex Working (re-paste 13:46) |
| M7 plan manager-amendments | ⚪ queued |
| M7-S0..S8 implementation | ⚪ queued |

## M8 (queued)

| Sprint | Status |
|---|---|
| M8 forkable release | ⚪ queued |

## Big-picture critical path (UPDATED with S3.zz finding)

```
NOW (3 codex)
  ├─ M5-S3.zz close (PARTIAL — sfluxzen+setcoef CLOSED, new broadband root cause) → Opus
  │   → M5-S3.zzzz cldprmc_sw/spcvmc_sw intermediate oracle (NEW scope per worker)
  │   → M5-S3.zzz LW closeout (advance-bound, but order may swap)
  │   → M5 RRTMG PARITY
  ├─ M6-S2 close → Opus → M6-S3 → Opus → M6-S4..S7 parallel → M6-S8 → M6 GREEN
  └─ M7 critic close → manager amendments → M7-S0 ready
```

**Calendar update**: M5 RRTMG PARITY likely needs 1 more sprint than originally projected (S3.zzzz cldprmc + S3.zzz LW = 2 more cycles); M6 close 5-8 days; end-goal landing ~3-4 weeks.

## File-ownership snapshot

- `pyproject.toml`: M6-S2 amending (zarr+jax STEP 0)
- `src/gpuwrf/contracts/state.py`: M6-S2 extending (boundary leaves)
- `src/gpuwrf/coupling/{driver, boundary_apply}.py`: NEW M6-S2
- `src/gpuwrf/io/**`: M6-S2a CLOSED (frozen)
- `src/gpuwrf/physics/rrtmg_sw.py`: M5-S3.zz in flight (sfluxzen + setcoef precision + lax.scan; uncommitted changes)
- `src/gpuwrf/physics/rrtmg_lw.py`: FROZEN until M5-S3.zzz
- All other physics: CLOSED

## Watchman policy

- 30-min cadence per user
- **Routine: update tracker on every new dispatch + watchman tick**
- Next: 14:16

## Recent ticks

- 13:08: 3 codex dispatched (s3zz + m6s2 + m7critic); m7critic dispatch failure noted
- 13:46 (this tick): all 3 still in flight; s3zz delivered partial + new root cause discovery; m6s2 iterating; m7critic re-pasted (re-Working)
- Next: 14:16
