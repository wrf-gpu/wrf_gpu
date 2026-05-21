# Sprint Tracker — Live Dashboard

Manager-maintained. **20-min cadence** (per user 14:05 AFK directive). **Update on every dispatch** (per user 12:35).

## Currently in flight (3 agents — 1 Opus + 2 codex)

| Window | Sprint | AI | Started | Wall | Notes |
|---|---|---|---|---|---|
| `reviewer-s3zz` | M5-S3.zz RRTMG SW closeout Opus review | Opus 4.7 xhigh | 14:01 | ~15-25min | Verify worker partial; bind M5-S3.zzzz scope (cldprmc+spcvmc oracle) |
| `worker-m6s2` | M6-S2 coupled forecast driver (1h→6h→24h on d02) | codex gpt-5.5 xhigh | 12:50 (1h20m+ in) | 24-36h | Iterating 1h smoke after finite-state cap fix; long compile |
| `worker-s3zzz` | M5-S3.zzz RRTMG LW closeout (16 bands taumol+fracs) | codex gpt-5.5 xhigh | 14:08 | 24-48h | NEW dispatch in parallel with S3.zz Opus + M6-S2; file-disjoint (rrtmg_lw only) |

**Rate-limit watch**: 2 codex + 1 opus active = within tested limit. Opus quota separate from gpt quota.

## Just dispatched this tick

- **M5-S3.zzz LW closeout**: codex worker spawned per user "max parallel" directive
- Contract written per M5-S3.z reviewer §4 advance-binding (Option 2)
- File-disjoint from M5-S3.zz (SW) and M6-S2 (coupling/contracts)

## M5 sprint table (live)

| Sprint | Status | Detail |
|---|---|---|
| M5-S0 → M5-S3.z (14 sprints) | ✓ all CLOSED | M5 prologue completed; physics suite proven on column oracle |
| **M5-S3.zz RRTMG SW closeout** | 🟡 Opus reviewer IN FLIGHT | Worker delivered partial; sfluxzen+setcoef CLOSED; new broadband root cause |
| **M5-S3.zzz RRTMG LW closeout** | 🟡 codex worker IN FLIGHT (just dispatched) | 16 LW bands taumol+fracs transcription per M5-S3.z reviewer §4 |
| M5-S3.zzzz cldprmc+spcvmc SW oracle (NEW) | ⚪ contract STUB ready | Dispatch after S3.zz Opus accepts |
| M5-S1.z Thompson collision tables | ⚪ optional | Only if M6 RMSE flags microphysics drift |

**M5 RRTMG PARITY** requires: S3.zz Opus + S3.zzzz close + S3.zzz close. **Earliest unblock: ~24-48h.**

## M6 sprint table (live)

| Sprint | Status | Detail |
|---|---|---|
| M6 plan + S1 + S2a | ✓ all CLOSED | Infrastructure ready |
| **M6-S2 coupled forecast driver** | 🟡 worker iterating (1h20m+) | 1h smoke compile after dycore-step cap fix |
| M6-S3 surface layer + Noah-MP | ⚪ contract READY; depends on M6-S2 smoke | 30-48h |
| M6-S4..S7 (Tier-2/3/4 + 4× verdict) | ⚪ queued (parallel after M6-S3) | 4-way parallel |
| M6-S8 operational Gen2 + closeout | ⚪ queued (serial final) | 24-36h |

## M7 sprint table

| Sprint | Status |
|---|---|
| M7 plan: scout + critic + manager-amendments | ✓ all CLOSED (full ratification cycle) |
| M7-S0..S8 implementation | ⚪ queued (after M6 GREEN + M5 RRTMG PARITY) |

## M8 (queued)

| Sprint | Status |
|---|---|
| M8 forkable release | ⚪ queued |

## Big-picture critical path

```
NOW (3 parallel agents)
  ├─ S3.zz Opus → bind M5-S3.zzzz scope → S3.zzzz worker (codex) → Opus → SW PARITY
  ├─ S3.zzz worker → Opus → LW PARITY  (CRITICAL: dominant 24h T2 driver)
  └─ M6-S2 → Opus → M6-S3 → Opus → M6-S4..S7 4-way → M6-S8 → M6 GREEN

When (SW PARITY + LW PARITY): M5 RRTMG complete → M6-S8 operational T2 binding gate meaningful
When M6 GREEN: M7-S0 dispatch (plan already manager-ratified)
M7 GREEN → M8 release
```

**Calendar**: M5 RRTMG PARITY 24-48h; M6 close 5-9 days; end-goal landing ~3-4 weeks.

## File-ownership snapshot

- `src/gpuwrf/contracts/state.py`: M6-S2 extending (boundary leaves)
- `src/gpuwrf/coupling/{driver, boundary_apply}.py`: NEW M6-S2
- `src/gpuwrf/io/**`: M6-S2a CLOSED, frozen
- `src/gpuwrf/physics/rrtmg_sw.py`: M5-S3.zz Opus reviewing; future M5-S3.zzzz reopens for cldprmc/spcvmc
- `src/gpuwrf/physics/rrtmg_lw.py`: M5-S3.zzz worker editing (16 bands)
- All other physics: CLOSED, frozen

**3-way disjointness verified**: SW (rrtmg_sw) vs LW (rrtmg_lw) vs M6 (coupling+contracts). Zero conflict surface.

## Watchman policy (user-AFK)

- 20-min cadence per user 14:05
- On each tick: check 3 panes, process AGENT REPORTs, dispatch next sprint per critical path
- Maintain ≥2 parallel agents at all times
- Watch for rate-limit signals (empty stdout on codex = quota exhaust)

## Recent ticks

- 13:50 (this tick): S3.zz worker + M7 critic AGENT REPORTs landed; S3.zz Opus reviewer dispatched; M5-S3.zzzz stub written; M7 manager-amendments integrated; **M5-S3.zzz LW worker dispatched in parallel per user "max parallel" directive**
- Next: 14:25 (20-min cadence per user AFK directive)
