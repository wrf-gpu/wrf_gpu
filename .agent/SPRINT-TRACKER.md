# Sprint Tracker — Live Dashboard

Manager-maintained. 20-min cadence (AFK).

## Currently in flight (1 codex + 2 opus — within ≤3 codex policy)

| Window | Sprint | AI | Started | Wall |
|---|---|---|---|---|
| `worker-s3zzzz` | M5-S3.zzzz cldprmc+spcvmc SW broadband (R-8+R-9 hypothesis confrontation) | codex gpt-5.5 xhigh | 14:38 | 16-32h |
| `reviewer-m6s2` | M6-S2 coupled forecast driver Opus review | opus 4.7 xhigh | 14:58 | ~15-25min |
| `reviewer-s3zzz` | M5-S3.zzz LW closeout Opus review | opus 4.7 xhigh | 14:58 | ~15-25min |

**Codex count: 1 (within 3-cap). Opus reviewers don't count.** 

## Closed this tick (2 worker AGENT REPORTs)

| Sprint | Verdict | Status |
|---|---|---|
| **M6-S2 coupled forecast driver** | worker PASS all 7 AC + bundled 5 M6-S1 prereqs | merged `464cbc3`; Opus review IN FLIGHT |
| **M5-S3.zzz LW closeout** | worker PARTIAL: 16/16 LW bands taug+fracs PASS intermediate-oracle; Tier-1 broadband still FAILS (root cause: cldprmc_lw + rtrnmc) | merged; Opus review IN FLIGHT |

### M6-S2 highlights
- 1h + 6h + 24h forecasts PASS on real d02 (160×67×45)
- 0 H2D / 0 D2H / 0 host-device-transfer bytes (MEASURED via JAX profiler trace, not literal)
- `temporary_bytes_per_step = 136890408` (130 MB, measured via XLA `compiled.memory_analysis()`)
- 23/23 M6 tests pass
- pyproject.toml `zarr+jax` added (M6-S2a Opus follow-up closed)
- All 5 M6-S1 prereqs addressed: R-3 FP32 Path A, R-5 real GridSpec dz, R-7 measured temp bytes, R-9 robust cadence, R-13 boundary State extension
- WRF-style boundary apply (specified + relaxation zone) per `module_bc.F` citations
- ADR-010 amended
- Honest unresolved-risks: 1s internal dycore guard, finite-state guard (residency proof not physical validation), mu_bdy first-step replay, WRF-shaped NPZ instead of NetCDF

### M5-S3.zzz LW highlights
- 16/16 LW bands FULL_BRANCH_ACCEPTED at intermediate-oracle
- Per-band table with WRF source citations for each band
- LW launches 43 (lax.scan barrier present, no full fusion)
- Combined raw launches 97
- Root cause now: LW cldprmc + rtrnmc transfer/source (analog to SW M5-S3.zz finding)
- Worker recommends M5-S3.zzzzz LW cldprmc+rtrnmc oracle (file-disjoint with SW S3.zzzz — could parallel)

## M5 sprint table (live)

| Sprint | Status |
|---|---|
| M5-S0 through M5-S3.zzz (16 sprints) | ✓ all CLOSED at worker level |
| **M5-S3.zzzz cldprmc+spcvmc SW broadband** | 🟡 codex worker IN FLIGHT (~50min in) |
| **M5-S3.zzz LW** | 🟡 Opus review IN FLIGHT (worker just closed) |
| M5-S3.zzzzz LW cldprmc+rtrnmc oracle (NEW per S3.zzz worker rec) | ⚪ contract pending; can dispatch parallel after S3.zzz Opus accepts |
| M5-S1.z Thompson collision tables | ⚪ optional |

## M6 sprint table (live)

| Sprint | Status |
|---|---|
| M6 plan + S1 + S2a | ✓ all CLOSED |
| **M6-S2 coupled forecast driver** | 🟡 Opus review IN FLIGHT (worker just closed) |
| M6-S3 surface + Noah-MP | ⚪ contract READY; dispatch after M6-S2 Opus accepts |
| M6-S4..S7 + S8 | ⚪ queued |

## M7 sprint table

| Sprint | Status |
|---|---|
| M7 plan: scout + critic + manager-amendments | ✓ all CLOSED |
| M7-S0..S8 | ⚪ queued |

## Big-picture critical path

```
NOW (1 codex + 2 opus reviews)
  → Opus reviews → manager closeouts → next dispatch:
     ├─ M5-S3.zzzzz LW cldprmc+rtrnmc (codex, after S3.zzz Opus)
     ├─ M6-S3 surface+Noah-MP (codex, after M6-S2 Opus)
     └─ S3.zzzz still running (SW broadband closeout)
  → 3 codex parallel again (s3zzzz + s3zzzzz LW + m6s3 surface) once Opus reviews land
```

**Calendar refined**: M5 RRTMG full PARITY now needs S3.zzzz (SW broadband) + S3.zzzzz (LW broadband) = 2 more sprints (16-32h each, file-disjoint can parallel). Closer to end-goal than expected.

## File-ownership snapshot

- `coupling/{driver, boundary_apply}.py`: M6-S2 LANDED
- `contracts/state.py`: boundary leaves landed
- `physics/rrtmg_sw.py`: M5-S3.zzzz worker active
- `physics/rrtmg_lw.py`: M5-S3.zzz LANDED; M5-S3.zzzzz will reopen for cldprmc+rtrnmc
- `io/**`: M6-S2a frozen
- Other physics: CLOSED, frozen

## Recent ticks

- 14:38: S3.zz Opus + M5-S3.zzzz dispatched (3 codex)
- 14:58 (this tick): M6-S2 worker DONE + M5-S3.zzz LW worker DONE; both AGENT REPORTs landed; 2 Opus reviewers dispatched in parallel; codex count drops to 1 (s3zzzz)
- Next: 15:18 (20-min cadence)
