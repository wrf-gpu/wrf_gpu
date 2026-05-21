# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence. **Update on every dispatch per user 12:35.**

## Currently in flight (3 codex — all multi-Enter watchdog)

| Window | Sprint | AI | Started | Wall | File ownership |
|---|---|---|---|---|---|
| `worker-s3zz` | M5-S3.zz RRTMG SW closeout (sfluxzen + setcoef precision + lax.scan SW fusion) | codex gpt-5.5 xhigh | 12:48 | 8-16h | `physics/rrtmg_sw.py`, intermediate_oracles, ADR-009 amend |
| `worker-m6s2` | M6-S2 coupled forecast driver (1h→6h→24h on d02) | codex gpt-5.5 xhigh | 12:50 | 24-36h | NEW `coupling/{driver, boundary_apply}.py`, state.py boundary leaves, physics_couplers GridSpec, `m6_run_coupled_forecast.py`, pyproject.toml |
| `critic-m7plan` | M7 milestone plan critical-review | codex gpt-5.5 xhigh | 13:05 | 30-60min | `critical-review-codex.md` (READ-only otherwise) |

**Rate-limit watch**: 3 codex parallel — at user's noted limit. If any agent errors with empty stdout, that's quota-exhaust signal.

## Closed this tick (continuing the big push)

| Sprint | Verdict | Action |
|---|---|---|
| **M7 plan scout** | DONE (34 KB plan delivered, 9 sprints S0-S8 with concrete schemas + AIFS poll + conditional 1km gate) | Plan committed to main `4a0815a`; critic dispatched now |

## M7 plan scout highlights (just landed)

- **9 sprints**: S0 prologue (12-18h) → S1 3km daily (36-60h critical) → {S2 1km audit 12-20h + S4 post 18-30h + S5 verification 24-36h + S6 restart 18-30h + S7 monitoring 12-24h parallel} → S3 1km if S2 passes (36-72h) OR deviation doc (12-18h) → S8 closeout (12-24h)
- **AIFS poll** 01:25-05:25 UTC mirroring Gen2 `poll_live_18z_cycle_v1.py`
- **1km conditional gate** at S2 (RTX 5090 32GB memory/compile audit)
- **18Z init only** for v0; no "00Z in / 06Z publish" SLA claim (Gen2-correct)
- **Cold-start each cycle** (no HBM persistence assumed)
- **Calendar**: 5-8 days for 3km-only; 8-12 days for 3km+1km
- 10 explicit hard blockers and 10 M7-specific risks

## M5 sprint table (live)

| Sprint | Status |
|---|---|
| M5-S0..S3-A3, S1.y, S3.x, S3.y, S3.z | ✓ all CLOSED (11 sprints) |
| **M5-S3.zz RRTMG SW closeout** | 🟡 worker IN FLIGHT |
| M5-S3.zzz RRTMG LW closeout | ⚪ queued |
| M5-S1.z Thompson collision tables | ⚪ optional |

## M6 sprint table (live)

| Sprint | Status |
|---|---|
| M6 plan {scout, critic, manager} | ✓ closed |
| M6-S1, M6-S2a | ✓ closed |
| **M6-S2 coupled forecast driver** | 🟡 worker IN FLIGHT |
| M6-S3 surface + Noah-MP | ⚪ queued (after M6-S2 smoke) |
| M6-S4..S7 (Tier-2/3/4 + 4× verdict) | ⚪ queued (parallel after M6-S3) |
| M6-S8 operational + closeout | ⚪ queued (serial final) |

## M7 sprint table (live, NEW)

| Sprint | Status |
|---|---|
| M7 plan scout | ✓ CLOSED (34KB plan committed `4a0815a`) |
| **M7 plan critic** | 🟡 codex IN FLIGHT |
| M7 plan manager-amendments | ⚪ queued (after critic) |
| M7-S0..S8 implementation | ⚪ queued (after manager amendments + M6 close) |

## M8 (queued)

| Sprint | Status |
|---|---|
| M8 forkable release | ⚪ queued (after M7 close) |

## Big-picture critical path

```
NOW (3 codex)
  ├─ M5-S3.zz (8-16h) → Opus → M5-S3.zzz (LW) → Opus → M5 RRTMG PARITY
  ├─ M6-S2 (24-36h) → Opus → M6-S3 (30-48h) → Opus → M6-S4..S7 4-way parallel → M6-S8 closeout
  └─ M7 critic (30-60min) → manager-amendments → M7-S0 ready to dispatch
M6 GREEN + M5 RRTMG PARITY → M6 operational meaningful → M7-S0 dispatch
M7 GREEN → M8
```

**Calendar**: M6 close 4-7 days; end-goal landing ~3-4 weeks.

## File-ownership snapshot

- `pyproject.toml`: M6-S2 amending (STEP 0)
- `src/gpuwrf/contracts/state.py`: M6-S2 extending (boundary leaves)
- `src/gpuwrf/coupling/{physics_couplers}.py`: M6-S2 threading GridSpec
- `src/gpuwrf/coupling/{driver, boundary_apply}.py`: NEW M6-S2
- `src/gpuwrf/io/**`: M6-S2a closed (frozen)
- `src/gpuwrf/physics/rrtmg_sw.py`: M5-S3.zz reopened (SW closeout)
- `src/gpuwrf/physics/rrtmg_lw.py`: FROZEN until M5-S3.zzz
- `src/gpuwrf/physics/{thompson_*, mynn_*}`: CLOSED, frozen
- `.agent/sprints/2026-05-21-m7-milestone-plan-scout/critical-review-codex.md`: M7 critic writing

## Recent ticks

- 12:55: M5-S3.z + M6-S2a Opus accepts + M7 scout done; M5-S3.zz + M6-S2 codex dispatched
- 13:05 (this tick): M7 plan scout delayed AGENT REPORT (single-Enter — manager error in launcher); M7 plan critic dispatched (3rd codex parallel)
- Next watchman: 13:35 (30-min cadence)
