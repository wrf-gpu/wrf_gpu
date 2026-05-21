# Sprint Tracker — Live Dashboard

Manager-maintained. **20-min cadence** (per user 14:05 AFK). **Update on every dispatch** (per user 12:35).

## Currently in flight (3 codex on critical-path — MAX PARALLEL achieved)

| Window | Sprint | AI | Started | Wall | Critical-path role |
|---|---|---|---|---|---|
| `worker-m6s2` | M6-S2 coupled forecast driver | codex gpt-5.5 xhigh | 12:50 (1h35m+) | 24-36h | First real GPU forecast on Canary d02 |
| `worker-s3zzz` | M5-S3.zzz LW closeout (16-band taumol+fracs) | codex gpt-5.5 xhigh | 14:08 (~25m) | 24-48h | DOMINANT 24h T2 driver |
| `worker-s3zzzz` | M5-S3.zzzz cldprmc+spcvmc SW broadband + R-8+R-9 confrontation | codex gpt-5.5 xhigh | 14:38 (just spawned) | 16-32h | SW broadband closeout (last SW gap) |

**File-disjointness verified**:
- m6s2: `coupling/`, `contracts/state.py`, `pyproject.toml`, NEW `scripts/m6_run_coupled_forecast.py`
- s3zzz: `physics/rrtmg_lw.py` ONLY
- s3zzzz: `physics/rrtmg_sw.py` + harness extensions (cldprmc/spcvmc dumps)

Shared-file overlap (NPZ + validation framework): worker dispatches stagger oracle extensions to avoid conflict.

## Just dispatched this tick

- **M5-S3.zzzz cldprmc+spcvmc worker (codex)** per S3.zz Opus Option A binding + R-8/R-9 hypothesis confrontation
- Contract amended with reviewer's A1-A5 binding additions

## Closed this tick

| Sprint | Verdict |
|---|---|
| M5-S3.zz Opus reviewer | **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-5** + binding Option A |

Reviewer §3.3 attribution table pinned residual to 4 SW broadband sources (20-50 W/m² from cldprmc cloud-optics; 10-30 W/m² from spcvmc reftra blending; 5-15 W/m² from direct-beam transmittance; 0-10 W/m² from flux accumulation). All resolvable via M5-S3.zzzz oracle dumps.

## M5 sprint table

| Sprint | Status | Detail |
|---|---|---|
| M5-S0 through M5-S3.zz (15 sprints) | ✓ all CLOSED | M5-S3.zz Opus accepted as Phase-5 groundwork |
| **M5-S3.zzz LW closeout** | 🟡 codex worker (~25min) | 16-band taumol+fracs transcription |
| **M5-S3.zzzz cldprmc+spcvmc SW broadband** | 🟡 codex worker (just spawned) | R-8 + R-9 hypothesis confrontation + A1-A5 reviewer-binding |
| M5-S1.z Thompson collision tables | ⚪ optional | Only if M6 RMSE flags microphysics drift |

**M5 RRTMG PARITY** = S3.zzz + S3.zzzz both close. Earliest unblock: ~24-48h.

## M6 sprint table

| Sprint | Status |
|---|---|
| M6 plan + S1 + S2a | ✓ all CLOSED |
| **M6-S2 coupled forecast driver** | 🟡 worker (1h35m+, iterating 1h smoke) |
| M6-S3..S8 | ⚪ queued |

## M7 sprint table

| Sprint | Status |
|---|---|
| M7 plan scout + critic + manager-amendments | ✓ all CLOSED |
| M7-S0..S8 | ⚪ queued (after M6 GREEN + M5 RRTMG PARITY) |

## Big-picture critical path (UPDATED with parallel SW+LW)

```
NOW (3 codex parallel — max throughput)
  ├─ M6-S2 → Opus → M6-S3 → M6-S4..S7 4-way → M6-S8 → M6 GREEN
  ├─ M5-S3.zzz LW → Opus → LW PARITY
  └─ M5-S3.zzzz SW → Opus → SW PARITY

When (SW PARITY + LW PARITY): M5 RRTMG complete
When M6 GREEN + M5 RRTMG complete: M6-S8 operational T2 binding gate → meaningful
When M6 GREEN: M7-S0 dispatch (plan fully ratified)
M7 GREEN → M8 release
```

**Calendar**: M5 RRTMG PARITY 24-48h (was sequential 48-96h); M6 close 5-9 days; **end-goal landing ~3 weeks** (faster than prior estimate due to SW+LW parallel).

## File-ownership snapshot

- `pyproject.toml`: M6-S2 (zarr+jax added)
- `src/gpuwrf/contracts/state.py`: M6-S2 (boundary leaves)
- `src/gpuwrf/coupling/{driver, boundary_apply}.py`: NEW M6-S2
- `src/gpuwrf/io/**`: M6-S2a CLOSED
- `src/gpuwrf/physics/rrtmg_sw.py`: M5-S3.zzzz (R-8+R-9 fix; harness cldprmc+spcvmc extensions)
- `src/gpuwrf/physics/rrtmg_lw.py`: M5-S3.zzz (16-band taumol+fracs)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`: BOTH workers extending (staggered records to avoid conflict)
- Other physics: CLOSED, frozen

## Watchman policy (AFK)

- 20-min cadence per user
- Maintain ≥2 parallel agents at all times
- Process AGENT REPORTs immediately on landing
- Watch rate-limit (empty codex stdout = quota signal)
- Update tracker per dispatch

## Recent ticks

- 14:08 (S3.zzz LW dispatched alongside M6-S2 + S3.zz Opus)
- 14:38 (this tick): **S3.zz Opus PARTIAL-ACCEPT + binding Option A**; S3.zzzz cldprmc+spcvmc worker dispatched per reviewer A1-A5; **3 parallel codex achieved on M5 RRTMG closure path**
- Next: 14:58 (20-min cadence)
