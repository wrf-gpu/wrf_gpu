# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence. **Update on every dispatch per user 12:35.**

## Currently in flight (2 codex workers, all using multi-Enter watchdog)

| Window | Sprint | AI | Started | Wall | File ownership |
|---|---|---|---|---|---|
| `worker-s3zz` | M5-S3.zz RRTMG SW closeout (sfluxzen + setcoef precision + lax.scan SW fusion) | codex gpt-5.5 xhigh | 12:48 | 8-16h | `src/gpuwrf/physics/rrtmg_sw.py`, validation/intermediate_oracles, ADR-009 amend |
| `worker-m6s2` | M6-S2 coupled forecast driver (1h→6h→24h on d02) | codex gpt-5.5 xhigh | 12:50 | 24-36h | NEW `coupling/{driver, boundary_apply}.py`, `state.py` boundary leaves, `physics_couplers` GridSpec, `m6_run_coupled_forecast.py`, pyproject.toml (STEP 0) |

File-disjoint. Both have multi-Enter watchdog (Claude Code's 3-Enter pattern).

## Closed this tick (3 sprints + 1 scout — biggest tick yet)

| Sprint | Verdict | Wall (worker + reviewer/critic) | Merge |
|---|---|---|---|
| **M5-S3.z RRTMG intermediate-oracles** | Opus **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4** | 24m + ~15m | merged; M5-S3.zz Option 1 binding |
| **M6-S2a Gen2 accessor + d02 boundary replay + shared I/O** | Opus **ACCEPT-WITH-MINOR-FOLLOWUPS** (12P/5F/0R) | ~25m + ~18m | merged; M6-S2..S8 UNBLOCKED |
| **M7 milestone plan scout** | DONE (34 KB plan written) | 9m 51s | branch `scout/codex/m7-milestone-plan`; critic + manager integration queued |

## M5 sprint table (UPDATED)

| Sprint | Worker | Reviewer | Verdict | Status |
|---|---|---|---|---|
| M5-S0 scout → M5-S2.x MYNN follow-ups | (see prior tick — 7 sprints closed) | | | ✓ all CLOSED |
| M5-S1.y Thompson HLO + residuals | codex 1 | Opus | ACCEPT-AS-GRAY-ZONE-CHECKPOINT | ✓ CLOSED |
| M5-S3.x RRTMG Eddington transfer-solver | codex 1 | Opus | ACCEPT-AS-GROUNDWORK-PHASE-2 | ✓ CLOSED |
| M5-S3.y RRTMG setcoef+taumol+Planck-1 | codex 1 | Opus | PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3 | ✓ CLOSED |
| M5-S3.z RRTMG intermediate-oracles | codex 1 | Opus | PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4 | ✓ CLOSED (just now) |
| **M5-S3.zz RRTMG SW closeout** | codex 1 | pending | TBD | 🟡 worker IN FLIGHT |
| M5-S3.zzz RRTMG LW closeout | queued — after S3.zz | — | — | ⚪ queued (advance-bound to Option 2) |
| M5-S1.z Thompson collision tables | optional | — | — | ⚪ optional |

## M6 sprint table (UPDATED)

| Sprint | Status | Wall |
|---|---|---|
| M6 plan scout + critic + manager-amendments | ✓ all CLOSED | 17m + 8m + manager |
| M6-S1 coupled interface freeze | ✓ CLOSED (ACCEPT-WITH-MINOR-FOLLOWUPS, 12P/5F/0R) | 22m + 9m |
| M6-S2a Gen2 accessor + boundary replay + shared I/O | ✓ CLOSED (ACCEPT-WITH-MINOR-FOLLOWUPS, 12P/5F/0R) | 25m + 18m |
| **M6-S2 coupled forecast driver** | 🟡 worker IN FLIGHT (Phase 0: pyproject.toml zarr+jax then 1h smoke → 6h → 24h) | 24-36h |
| M6-S3 surface layer + bounded Noah-MP | queued — depends on M6-S2 smoke | 30-48h |
| M6-S4 Tier-2 coupled invariants | READY (per M6-S2a Opus) | 16-24h |
| M6-S5 ADR-007 4× verdict | READY (with `-r4` precision + denominator basis prereqs) | 12-20h |
| M6-S6 Tier-3 TSC1.0 | READY | 18-30h |
| M6-S7 Tier-4 probtest prototype | READY | 18-30h |
| M6-S8 operational Gen2 + closeout | queued — serial final | 24-36h |

## M7 sprint table (NEW)

| Sprint | Status | Wall |
|---|---|---|
| **M7 milestone plan scout** | ✓ DONE (34 KB plan written, branch `scout/codex/m7-milestone-plan`) | 10m |
| M7 plan critic | queued — codex critical-review of scout's plan | 30-60min |
| M7 plan manager-amendments | queued — manager integration after critic | 30min |
| M7-S0..Sn Canary operational v0 | queued — defined by integrated M7 plan | TBD |

## M8 (queued, post-M7)

| Sprint | Status |
|---|---|
| M8 forkable release | queued |

## What can run in parallel NOW (utility analysis)

| Opportunity | Status | Dispatch when |
|---|---|---|
| M7 plan critic (codex critical-review of scout plan) | READY | NOW or next tick (uses idle codex capacity) |
| M5-S1.z Thompson collision tables | Held | Only if M6 RMSE flags microphysics drift |
| Skill-patch consolidation | Held | Lower priority than M7 plan track |
| Gemini bug-chase | Quota-conserved for M7 ops | — |

**Decision**: at next watchman, if rate-limit allows, dispatch M7 plan critic. Currently: 2 codex active (s3zz + m6s2). Adding M7 critic = 3 codex parallel = our tested limit. Could dispatch now to keep momentum.

## Big-picture path to PROJECT_CONSTITUTION end goal

```
NOW (2 codex + 1 done) → +8-16h: M5-S3.zz close → +Opus → M5-S3.zzz dispatch (LW closeout)
NOW (2 codex)         → +24-36h: M6-S2 close → +Opus → M6-S3 dispatch (surface+Noah-MP)
After M5-S3.zzz close → +Opus → M5 prologue Phase-5 FINAL CLOSE → RRTMG PARITY
After M6-S3 close     → M6-S4/S5/S6/S7 4-way parallel (18-30h) → M6-S8 closeout (24-36h)
M6 GREEN → M7 implementation (plan integration ready) → M8 forkable release
```

**Calendar update**: M6 close 4-7 days from now (faster — M6-S2a already done; M5-S3.zz is smaller); end-goal landing **~3-4 weeks**.

## File-ownership snapshot

- `src/gpuwrf/contracts/state.py`: M6-S2 extending now (boundary leaves)
- `src/gpuwrf/contracts/precision.py`: FROZEN
- `src/gpuwrf/coupling/{physics_couplers, __init__}.py`: M6-S1 froze; M6-S2 threading GridSpec
- `src/gpuwrf/coupling/{driver, boundary_apply}.py`: NEW, M6-S2 owns
- `src/gpuwrf/io/**`: M6-S2a OWNS (closed)
- `src/gpuwrf/physics/thompson_*, mynn_*`: CLOSED, frozen (M5-S1.z optional reopen)
- `src/gpuwrf/physics/rrtmg_sw.py`: M5-S3.zz reopening (SW closeout)
- `src/gpuwrf/physics/rrtmg_lw.py`: FROZEN until M5-S3.zzz
- `src/gpuwrf/physics/{surface_layer, noah_mp}.py`: NEW, M6-S3 owns (queued)
- `src/gpuwrf/dynamics/**`: M4 frozen
- `pyproject.toml`: M6-S2 adding zarr+jax (STEP 0 of worker)

## Watchman policy

- 30-min cadence per user
- Next ~13:25
- M7 plan critic dispatch decision at next tick (or now if rate-limit clear)

## Recent ticks (compact)

- 12:08-12:40: M5-S3.z worker → Opus closeout; M6-S1 closeout; M5-S3.y closeout; M5-S3.z worker + M6-S2a worker dispatched; M6-S2a Opus dispatched; M5-S3.z Opus dispatched; M7 plan scout dispatched
- 12:40-12:55 (this tick): M5-S3.z Opus + M6-S2a Opus + M7 scout ALL closed; M5-S3.zz + M6-S2 codex workers dispatched (2 parallel, multi-Enter watchdog)
- Next: 13:25
