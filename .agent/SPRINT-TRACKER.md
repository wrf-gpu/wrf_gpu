# Sprint Tracker — Live Dashboard

Manager-maintained. 30-min cadence per user.

## Currently in flight (2 codex workers with WATCHDOG launchers)

| Window | Sprint | AI | Started | Wall budget | Notes |
|---|---|---|---|---|---|
| `worker-s3z` | M5-S3.z RRTMG intermediate-oracle extraction + per-band validation + LW completion + SW fusion | codex gpt-5.5 xhigh | 12:00 | 24-48h | Phase-3 RRTMG sprint per M5-S3.y reviewer §5 binding methodology |
| `worker-m6s2a` | M6-S2a Gen2 backfill accessor + d02 boundary replay + shared validation I/O + CPU denominator + proof-object schemas + ADR-011 | codex gpt-5.5 xhigh | 12:00 | 12-18h | Critic-mandated infrastructure for ALL M6-S2..S8 |

Both file-disjoint (rrtmg_* vs io/). Both have WATCHDOG launchers (old single-Enter; manager will manually Enter their AGENT REPORTs if needed).

## Closed this tick (2 — M5 prologue Phase-2 + M6 first-impl)

| Sprint | Verdict | Wall (worker + Opus) |
|---|---|---|
| **M5-S3.y RRTMG setcoef/taumol/Planck-attempt-1** | Opus **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3** | 24m + ~10m |
| **M6-S1 coupled interface freeze** | Opus **ACCEPT-WITH-MINOR-FOLLOWUPS** (12 PASS / 5 FOLLOWUP / 0 REJECT) | 22m + 9m |

**4 permanent M5-S3.y artifacts preserved**: Eddington oracle (kmodts=1), native tables, `_sw_setcoef`, LW Planck-source.
**M6 implementation UNBLOCKED**: M6-S2 dispatch waits on M6-S2a; will bundle 5 prerequisites (R-3/R-5/R-7/R-9/R-13) when contract written.

## Auto-notify fix (this tick)

**Two-stage fix**:
1. **Watchdog** (skill update `fd5c214`): polls report file → 60s stability → force-tap Enter + /exit + kill window + fire AGENT REPORT. Fired clean on M5-S3.y reviewer. M6-S1 reviewer required manual Enter from user (single Enter unreliable).
2. **Multi-Enter** (skill update `8c5a261`): 3 Enters with 2-3s delays empirically required to reliably submit pasted text in Claude Code prompt. Single Enter sometimes silently fails. **All future launchers will use multi-Enter.**

Currently-running s3z + m6s2a watchdogs were forked with OLD single-Enter pattern; manager will manually Enter if their AGENT REPORTs need help. All future dispatches use upgraded pattern.

## Sprint contracts queued for next swarm (after s3z + m6s2a close)

| Sprint | Wall | Owns | Blocked on |
|---|---|---|---|
| **M5-S3.z Opus reviewer** | ~10m | reviewer-report.md | s3z worker AGENT REPORT |
| **M6-S2a Opus reviewer** | ~10m | reviewer-report.md | m6s2a worker AGENT REPORT |
| **M6-S2 coupled forecast driver** (12-18h) | bundle R-3/R-5/R-7/R-9/R-13 + uses boundary replay from S2a + uses real GridSpec metrics + 1h smoke → 6h → 24h d02 | M6-S2a ACCEPT |
| **M6-S3 surface layer + bounded Noah-MP minimum** (30-48h) | NEW physics; uses M6-S2a accessor for Gen2 surface fixtures | M6-S2 smoke; can run parallel with S4/S5/S6/S7 |

## After that swarm (validation phase, 4-way parallel)

| Sprint | Wall |
|---|---|
| M6-S4 Tier-2 coupled invariants (external/cross-implementation oracle) | 16-24h |
| M6-S5 ADR-007 4× verdict (uses M6-S2a CPU denominator + FAIL fallback ladder) | 12-20h |
| M6-S6 Tier-3 TSC1.0 (controlled dt-refinement) | 18-30h |
| M6-S7 Tier-4 probtest (stratified by land/sea/elevation) | 18-30h |

## Final M6 close

| Sprint | Wall |
|---|---|
| M6-S8 operational Gen2 + closeout (CPU-vs-obs binding gate; GREEN/PARTIAL/BLOCKED/FAIL statuses) | 24-36h |

## Big-picture path to PROJECT_CONSTITUTION end goal

```
NOW: 2 codex workers in flight (s3z + m6s2a)
  ↓ ~12-48h
M5-S3.z RRTMG PARITY + M6-S2a infrastructure close
  ↓ Opus reviewers
M6-S2 + M6-S3 parallel dispatch (30-48h)
  ↓
M6-S4/S5/S6/S7 4-way parallel validation (18-30h)
  ↓
M6-S8 operational closeout (24-36h)
  ↓ GREEN
M7 Canary operational v0 (3km then 1km pipeline, daily-run, ops verification)
  ↓ GREEN
M8 forkable release (docs, packaging, public review)
```

**Calendar**: M6 close 7-10 days; end-goal landing ~4-5 weeks.

## File-ownership snapshot (post-M6-S1)

- `src/gpuwrf/contracts/state.py, precision.py`: FROZEN by M6-S1 (modulo R-13 boundary-handle extension by M6-S2)
- `src/gpuwrf/coupling/physics_couplers.py`: FROZEN by M6-S1
- `src/gpuwrf/io/**`: M6-S2a OWNS (NEW module)
- `src/gpuwrf/physics/thompson_*, mynn_*`: M5-S1.y/S2.x CLOSED, frozen
- `src/gpuwrf/physics/rrtmg_*`: M5-S3.z REOPENS (in flight)
- `src/gpuwrf/physics/surface_layer.py, noah_mp.py`: NEW, M6-S3 owns (queued)
- `src/gpuwrf/dynamics/**`: M4 frozen
- `src/gpuwrf/validation/tier{2,3,4}_coupled.py`: M6-S4/S6/S7 own respectively (queued)
- `scripts/m6_*.py`: per-sprint ownership in ADR-010

## Watchman policy

- 30-min cadence per user (next ~12:35)
- On each tick: check 2 codex panes; if AGENT REPORT arrived → dispatch Opus reviewer; update tracker
- Rate-limit: 2 codex active = within budget; opus quota separate

## Recent ticks

- 2026-05-21 11:11-11:12 — 3 codex (s3y + m6s1 + critic) dispatched
- 2026-05-21 11:42 — watchman #4: all 3 finished stuck-at-/exit → watchdog fix encoded; Opus reviewers for s3y + m6s1 dispatched
- 2026-05-21 12:00-12:10 — watchman #5 (user-triggered via AGENT REPORT):
  - M5-S3.y Opus PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3 → merged; M5-S3.z stub updated with reviewer §5 binding scope
  - M6-S1 Opus ACCEPT-WITH-MINOR-FOLLOWUPS (5 prerequisites) → merged + closeout
  - User flagged single-Enter unreliable → multi-Enter pattern encoded in skill file
  - **M5-S3.z + M6-S2a workers dispatched in parallel** with watchdog (old single-Enter; manager will manually Enter if needed)
- Next: 30-min tick at ~12:35
