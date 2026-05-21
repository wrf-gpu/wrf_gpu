# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick.

**Per user 2026-05-21 ~11:10**: keep 30-min watchman cadence; next swarm dispatched.

## Currently in flight (3 codex)

| Window | Sprint | Phase | Started | Wall budget | Status |
|---|---|---|---|---|---|
| `worker-s3y` | M5-S3.y RRTMG setcoef+taumol+Planck | worker (codex gpt-5.5 xhigh) | 11:11 | 16-32h | Working — blocked M5 close depends on this |
| `worker-m6s1` | M6-S1 coupled interface freeze | worker (codex gpt-5.5 xhigh) | 11:11 | 12-18h | Working — first M6 implementation sprint |
| `critic-m6plan` | M6 milestone plan consensus critique | critical-reviewer (codex gpt-5.5 xhigh) | 11:11 | 30-60min | Working — plan ratification before M6-S2..S8 dispatch |

## Manager decision recorded (this tick)

- **Eddington-vs-PIFM**: option (a) — patch local WRF `kmodts=1` + rebuild harness. Preserves M5-S3.x Eddington implementation. Worker AC0 = first step.

## Closed this tick (1)

| Sprint | Verdict | Merge | Closeout |
|---|---|---|---|
| **M5-S1.y Thompson HLO + residuals** | Opus **ACCEPT-AS-GRAY-ZONE-CHECKPOINT** | `0bd1fd2` (worker + reviewer `bbabd32`) | M6-S1 UNBLOCKED-WITH-DEBT; optional M5-S1.z follow-up if M6 RMSE flags |

## M5 prologue close status

| Sprint | Status |
|---|---|
| M5-S1.y Thompson | ✓ CLOSED (GRAY-ZONE-CHECKPOINT) |
| M5-S2.x MYNN | ✓ CLOSED (ACCEPT) |
| M5-S3.x RRTMG transfer-solver | ✓ CLOSED (GROUNDWORK-PHASE-2) |
| **M5-S3.y RRTMG setcoef+taumol+Planck** | in flight (codex) |

## M6 sprint status

| Sprint | Status |
|---|---|
| **M6-S1 coupled interface freeze** | in flight (codex) — UNBLOCKED today |
| M6 plan critical-review | in flight (codex) — needed before M6-S2..S8 dispatch |
| M6-S2 forecast driver | queued — after M6-S1 + plan ratification |
| M6-S3 surface + Noah-MP minimum | queued — after M6-S2 smoke; M5-S2.x interface stub ready |
| M6-S4 Tier-2 coupled invariants | queued — after M6-S2 smoke; parallelizable with S3/S5/S6/S7 |
| M6-S5 ADR-007 4× verdict | queued — after M6-S2 smoke; parallelizable |
| M6-S6 Tier-3 TSC1.0 | queued — after M6-S2 + S4 |
| M6-S7 Tier-4 probtest | queued — after M6-S1 |
| M6-S8 operational Gen2 + closeout | queued — after S0-S7 |

## Watchman policy

- **30-min cadence per user directive**
- On each tick: check 3 codex panes, dispatch Opus reviewers on worker AGENT REPORTs, update tracker
- Next tick: ~11:42

## Recent ticks

- 2026-05-21 09:30-09:37 — 3 codex workers dispatched (s1y, s2x, s3x)
- 2026-05-21 10:10-10:18 — watchman tick #1: P2+P3 workers done, Opus reviewers dispatched
- 2026-05-21 10:38-10:43 — watchman tick #2: P2 ACCEPT + P3 GROUNDWORK-PHASE-2 merged; M5-S3.y stub created
- 2026-05-21 11:01-11:12 — watchman tick #3 (user-triggered):
  - P1 Opus reviewer ACCEPT-AS-GRAY-ZONE-CHECKPOINT → merge + closeout
  - **Next swarm dispatched**: M5-S3.y (16-32h codex), M6-S1 (12-18h codex), M6 plan critic (30-60min codex)
- Next: 30-min tick at ~11:42

## File-ownership disjointness (3 parallel agents)

- `worker-s3y`: `src/gpuwrf/physics/rrtmg_*`, `scripts/wrf_rrtmg_*`, `module_ra_rrtmg_sw.F` patch, `ADR-009`
- `worker-m6s1`: `src/gpuwrf/contracts/state.py`, `precision.py`, NEW `coupling/`, NEW `tests/test_m6_*`, NEW `ADR-010`, NEW `scripts/m6_run_dummy_coupled.py`
- `critic-m6plan`: READ-only — writes `.agent/sprints/2026-05-21-m6-milestone-plan-scout/critical-review-codex.md`

All three disjoint. Safe parallel.

## Rate-limit watch

3 codex (gpt-5.5 xhigh) simultaneous. User flagged untested rate-limit at 3x earlier (2026-05-21 09:35). First run worked. Monitor for empty-output / 429-equivalent signals on first tick.
