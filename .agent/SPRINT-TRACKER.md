# Sprint Tracker — Live Dashboard

Manager-maintained. Updated every watchman tick. Source of truth for parallel-management state.

**Per user directive 2026-05-21 ~09:35**: M6 prologue full-steam-ahead; rate-limit watch on gpt-5.5.

## Currently in flight (3 parallel: 1 codex + 2 opus)

| Window | Sprint | AI | Phase | Started | Status |
|---|---|---|---|---|---|
| `worker-s1y` | M5-S1.y Thompson | codex gpt-5.5 xhigh | worker | 09:35 | running pytest validation (35m) — implementation likely complete, awaiting suite |
| `reviewer-s2x` | M5-S2.x MYNN | claude-opus-4-7 xhigh | reviewer | 10:15 | thinking/transmuting; verifying AC1 budget probe + AC2 radicand + AC3 surface-layer interface |
| `reviewer-s3x` | M5-S3.x RRTMG | claude-opus-4-7 xhigh | reviewer | 10:15 | thinking/ionizing; verdict expected: REJECT-bounded / ACCEPT-AS-GROUNDWORK-PHASE-2 |

## Worker reports received this tick (10:10)

| Sprint | Worker | Wall | Verdict signal | Commit |
|---|---|---|---|---|
| M5-S2.x MYNN | codex | 27m 03s | self-clean: AC1-AC5 pass; full pytest blocked by `/tmp` disk-space env failures (unrelated) | `7f9f4f1` |
| M5-S3.x RRTMG | codex | 29m 03s | **self-flag: "do not close as accepted"**; strict Tier-1 SW+LW fail; 40 raw launches > 10 cap; new Eddington pieces sound but full `setcoef + taumol` + LW Planck-source machinery missing | `cbce2e5` |

## Key M5-S2.x signals (worker self-report)

- AC1 independent budget probe: max residuals u=2.5e-5, v=9.3e-6, theta=1.6e-6, qv=2.4e-10 (target ≤1e-3)
- AC2: Path A chosen — unguarded SQRT matching WRF source `module_bl_mynnedmf.F90:1918-1919`
- AC3: `surface_layer(state) -> SurfaceFluxes` hook + ADR-008 interface section with WRF citations
- AC4: 35 raw launches, no fudge, 0 transfers, HLO 279 KB
- AC5: 11/11 MYNN tests pass; full pytest has unrelated env failures (out-of-scope per file ownership)

## Key M5-S3.x signals (worker self-report)

- Removed fabricated `tau_gas = vapor_path * 0.01 * log1p(gas_coeff)` ✓
- Added Joseph-Wiscombe-Weinman delta-scaling + Eddington layer reflectance/transmittance ✓
- Added LW g-point recurrence + WRF molecular column construction ✓
- **MISSING**: full `setcoef + taumol` per-band gas-absorption interpolation, full LW Planck-source machinery
- 40 raw launches (cap was 10), HLO 497 KB SW + 137 KB LW (within 500 KB ceiling)
- Tier-2 invariants pass, Tier-1 strict fail
- Honest gate: FALLBACK, no `min(raw, cap)` fudge

## On deck (queued)

| Sprint | Trigger | Notes |
|---|---|---|
| Manager closeout for s2x, s3x | After respective Opus reviewer verdicts | Likely s2x ACCEPT or ACCEPT-WITH-MINOR-FOLLOWUPS; s3x REJECT-bounded or ACCEPT-AS-GROUNDWORK-PHASE-2 (manager will need M5-S3.y stub) |
| Manager closeout for s1y | After P1 worker delivers + Opus reviewer | Codex still running pytest validation |
| P4 M6 plan consensus (codex critical-review) | After ≥1 codex worker frees gpt-5.5 quota | Can dispatch now since codex is only on P1 |
| M6-S1 coupled interface freeze | After all 3 prologue close + P4 ratified | Serial |
| M6-S2..S8 | Per plan scout sequence | Mostly parallelizable after S2 smoke |

## Watchman policy

- Next tick: 20 min (10:55) — reviewers expected to finish ~10:30-10:45, P1 worker likely wrapping pytest
- On reviewer AGENT REPORT: read verdict, dispatch manager closeout commit, update tracker
- If P1 worker AGENT REPORT lands: dispatch Opus reviewer for s1y

## Recent ticks

- 2026-05-21 09:30-09:37 — 3 codex workers dispatched (s1y, s2x, s3x); user authorized sandbox spawn
- 2026-05-21 10:10 — watchman tick; P2 worker done (27m), P3 worker done (29m, self-flagged not-acceptable), P1 still running fixture build
- 2026-05-21 10:13 — sent `/exit Enter Enter` to P2+P3 panes; AGENT REPORTs fired
- 2026-05-21 10:15 — dispatched Opus reviewer for s2x + s3x in parallel
- 2026-05-21 10:18 — tracker updated; next watchman scheduled 10:55
