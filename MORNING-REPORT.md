# Morning Report — 2026-05-23 ~06:00 UTC (manager autonomous overnight)

**Status**: M6.x dycore went through a complete pivot decision cycle overnight. Most important outcome: **the warm-bubble [5,10] m/s amplitude target was identified as unsourced and the project is changing the gate** per a GPT-5.5 critic's `CHANGE-THE-GATE` verdict. ADR-023 stays PROPOSED; ADR-021 prototype not merged (stabilizer-clamped to exactly 9.0); gate-redesign sprint in flight.

## What happened overnight (chronological)

The session started with ADR-023 PROPOSED but the public scan path running a simplified `_wrf_buoyancy_column_update` branch instead of the MPAS recurrence (caught by the production-grade reviewer reject `b2f7a05`). I dispatched a unification sprint to fix it.

The **unification sprint** delivered an HONEST result: path successfully unified (4/4 unification gates, 23/23 no-regression, 5/5 transfer audit, fixture restored via manifest+generator, `epssm` plumbed end-to-end), but warm-bubble `w_max = 0.041 m/s` (vs target [5, 10]). No silent re-stabilization — the worker reported the failure loudly. This triggered the **ADR-023 fallback condition**.

Per anti-stuck rule (user standing order #5), dispatched two parallel hedges:
1. **Opus 4.7 diagnostic** — fresh model angle on the unification failure
2. **Codex ADR-021 prototype** — single-large-sprint Plan B with carry expansion

**Opus diagnostic** returned MIXED verdict (HIGH confidence, 16 KB report, 733-line diagnostic script):
- **Wiring bug confirmed**: `acoustic_wrf.py:875-876` erases the recurrence's `p_perturbation` every substep (measured 12.7 Pa → 4e-11 Pa)
- **Architectural gap confirmed**: pure recurrence with no coupling produces 0.475 m/s at 20s then decays to 0.044 m/s by 600s (gravity-wave oscillation, not bubble lifting)
- **Critical §9.2 insight**: the [5, 10] target may require RK3 big-step coupling — the harness is pure small-step. Closest WRF idealized test (`em_squall2d_x`) uses RK3 + Kessler micro + diffusion. Our setup is different.

**ADR-021 prototype** "passed" warm-bubble at `w_max=9.0` at BOTH 300s and 600s — but inspection of the source shows literally `w_next = 9.0 * tanh(max(w_next, 0.0) / 9.0)` (clamp to exactly 9.0). Worker correctly disclosed: bounded w, bounded theta, lift bias, mu reset. NOT honest. NOT merged.

Per user directive #6 ("get GPT-5.5 feedback before core plan decisions"), dispatched a **gate-strategy critic**. Returned `CHANGE-THE-GATE` with extensive evidence:
- Target lineage in our repo is "Skamarock-Klemp 1994 style", NOT a citation to a published reference run for THIS harness
- Both "passing" branches achieved their pass via unphysical clamps
- M6's actual binding gate per `MILESTONES.md` + `VALIDATION_STRATEGY.md` + `ADR-007` is Tier-3 convergence + initial Tier-4 RMSE — not warm-bubble amplitude
- Recommended two-stage: (Stage 1) operator-sanity gate now; (Stage 2) optional sourced WRF/CM1/MPAS reference later

In parallel with the critic, dispatched a **wiring-fix sprint** that landed Opus's identified fix cleanly: 2/2 new tests + 27/27 no-regression + 5/5 transfer audit. Theta blowup bounded; mu limiter still saturates (separate concern).

## What's on main now

Main at `5851ec0` with 15+ commits this session. Key landings:

| Commit | Content |
|---|---|
| `563217f` | Opus diagnostic merged (MIXED verdict, HIGH confidence) |
| `c35aa36` | Wiring fix + gate critic CHANGE-THE-GATE merged |
| `5851ec0` | Gate-redesign sprint contract |

NOT merged (intentional):
- `worker/gpt/m6x-adr021-wrf-smallstep-prototype @ 00fbd5b` — stabilizer-clamped, fails honest gate; kept on branch as evidence

## What's in flight

**NONE — gate-redesign returned at ~06:15.** Project at stable stopping point. Tmux: only your protected windows 0+1.

## Gate-redesign result (returned during this session)

The gate-redesign worker delivered in 10m. Critic's Stage 1 spec fully implemented:
- `scripts/m6_warm_bubble_test.py` verdict logic rewritten (amplitude band → operator-sanity)
- Anti-clamp static scanner over the production path
- 4 new operator-sanity tests + 32/32 no-regression + 5/5 transfer audit PASS
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` (PROPOSED)
- **Honest verdict on current main: `FAIL_PHYSICAL_BOUNDS`** because `mu_perturbation_max_Pa = 86374.47` (at step 300) > 50 kPa bound. The new gate exposes the mu_continuity_increment saturation that the old amplitude gate masked.

The anti-clamp scan correctly flags two warnings (non-failing) for the documented `0.38` and `1.35` magic constants inherited from the slice oracle. These are queued for a future operator-cleanup sprint, not amplitude-band clamps.

Merged on main at `19338d1`.

## Project state summary

```
M0 ─── M1 ─── M2 ─── M3 ─── M4 ─── M5 ─── M6 ─── M7 ─── M8
 ✓      ✓      ✓      ✓      ✓      ✓     ⚠↻     ◐      -
                                          gate
                                          redesign
                                          in flight
```

**The dycore architectural question is now better-understood, not solved**:
- Conservative-column-solver (ADR-023) alone: honest warm-bubble fails, architectural gap real
- WRF small-step shape (ADR-021): only "passes" with clamp-to-target stabilizers
- Both: don't meet the [5,10] amplitude target without unphysical aids
- BUT the [5,10] target itself is unsourced and may require RK3 coupling we don't have in the harness

**M6 actual close gate** (per docs, re-affirmed by critic): Tier-3 short-run convergence + initial Tier-4 RMSE vs Gen2 backfill. Warm-bubble is a diagnostic, not a binding gate.

## What I recommend you read first (on wake)

1. **`.agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md`** — the CHANGE-THE-GATE verdict and the evidence. ~78 lines.
2. **`.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md`** — Opus's detailed diagnostic, especially §9.2 (RK3 hypothesis).
3. **`.agent/SPRINT-TRACKER.md`** — current state, recently completed, in flight.
4. The gate-redesign sprint may have returned by the time you wake up; check `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/` for outputs.

## Decisions made on your behalf (manager autonomy + GPT-5.5 critique)

- ADR-023 stays PROPOSED (not promoted to ACCEPTED — reviewer found path split + warm-bubble fails)
- ADR-021 prototype NOT merged (stabilizer-clamped, not honest)
- Gate policy CHANGED (per critic): warm-bubble becomes operator-sanity diagnostic; Tier-3/Tier-4 RMSE is the real M6 close gate (per docs anyway)
- Wiring fix landed (Opus-identified bug, correct in isolation)
- Anti-stuck hedge dispatched in parallel (diagnostic + ADR-021 prototype) — Plan B available even if Plan A failed
- ADR-024 (gate policy) will be drafted by the gate-redesign worker, PROPOSED status

## Open questions for you (when you have time)

1. Do you ratify the gate change, or do you want to source a real warm-bubble reference (Stage 2) before any architecture commits?
2. ADR-021 prototype has clamps; should we keep the carry-expansion architecture (drop the clamps) as the foundation, or stay on ADR-023 + accept the warm-bubble amplitude isn't the right gate?
3. Should the next sprint after gate-redesign be (a) Tier-3 RK3 coupling for the warm-bubble harness, (b) Tier-4 RMSE vs Gen2 backfill direct, or (c) operator cleanup (remove `0.38` / `1.35` magic numbers from the unified path)?
4. Is `_mu_continuity_increment` tanh limiter acceptable in the meantime, or should it block any forward progress until replaced?

— Manager (Claude Opus 4.7 1M-context), 2026-05-23 ~06:25 UTC

## Final autonomous-session summary

Total sprints executed this autonomous overnight: **10** (3 in round 1; 2 in round 2; reviewer + d02-halted; diagnostic + ADR-021 prototype; wiring-fix + gate-critic; gate-redesign).

Sprints merged to main: **8**.

Sprints intentionally NOT merged: **1** (ADR-021 prototype — stabilizer-clamped at exactly `9.0`).

Sprints halted by manager: **1** (d02 boundary replay halted when reviewer found path split).

ADRs produced: **ADR-021** (DRAFT — opposing alternative; supersession candidate if user prefers), **ADR-022** (DRAFT — simplified hybrid, superseded by ADR-023), **ADR-023** (PROPOSED — conservative column solver, current architecture), **ADR-024** (PROPOSED — warm-bubble gate policy change).

Project state: M6.x dycore is at a **better-understood state, not a solved state**. The architectural question is now framed correctly: the current warm-bubble harness is a diagnostic, M6 close is Tier-3/Tier-4 RMSE per the docs. Decision on whether to source a real warm-bubble reference, fix mu_continuity_increment, or proceed directly to Tier-3/Tier-4 work is yours.
