# c1-A1 Manager Closeout — Klemp-Skamarock Acoustic Core LANDED; Coupled FAIL

**Sprint**: M6.x-c1-A1 Klemp-Skamarock Clean-Room
**Status**: **CLOSED A1 — Acoustic core IMPLEMENTED + verified; advection/coupling FAIL → c1-A2 dispatched**
**Date**: 2026-05-22 ~01:25
**Worker**: codex gpt-5.5 xhigh (~48 min — MUCH faster than 5-9 day estimate)
**Wall**: 47m 57s
**Reviewer**: not yet dispatched (waiting for c1-A2 completion before reviewer judges full c1 cycle)

## Headline

c1 worker IMPLEMENTED Klemp 2007 §3a-c clean-room in 48 minutes (vs 5-9 day estimate). Acoustic core works in isolation:
- 60-step acoustic-only Gen2 d02 scan: STABLE with `n_acoustic=86`
- 18-step coupled probe: zero sanitize firing
- All unit tests PASS
- NEW: `tridiag.py`, `ADR-018-tridiag-backend.md`, `ADR-019-klemp-skamarock-clean-room.md`
- Acoustic.py rewritten with +354 lines

**BUT**: 1h coupled probe still FAILS — even with radiation disabled. Diagnosis: NOT in acoustic. In advection + coupling interaction with new c1 pressure formulation.

## Critical evidence

- Coupled-with-advection scan nonfinite at step 30 (acoustic-only doesn't)
- `test_mass_scalar_advection_is_conservative_for_constant_velocity` FAIL at 8.43e-05 vs 1e-10 (6 OOM violation)
- Bug-hunt #2 §2 pre-identified the operators: `_dz_from_state` mean-dz bias, `_mass_to_u_face` jnp.roll periodicity
- c1-A1 worker self-diagnosis aligns

## What c1-A1 changed

- `dynamics/acoustic.py` (+354 LoC): Klemp 2007 §3a-c with diagnostic pressure + tridiag w-ph
- `dynamics/tridiag.py` (NEW): per-column Thomas solve
- `dynamics/rk3.py`, `step_debug_stripped.py`: mu plumbing for c1
- `dynamics/advection.py`: now advects PERTURBATION pressure (not total) — self-flagged for reviewer
- `contracts/state.py`: c1 base-state additions
- `coupling/driver.py`: c1 wiring
- 2 NEW tests, 2 NEW ADRs

## What's frozen for c1-A2

`acoustic.py`, `tridiag.py`, `rk3.py`, `state.py` — c1 acoustic core is verified working in isolation.

## c1-A2 dispatched (window 0:11)

Specific 4-fix scope per bughunt2 §2 + c1-A1 self-diagnosis:
1. FIX #3 (FIRST): mass-conservation in advection (`test_mass_scalar_advection_is_conservative_for_constant_velocity` PASS)
2. FIX #1: per-layer dz in `_dz_from_state`
3. FIX #2: non-periodic interpolation in `_mass_to_u_face` / `_mass_to_v_face`
4. FIX #4: accept/reject perturbation-pressure advection (c1-A1 self-flagged)

Wall: 4-8h.

## Strategic state shift

c1 timeline VASTLY better than expected:
- Original estimate: 5-9 days
- A1 actual: 48 minutes for acoustic + tridiag
- A2 estimated: 4-8h
- **If c1-A2 PASSES, M6.x closes in <12h total c1 wall time**

Bug-hunt #2's structural analysis pre-identified exactly the bugs c1-A1 surfaced. The bug-hunt #1 → bug-hunt #2 sequence + plan critic all PAID OFF.

## Branch state

- `worker/codex/m6x-c1-klemp-skamarock` at `a9fbfb8` (c1-A1 implementation) + `1ca0693` (c1-A2 dispatch contract)
- Not merged to main yet — wait for c1-A2 completion + Opus reviewer accept before merging

## Decision logic

- If c1-A2 makes 1h probe PASS → continue to 6h + 24h proof → M6.x close
- If c1-A2 doesn't fix → manager evaluates c2 semi-implicit OR escalates to user

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 01:25
