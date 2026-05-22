# c1-A5-H3 Manager Closeout — Inconclusive (OOM) but Disproven by Bisection

**Sprint**: c1-A5-H3 Discriminator (n_acoustic 1-line patch)
**Status**: **CLOSED — patch landed, probe failed CUDA OOM, but bisection already proved H3 isn't the bug**
**Date**: 2026-05-22 ~04:15
**Worker**: codex gpt-5.5 xhigh (~25 min)

## What happened

1. Worker applied bughunt #4 H3 patch at `dynamics/rk3.py:49` (1-line, removing n_acoustic auto-promotion)
2. Ran 1h probe → CUDA OOM during transfer audit, didn't write final verdict
3. **Worker discovered**: the patch was effectively dead code — `n_acoustic` is computed at the ENTRYPOINT via `required_n_acoustic_for_state` BEFORE `rk3_step` is called

## Why this is moot

The empirical bisection (closed earlier) DEFINITIVELY proved:
- Acoustic-only probe: stable for 360 steps
- The bug is in ADVECTION, not acoustic
- n_acoustic value is irrelevant (acoustic isn't broken regardless)

Bug-hunt #4 H3 was wrong on TWO counts:
1. The call site identified for patching was wrong (entrypoint, not rk3.py:49)
2. Even if patched correctly, acoustic isn't where the bug lives

## Honest accounting

c1-A5-H3 was dispatched BEFORE bisection report landed. Both ran in parallel. By the time bisection finding came in, c1-A5-H3 was nearly done. Wasted ~25 min of compute, but the slot is now free for productive work.

## Slot recovery

1 codex slot freed. c1-A6 (advection bisection) continuing in window 0:18 with full attention.

Per pattern: c1-A4 (buoyancy) also expected to fail soon — it's pursuing same disproven path (acoustic is stable per bisection). Slot will free up.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 04:15
