# Sprint Contract — c1-A8 Vertical Momentum Bisect + Surgical Fix

**Sprint ID**: `2026-05-22-m6x-c1-a8-vertical-momentum-bisect`
**Created**: 2026-05-22 ~04:50
**Status**: ACTIVE
**Trigger**: c1-A7 horizontal momentum flux-form fix LANDED (horizontal-only stable 360 steps), but VERTICAL momentum still fails at step 106 in diagnostic config; 1h coupled probe sanitize 86.1%.
**Branch**: `worker/codex/m6x-c1-a8-vertical-momentum-bisect` (from c1-A7 HEAD `abd12c4`)
**Worktree**: `/tmp/wrf_gpu2_c1_a8`

## Objective

Bisect within vertical momentum advection to isolate the specific component (u/v/w vertical, sign, metric, BC) causing step-106 failure. Then SURGICAL fix per WRF canonical. Then 1h + 24h + speedup probes if green.

## Acceptance

- **AC1**: Bisection probes (≥3) isolate vertical momentum bug to specific field + component
- **AC2**: Surgical fix applied per WRF canonical; cite WRF source
- **AC3**: vertical-momentum-only probe now stable past step 200 (preferably 360)
- **AC4**: 1h coupled probe sanitize firing <5%
- **AC5**: 24h coupled probe finite, theta away from clip bounds, mu in physical range
- **AC6**: Speedup re-measurement ≥4× (likely already proven 43.90× per c1-A7)
- **AC7**: ADR-007 → PASS-with-evidence
- **AC8**: ADR-019 amended with c1-A8 findings

## File ownership

- **MODIFY**: `src/gpuwrf/dynamics/advection.py` (vertical components of `advect_u/v/w_face`)
- **MODIFY**: tests + scripts/m6_full_domain_batching.py (bisection flags if needed)
- **DON'T TOUCH**: acoustic.py (works in isolation), tridiag.py, rk3.py, state.py, physics/, io/, boundary_apply.py

## Hard rules

- Bisect FIRST, fix SECOND (per bughunt #4 anti-pattern warning)
- ONE fix per probe
- Cite WRF source for sign convention and metric
- c1-A7 horizontal momentum flux form is FROZEN — don't touch
- BEFORE `/exit`: `git add . && git commit && git push`
- `/exit` slash-command

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (after green)
- Wall: 3-6h
