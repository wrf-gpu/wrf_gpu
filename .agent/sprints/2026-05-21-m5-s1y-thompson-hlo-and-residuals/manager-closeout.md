# M5-S1.y Manager Closeout — Thompson HLO + Process Residuals

**Sprint**: `2026-05-21-m5-s1y-thompson-hlo-and-residuals`
**Status**: **CLOSED — Opus reviewer ACCEPT-AS-GRAY-ZONE-CHECKPOINT; M6-S1 UNBLOCKED-WITH-DEBT**
**Date**: 2026-05-21 ~11:05
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

Codex worker (single 43m delivery; merged via `--no-ff`):

- **AC1 HLO-safe gather**: packed default-IN `tpg_qrfz` gather at `thompson_tables.py:104-113`, consumed at `thompson_column.py:263-269, 556-584`. Launches = 10 raw = 10 reported (exactly +5 over baseline of 5; matched manager target). Full HLO 421 KB > 350 KB target → GRAY-ZONE per gate.
- **AC2 process residuals**: `qr` met (3.3e-8 ≤ 1e-7), `Ni` 126975→772 (164×↓), `qc` 53×↓, `qi` 26×↓. Remaining `qg/qv/T/Ni/Nr` miss strict targets due to source-verified absent class of WRF collision tables (`module_mp_thompson.F.pre:2547-2609`). CGG11 verification preserved (computed via `math.gamma`, not literal copy).
- **AC3 non-tautological Tier-2**: aggregate water-budget cross-check against WRF Fortran harness fixture (3.975e-10 residual). Per-process out of harness reach — honest scope limit.
- **AC4 honest accounting**: `nm` confirms `module_mp_thompson_*` symbols still linked. No `min(raw, cap)` fudge anywhere. 0 post-init transfers. 0-byte debug-vs-stripped diff.
- **AC5 ADR-006 amended**: §74-86 documents HLO gather pattern + process map + strict-parity debt with WRF file:line citations.

## Reviewer verdict

Opus 4.7 reviewer (14m fresh-context): **ACCEPT-AS-GRAY-ZONE-CHECKPOINT**.

Operational-impact extrapolation per validation-philosophy memory:
- T residual 8.3e-3 K/step → sqrt(1440 steps) random-walk floor = **0.3 K at 24h** vs T2 obs noise ~0.5-1.5 K → below operational floor
- qg/qv = 10 ppm column mass → 1e-3 K/step thermal, redundant with T term already counted
- Ni/Nr near-zero references operationally inert until M7 radar/cloud-effective-radius diagnostics

Verifiability triple all clean (no `min/cap`, no fabricated tables, GRAY-ZONE honestly triggered).

## M5-S1.z optional follow-up (NOT a prerequisite for M6-S1)

Triggered ONLY if M6 RMSE on U10/V10/T2 flags microphysics-driven drift:

1. Collision-table export (`tmr_racs1/2, tcr_sacr1/2, tnr_racs1/2, tnr_sacr1/2, tmr_racg, tcr_gacr, tcg_racg, tnr_racg, tnr_gacr, tcs_racs1, tms_sacr1`) from WRF `:2547-2609`.
2. HLO-safe collision-table gather (target ≤15 total launches; widen HLO ceiling to 500 KB).
3. Wire `prr_rcs / prs_rcs / prg_rcs / prr_rcg / prg_rcg` paths.
4. HLO-size reduction pass for fusion opportunities.

If M6 RMSE passes with current M5-S1.y physics, M5-S1.z is droppable per validation philosophy.

## M6 dispatch impact

**M6-S1 (coupled interface freeze) UNBLOCKED-WITH-DEBT.** Thompson inherits to coupled driver with: 10 launches/step, 0 transfers, 421 KB HLO, Fortran-harness oracle binding. R-3 caveat: any future collision-table wiring in M5-S1.z must plan for `≤15 total launches` not `≤5 additional from M5-S1.y`.

**M6 coupled-forecast validation still BLOCKED on M5-S3.y close** (RRTMG setcoef+taumol+Planck-source remains M5-S3.x → M5-S3.y debt).

## Memory-patch proposals (deferred)

1. Feedback memory: when sprint AC specifies numerical ceiling, plan one sprint of headroom so the next sprint doesn't immediately burst it.
2. Positive memory: GRAY-ZONE escape valve in gate scripts worked as designed — worker did not relabel real miss as GO.
3. Reference memory pointer: `module_mp_thompson.F.pre:2547-2609` as canonical "absent collision-table class" citation.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 11:05
