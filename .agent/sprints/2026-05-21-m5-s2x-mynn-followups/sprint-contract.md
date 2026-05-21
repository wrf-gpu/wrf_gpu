# Sprint Contract — M5-S2.x MYNN Follow-Ups (M6 Prologue)

**Sprint ID**: `2026-05-21-m5-s2x-mynn-followups`
**Created**: 2026-05-21 by manager (Claude Opus 4.7 1M-context)
**Status**: ACTIVE — second M6 prologue sprint dispatched alongside M5-S1.y + M5-S3.x
**Trigger**: M5-S2-A2 Opus reviewer ACCEPT-WITH-MINOR-FOLLOWUPS deferred 4 items (`.agent/sprints/2026-05-20-m5-s2-mynn-pbl-column/reviewer-a2-report.md` §8). EDMF (item 4) is intentionally M6-not-prologue per contract Non-Goals; items 1-3 + a new mean-field oracle item land here.

## Objective

Close three deferred items from M5-S2-A2 reviewer that bear on M6 operational validation:

1. **Independent mean-field budget probe.** Today Tier-2 momentum/heat/moisture residuals compare the implicit solver against the SAME `drag, K_top, theta_flux, qv_flux` it consumed — a solver self-consistency check, not an independent physics validation. Add a WRF-harness-flux-vs-JAX-flux comparison at the same state so the mean-field residual is load-bearing.
2. **Flux-Richardson radicand guard resolution.** `mynn_pbl.py:202-205` guards `radicand = max(ri²-ri3·ri+ri4, 0)` while WRF does plain `SQRT(...)`. WRF NaNs on negative radicand; we silently clamp. Either remove the guard (bit-match WRF) OR document as intentional defensive deviation in ADR-008 + add a test demonstrating the boundary condition.
3. **Surface stub realism boundary.** The current `bulk_surface_fluxes` neutral stub is diagnostic-only. Document the exact gap between this stub and what M6-S3 will need from a real Monin-Obukhov surface layer (interface contract: what M6-S3 hands MYNN, what MYNN expects), so the M6-S3 dispatch can plug in cleanly without rewriting MYNN. This is a **memo + interface stub**, NOT a real surface-layer implementation (M6-S3 owns that).

Explicit Non-Goal: **NO EDMF mass-flux extension here** — that is M6-S3 or M6 fold-on per ADR-008 and the M5-S2-A2 reviewer §8 item 4.

## Acceptance (pre-M6-coupled-implementation gate)

- **AC1 — Independent budget probe.** Tier-2 mean-field check uses WRF harness fluxes as the ground truth, not JAX-internal. New artifact `artifacts/m5/tier2_mynn_independent_budget.json` showing per-field residuals against harness oracle.
- **AC2 — Radicand decision.** ADR-008 amended with the chosen path (remove or document). Test added that exercises the WRF negative-radicand boundary or asserts the deliberate clamp.
- **AC3 — Surface-layer interface memo.** `.agent/decisions/ADR-008-mynn-jax-implementation.md` extended with a **Section "Surface-layer coupling interface (for M6-S3)"** that names: inputs MYNN expects (`ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv`), outputs it produces (`qke_surf, mean-field tendencies`), units, sign conventions. Plus a stub Python type contract in `src/gpuwrf/physics/mynn_surface_stub.py` documenting the same.
- **AC4 — Honest accounting.** No `min(raw, cap)` fudge. Transfer audit clean. Existing Tier-1 parity (already strong at `u≤7.7e-4, theta≤6.3e-5, tke≤1.5e-6, el≤3.1e-3`) unbroken.
- **AC5 — Tests pass.** 410+ pytest pass count maintained.

## Inputs (carried forward from M5-S2)

- `src/gpuwrf/physics/mynn_pbl.py`, `mynn_constants.py`, `mynn_surface_stub.py` — preserve.
- `scripts/wrf_mynn_harness.f90`, `wrf_mynn_harness_build.sh` — already links real WRF-EDMF; extend as needed for per-step flux dump.
- `scripts/m5_run_mynn.py`, `m5_gate_mynn.py` — preserve gate logic.
- `tests/test_m5_mynn_*` — extend.

## Files Worker May Modify

- `src/gpuwrf/physics/mynn_pbl.py` (radicand decision if "remove" chosen; minor numerical hygiene only)
- `src/gpuwrf/physics/mynn_surface_stub.py` (extend interface contract)
- `src/gpuwrf/validation/tier2_mynn.py` (independent budget probe)
- `scripts/wrf_mynn_harness.f90` (extend for per-step flux dump if needed for AC1)
- `scripts/m5_run_mynn.py`, `scripts/m5_gate_mynn.py` (extended reporting)
- `tests/test_m5_mynn_*` (new tests for AC1 + AC2)
- `.agent/decisions/ADR-008-mynn-jax-implementation.md` (amend; add Surface-layer-coupling-interface section)
- Worker report

## Files Worker Must NOT Modify

- Anything under `src/gpuwrf/physics/thompson_*` (P1 owns)
- Anything under `src/gpuwrf/physics/rrtmg_*` (P3 owns)
- Anything under `src/gpuwrf/dynamics/**`, `src/gpuwrf/contracts/**`, `src/gpuwrf/timestep/**`, `src/gpuwrf/coupling/**` (M6-S1 owns)
- Anything that touches Thompson tridiagonal solver (shared infrastructure — preserve as-is)
- Real Monin-Obukhov implementation (M6-S3 owns)
- EDMF mass-flux extension (M6 fold-on)
- Any other ADR or governance file

## Dispatch

- Primary worker: codex gpt-5.5 xhigh
- Reviewer (mandatory): Claude Opus 4.7 xhigh
- Wall-time: 2-6 hours
- Worktree: `/tmp/wrf_gpu2_s2x` (isolated from P1 + P3)
- Branch: `worker/codex/m5-s2x-mynn-followups`

## Hard rules

- The "independent budget probe" must compare against the WRF-linked harness output, not the same JAX path on both sides.
- ADR-008 amendment must cite WRF source `file:lineno` for any new claim.
- Surface-layer memo is a contract handoff — it specifies what M6-S3 will deliver, not implementation.
- Do not regress existing Tier-1 parity numbers; if you touch `mynn_pbl.py` body, regenerate the Tier-1 evidence.
