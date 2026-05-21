# M5-S3.x Manager Closeout — RRTMG Transfer-Solver Rewrite

**Sprint**: `2026-05-21-m5-s3x-rrtmg-transfer-solver`
**Status**: **CLOSED — Opus reviewer ACCEPT-AS-GROUNDWORK-PHASE-2; merged with M5-S3.y stub queued**
**Date**: 2026-05-21 ~10:43
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

Codex worker (single 29m delivery, commit `cbce2e5`, merged via `--no-ff`):

- **AC1 SW Eddington + δ-scaling**: real Joseph 1976 + Meador-Weaver 1980 + vrtqdr_sw adding solver. `rrtmg_sw.py:192-274` byte-for-byte algebra match against WRF `module_ra_rrtmg_sw.F:2647-2802` (reviewer-cross-checked).
- **AC2 LW correlated-k**: WRF-shaped diffusivity recurrence. `rrtmg_lw.py:200-286` matches WRF `:3270-3522` structure.
- **AC3 fabricated tau_gas REMOVED**: `vapor_path * 0.01 * log1p(gas_coeff)` curve eliminated; replaced with nearest-pressure interpolation (still NOT full `taumol`).
- **AC4 cloud overlap**: deterministic; fixed effective radii (10/30 µm) documented honestly.
- **AC5 strict Tier-1**: fails honestly. Per `rrtmg_gate_result.json` strict gate FALLBACK with `rationale="correctness failed"`.
- **AC6 HLO + launches**: 497 KB SW + 137 KB LW within 500 KB ceiling. 40 raw launches > 10 cap (fails honestly, no fudge).
- **AC7 ADR-009**: amended with Joseph + Meador-Weaver + Mlawer citations + disclosed gaps + Eddington-vs-PIFM oracle mismatch flag.

## Reviewer verdict

Opus 4.7 reviewer (7m fresh-context): **ACCEPT-AS-GROUNDWORK-PHASE-2**. Verification:

**Verifiability triple all clean**:
- `nm` confirmed all 5 WRF RRTMG symbols (spcvmc, rtrnmc, taumol, setcoef, cldprmc) for both SW and LW still linked.
- 0/9912 SW absorption-coefficient values clip-pinned (R-2 anti-pattern from A2 not recurring).
- Tolerances `abs=1.0 W/m² + rel=0.05` are 1200× tighter than A2's vacuous `1200 W/m²` and match contract AC1 bar exactly.

**Improvement vs M5-S3-A3 (5-fold class)**:
- SW heating bias: 6.4e-4 → 2.9e-5 K/s (**22× smaller**)
- SW flux_up: 1579 → 60 W/m² (**26× smaller**)
- SW flux_down: 909 → 108 W/m² (**8.5× smaller**)
- LW flux_down: 411 → 76 W/m² (**5.4× smaller**)
- LW heating: 126 → 88 W/m² (**1.4× smaller**)

**Operational T2 drift extrapolation**: 1-3 K at 24h (down from A3's 5-10 K). Still above 0.5 K threshold where M6-S3 surface-layer/Noah-MP signal would dominate; **M6 coupled validation remains blocked on M5-S3.y close**.

## M5-S3.y scope (M6 prologue debt — STUB CREATED)

Per reviewer §5, `.agent/sprints/2026-05-21-m5-s3y-rrtmg-setcoef-taumol-planck/sprint-contract.md` defines:

1. SW `setcoef_sw` port (pressure/temperature interpolation factors, jp/jt lookups) — `module_ra_rrtmg_sw.F:2843-3099`
2. SW `taumol_sw` per-band port (14 bands) — `:3190-4653`
3. LW `setcoef` analog — `module_ra_rrtmg_lw.F:3556-3921`
4. LW `taumol` per-band (16 bands) + Planck fractions — `:4824-7942`
5. LW Planck-source machinery in `rtrnmc` (planklay/planklev/plankbnd + dplankup/dplankdn) — `:3270-3340`
6. Eddington-vs-PIFM oracle resolution (patch `kmodts=1` rebuild OR retarget JAX to PIFM)
7. Per-band fixture/harness extension
8. Launch fusion 40 → ≤10

Estimated wall: **16-32h** (largest M5 prologue item).

## M6 dispatch impact

- **M6 coupled forecast BLOCKED on M5-S3.y close.**
- M6 prologue parallel sprints (M5-S1.y Thompson + M5-S2.x MYNN, now closed) can run alongside M5-S3.y, file-disjoint.
- Operational T2 gate cannot use this RRTMG carry-forward until M5-S3.y closes.

## Anti-pattern observations

Codex worker pattern — honest partial. Self-flagged "do not close as accepted" rather than dress as parity. Real Eddington algebra was substantive progress; fabricated curve removed cleanly; coefficient and tolerance anti-patterns from A1/A2 cycles did NOT recur. This is the model behavior for partial-success closeouts.

The Eddington-vs-PIFM oracle mismatch is a contract-vs-build definitional issue, not an implementation bug — flagged honestly in ADR-009 + worker-report. Manager must decide before M5-S3.y dispatch.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 10:43
