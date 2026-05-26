# Sprint Contract — M6 Boundary/Dynamics Audit (Opus parallel)

## Objective

The microphysics-feedback worker noted unresolved risk: "finite but physically absurd pressure/wind excursions remain in dynamics/boundary fields. Theta and qc acceptance pass, but a follow-up boundary/dynamics audit is needed."

Run in parallel with the M6 acceptance sprint. Determine whether the absurd p/u/v/w excursions:
- (A) Originate in boundary forcing (AIFS interpolation, wrfbdy_d01 to wrfinput_d02)
- (B) Originate in dynamics core (rk3, acoustic loop, advection)
- (C) Originate in operational composition (operational_mode.py wrapper)
- (D) Are bounded enough to be M6-acceptable (depend on what "absurd" means — < 200 m/s wind is "absurd-looking" but bounds-passing)

If (D), bless M6 close; if (A-C), recommend a fix sprint.

## Non-Goals

- NO fix. Audit only.
- NO remote push.
- NO modification to source.

## File Ownership

Worktree at `/tmp/wrf_gpu2_boundaudit` on branch `tester/opus/m6-boundary-dynamics-audit`.
FIRST: `cd /tmp/wrf_gpu2_boundaudit`.

Write-only:
- `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/` — proofs + tester-report.md
- `scripts/m6_boundary_dynamics_audit.py` (NEW — read-only investigation driver)

Read-only:
- Everything else.

## Inputs

1. This contract.
2. `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/worker-report.md` — the flagged risk.
3. `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/fixed/proof_theta_explosion.json` — post-fix snapshot showing p/u/v/w values.
4. Gen2 wrfout truth for the 3 V3 ICs.
5. WRF Fortran reference for `advance_uv`, `advance_w`, `pressure_perturbation` in module_small_step_em.F.

## Acceptance Criteria

### Stage 1 — Catalog the excursions

Run the operational forecast on each of the 3 V3 ICs for 1h, capture per-step max/min of: p_perturbation, p_total, u, v, w (full domain). Compare to WRF wrfout per-step max/min (from Gen2 hourly archive — best you can get with hourly truth).

Write `proof_excursion_catalog.json` per IC.

### Stage 2 — Classify excursions

For each field (p, u, v, w) per IC: is the max excursion within physical reason (top-of-atmosphere ~100 Pa pressure perturbation, jet wind ~80 m/s, vertical motion ~3 m/s for stratiform / 10 m/s for convective)?

Pass=A if all fields physically reasonable. Pass=D if exceedances are bounded < 2× physical reason. Fail=B/C if exceedances are > 10× physical reason.

Write `proof_excursion_classification.json`.

### Stage 3 — Localize the source (if Stage 2 = B/C)

For the worst-excursion field+IC, use:
- `diagnostic_spatial_divergence_map.py` to see WHERE on the domain the excursion happens
- `diagnostic_boundary_ring_error_profiler.py` to check if it's at the boundary ring (→ A) or interior (→ B/C)
- Cross-check our operational p tendency at one offending cell against the WRF Fortran formula at the same cell

Write `proof_source_localization.json`.

### Stage 4 — Decision memo

Write `tester-report.md` with literal `Decision:` token, one of:
- `Decision: M6 close GO — excursions within physical-reason 2x band, acceptance-bounds pass`
- `Decision: M6 close NO-GO — excursions > 10x physical reason, root cause = <A/B/C>, recommend <sprint>`
- `Decision: M6 close CONDITIONAL — excursions tolerable for short forecasts but degrade beyond 1h, M6c (24h) gate may fail`

Include edge cases, gaps, recommendations.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_boundaudit
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

taskset -c 0-3 python scripts/m6_boundary_dynamics_audit.py --output .agent/sprints/2026-05-26-m6-boundary-dynamics-audit/

git add -A && git commit -m "[boundary audit] $(date -u +%FT%TZ)"
```

## Handoff

Per universal spec. The Decision: token drives whether M6 closes tonight or needs another sprint.
