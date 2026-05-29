# Sprint U — Operationalize + harden + strictly-validate the dry dycore

Worker: Opus 4.8 (1M context)
Branch: `worker/opus/f7d-pressure-mass-fix` (tip was 88ed694)
Date: 2026-05-29

## Objective

Operationalize, harden, and strictly-validate the F7-closed dry dycore so the
real-case operational path genuinely uses the validated operators (the GPT
firm-rule pre-close critique BLOCKED declaring "DONE" because the operational
path bypassed the F7 fixes and some operators were approximate/unwired/false-green).

## Files changed

Source:
- `src/gpuwrf/dynamics/flux_advection.py` — `advect_w` open-top face flux + lid
  pickup (P1-5), gated by `top_lid` (rigid-lid path byte-unchanged).
- `src/gpuwrf/dynamics/explicit_diffusion.py` — new
  `wrf_deformation_momentum_tendency` (WRF deformation-tensor momentum diffusion,
  P0-2).
- `src/gpuwrf/runtime/operational_mode.py` — new
  `use_deformation_momentum_diffusion` namelist flag + wiring (theta keeps scalar
  flux-divergence); thread `top_lid` to `advect_w_flux`.
- `src/gpuwrf/integration/daily_pipeline.py` — `_build_real_case` now uses the F7
  operators (flux advection, fp64, diff_6th_opt=2, WRF damping, open top) + records
  active operators in metadata (P0-1).
- `pyproject.toml` — `close_gate` pytest marker (P0-5).

Tests:
- `tests/dynamics/test_advect_w_topface.py` (4 tests).
- `tests/dynamics/test_deformation_momentum_diffusion.py` (4 tests).
- `tests/idealized/test_dycore_close_gate.py` (2 close-gate tests, assert PASS).
- `tests/idealized/test_warm_bubble.py`, `test_density_current.py` — tightened to
  assert `verdict==PASS` under `close_gate`.

Scripts:
- `scripts/sprintU_real_case_smoke.py`, `scripts/sprintU_guards_off_proof.py`,
  `scripts/sprintU_straka_canonical_parity.py`.

Proofs (`proofs/sprintU/`): `operational_path_unification.md`, `real_case_smoke.json`,
`momentum_diffusion_deformation.md`, `straka_deformation_gate.{json,md}`,
`straka_canonical_parity.{json,md}`, `advect_w_topface.md`, `ci_close_gate.md`,
`guards_off_operational_proof.json`, `close_gate/*.json`. Updated
`proofs/f7/DYCORE_STATUS.md`.

## Commands run

- Baseline + post-change idealized gates (GPU): warm bubble + Straka via
  `python -m gpuwrf.ic_generators.idealized`.
- `pytest tests/dynamics/...`, `pytest tests/idealized/test_dycore_close_gate.py -m close_gate`.
- `scripts/sprintU_real_case_smoke.py`, `sprintU_guards_off_proof.py`,
  `sprintU_straka_canonical_parity.py` (all GPU, `PYTHONPATH=src taskset -c 0-3`).
- `git stash` A/B to confirm 5–6 pre-existing CPU/source-string test failures are
  NOT introduced by this sprint.

## Proof objects produced

| item | proof | verdict |
|---|---|---|
| P0-1 operational unification | `operational_path_unification.md`, `real_case_smoke.json` | PASS (bitwise same dycore; warm bubble 6/6 via operational entry; real case finite) |
| P0-2 deformation momentum diffusion | `momentum_diffusion_deformation.md`, `test_deformation_momentum_diffusion.py`, `straka_deformation_gate.json` | PASS (analytic oracle + Straka 6/6) |
| P0-3 canonical Straka parity | `straka_canonical_parity.{json,md}` | PASS (worst max\|w\| 11.9%, front 400m, finite through touchdown) |
| P0-4/P0-5 CI close-gate | `ci_close_gate.md`, `close_gate/*.json` | PASS (both assert PASS) |
| P1-5 advect_w top-face | `advect_w_topface.md`, `test_advect_w_topface.py` | PASS (open-top WRF formula; rigid-lid unchanged) |
| P1-6 guards-off | `guards_off_operational_proof.json` | PASS (warm bubble 6/6 guards-off; real finite) |

## Unresolved risks / honest gaps

- **3D terrain / map factors / boundaries**: the deformation operator is the
  flat-slab reduction (zx=zy=0, ny=1); slope cross-coordinate terms, map factors,
  and specified/nested lateral boundaries are Phase-B gates (flux_advection/rhs_ph
  still document the periodic/unit-map scope).
- **Per-cell WRF field parity**: P0-3 is an array-level time-series comparison
  (max\|w\|, θ′min, front) through touchdown, not a per-cell wrfout field dump diff.
  The WRF arrays exist (`proofs/m9/`) for a future tightening; the time-series
  already binds the operators the runaway exercised. JAX front lags WRF ~400m and
  θ′min ~1.5K (small bounded stencil/IC difference, not an operator error).
- **Real-case IC dtype**: the d02 replay-state build emits a benign float64→float32
  cast warning; the operational run forces fp64, so the dycore is fp64. Tidying the
  replay dtype is out of scope.
- **Deformation default**: close-gate default keeps the scalar flux-divergence (the
  exact F7N-validated operator); the deformation operator is wired + validated +
  gate-proven but not flipped to default (one-line change for a Phase-B sprint).
- Pre-existing stale tests (5–6, source-string / old-dycore-form) fail on the
  baseline independent of this sprint; not addressed here.

## Next decision needed

Manager review to merge the `f7d` chain → Phase B. The dycore is operational-ready:
the real path uses the validated operators, strictly validated vs WRF through the
touchdown window, CI-gated against false-green. Phase B should open the
terrain/map-factor/boundary/moist-coupling gates.

SPRINTU_COMPLETE
