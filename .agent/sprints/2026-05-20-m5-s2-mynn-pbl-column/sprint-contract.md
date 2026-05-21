# Sprint Contract — M5-S2 MYNN PBL Column Kernel

**Sprint ID**: `2026-05-20-m5-s2-mynn-pbl-column`
**Created**: 2026-05-20 ~21:55 by manager (Claude Opus 4.7 1M-context)
**Trigger**: M5-S1 closed; M5-S1.x closed (partial, deferred residual to M6); ADR-007 closed. Per ADR-005 deferred-schemes section, MYNN-EDMF PBL is the next M5 sprint. User directive: M5 finished by morning, full overnight autonomy, big steps in debugging.

## Objective

Implement WRF MYNN2.5 (Mellor-Yamada-Nakanishi-Niino level-2.5) PBL column kernel in JAX with the same governance pattern as M5-S1 Thompson: Fortran-harness oracle (structural anti-tautology), per-field tier-1 fixture parity, tier-2 conservation invariants, 1-launch fused JAX kernel, 0-byte HLO debug-vs-stripped diff. Closes M5 milestone scope alongside Thompson.

## Non-Goals

- Full MYNN-EDMF (mass-flux) — start with **MYNN2.5 only** (1.5-order closure with prognostic TKE). EDMF mass-flux is a follow-on (M5-S2.x or M6).
- Real surface-layer scheme (Monin-Obukhov, etc.) — use a **bulk-formula stub** that produces reasonable surface fluxes from `T`, `qv`, `U/V` at the lowest level. Real surface-layer is a future M6/M7 surface-coupling sprint.
- Multi-domain / nesting
- Full coupling to dycore — column-only kernel test.
- Sub-stepping for stability — single Δt per call; if instability surfaces, document and defer.

## File Ownership

Worker may CREATE:
- `src/gpuwrf/physics/mynn_pbl.py` (the JAX kernel)
- `src/gpuwrf/physics/mynn_constants.py` (constants — TKE coefficients, mixing-length parameters, etc.)
- `src/gpuwrf/physics/mynn_surface_stub.py` (bulk-formula surface-flux stub)
- `src/gpuwrf/physics/tridiagonal_solver.py` (vertical implicit tridiagonal solver — reusable component)
- `scripts/wrf_mynn_harness.f90` (Fortran oracle linking compiled WRF MYNN objects)
- `scripts/wrf_mynn_harness_build.sh` (nvfortran build)
- `scripts/m5_generate_mynn_fixture.py` (fixture generator using harness)
- `scripts/m5_run_mynn.py` (JAX runner)
- `scripts/m5_gate_mynn.py` (GO/GO_CARRYFORWARD/FALLBACK gate)
- `fixtures/manifests/analytic-mynn-pbl-column-v1.yaml`
- `fixtures/samples/analytic-mynn-pbl-column-v1.npz`
- `data/fixtures/analytic-mynn-pbl-column-v1/full.npz`
- `data/scratch/wrf_mynn_harness` (external binary, gitignored)
- `artifacts/m5/tier1_mynn_parity.json`
- `artifacts/m5/tier2_mynn_invariants.json`
- `artifacts/m5/mynn_profile.json`
- `artifacts/m5/mynn_gate_result.json`
- `artifacts/m5/hlo_dump/mynn_pbl_production.txt`
- `artifacts/m5/hlo_dump/mynn_pbl_debug_stripped.txt`
- `artifacts/m5/hlo_dump/mynn_pbl_debug_vs_stripped.diff`
- `src/gpuwrf/validation/tier1_mynn.py`
- `src/gpuwrf/validation/tier2_mynn.py`
- `tests/test_m5_mynn_*.py` (kernel shapes, constants, harness, tier1, tier2 — mirroring Thompson test layout)
- `.agent/decisions/ADR-008-mynn-jax-implementation.md` (notes parallel to ADR-006)
- Worker report.

Worker may MODIFY:
- `.agent/sprints/2026-05-20-m5-s2-mynn-pbl-column/sprint-contract.md` (only attempt amendments at bottom)
- `.agent/decisions/ADR-005-first-physics-suite.md` (only minor MYNN-completion cross-reference)

Worker may NOT modify:
- `src/gpuwrf/physics/thompson_*.py` (M5-S1 work, frozen)
- ADRs other than ADR-005 (minor) + new ADR-008
- Other sprint folders
- `feedback_*.md` memory files
- `MORNING-REPORT.md`

## Inputs

Required read order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/writing-gpu-kernels/SKILL.md`
4. `.agent/skills/validating-physics/SKILL.md`
5. `.agent/decisions/ADR-001-backend-selection.md`
6. `.agent/decisions/ADR-002-state-layout.md` (state pytree — MYNN consumes u/v/w/theta/qv + adds TKE)
7. `.agent/decisions/ADR-003-dycore-precision.md` (precision lock through M5-S2 still applies for new fields)
8. `.agent/decisions/ADR-005-first-physics-suite.md` (MYNN scope context)
9. `.agent/decisions/ADR-006-thompson-jax-implementation.md` (Thompson pattern; mirror for MYNN)
10. `.agent/decisions/ADR-007-precision-policy.md` (ADR-007 Authorization Matrix — note `state.w` is `needs-empirical-test` so MYNN's vertical advection of TKE stays FP64 until M6 test)
11. `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md` (operational RMSE binds)
12. `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/manager-closeout.md` (the Thompson playbook — Fortran-harness oracle, attempt-amendments, tester+reviewer pattern)
13. `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/sprint-contract.md` (the Thompson contract — mirror for MYNN)
14. `.agent/references/dispatching-gemini.md` (current Gemini policy — REACTIVE bug-chase + architecture-tiebreak only; budget conservation for M6/M7)
15. WRF source: `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_bl_mynn.F.pre` (the MYNN source-of-truth — search if exact path differs; the worker may need to extract from the compiled WRF tree)
16. The compiled WRF object tree: `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynn.o` (Fortran harness links against this)

## Acceptance Criteria

1. **Fortran-harness oracle** (`scripts/wrf_mynn_harness*`) compiles via NVHPC `nvfortran`, links against compiled WRF MYNN object, produces fixture inputs/outputs from real WRF MYNN code. Reproducible: same SHA from two consecutive runs.

2. **JAX kernel** (`src/gpuwrf/physics/mynn_pbl.py`) implements WRF MYNN2.5 column scheme:
   - Prognostic TKE: $\partial e/\partial t$ from shear, buoyancy, dissipation, vertical transport
   - Diagnostic mixing length $\ell$ (master length scale, Nakanishi 2001)
   - Eddy diffusivities $K_m, K_h$ from TKE and length
   - Vertical mixing tendencies for $u, v, \theta, q_v$ via implicit tridiagonal
   - Surface fluxes from bulk-formula stub (`mynn_surface_stub.py`)
   - Single fused `@jit(static_argnames=("dt", "debug"))` entry point
   - `state` pytree extended with `TKE` field (FP64 per ADR-003 lock for new fields through M5)

3. **Vertical implicit tridiagonal solver** (`tridiagonal_solver.py`) — reusable component, Thomas algorithm. Tested independently against scipy reference.

4. **Tier-1 fixture parity** under per-field tolerances. Per validation philosophy: Tier-1 catches transcription bugs; carry-forward tolerances acceptable if strict ADR-005 (`abs=1e-10, rel=1e-8`) cannot be met initially — document residual fields for M5-S2.x. STRICT tolerances are NOT a blocker for M5 close; operational-RMSE is the binding M6 gate.

5. **Tier-2 conservation + positivity + NaN/Inf**: TKE non-negative, momentum/energy/mass conservation residuals ≤ 1e-10 fractional, no NaN/Inf in any tendency.

6. **Profile metrics**: 1 kernel launch per step target (≤5 acceptable with documented rationale; HLO unroll concern from M5-S1.x applies if launches climb). Temp bytes per step = 0, H2D post-init = 0.

7. **HLO debug-vs-stripped diff = 0 bytes** (debug-gated branch dead-code-eliminated in production HLO per the M4+ debuggability hooks pattern).

8. **`gate_status = GO` or `GO_CARRYFORWARD`** based on Tier-1 strict-vs-carry-forward.

9. `python scripts/validate_agentos.py` passes.

10. `pytest -q` passes (count grows by new MYNN tests; no regression).

## Validation Commands

```bash
bash scripts/wrf_mynn_harness_build.sh       # one-time
python scripts/m5_generate_mynn_fixture.py
python scripts/m5_run_mynn.py
python scripts/m5_gate_mynn.py
python scripts/validate_fixture_manifest.py fixtures/manifests/analytic-mynn-pbl-column-v1.yaml
python scripts/validate_agentos.py
pytest -q
```

## Performance Metrics

- `artifacts/m5/mynn_profile.json` — must show: `kernel_launches_per_step ≤ 5` (target 1), `temporary_bytes_per_step = 0`, `host_to_device_bytes_post_init = 0`, recorded wall time.
- `artifacts/m5/hlo_dump/mynn_pbl_debug_vs_stripped.diff` — must be 0 bytes.
- `artifacts/m5/hlo_dump/mynn_pbl_production.txt` — HLO size ≤ 300 KB (allow more headroom than Thompson because tridiagonal solver introduces vertical recurrence).

## Proof Object

- Worker report ≥3000 bytes including per-AC verdict (file:line citations), per-field parity numbers, profile/HLO/launches confirmation, Fortran-harness build log summary, ADR-008 path.
- ADR-008-mynn-jax-implementation.md ≥1500 bytes documenting key implementation choices (length-scale formula chosen, tridiagonal solver coefficients, surface stub formula, TKE bounds).

## Risks

- **Tridiagonal solver vs JAX fusion**: implicit vertical solve introduces a sequential vertical recurrence. JAX `jax.lax.scan` is the natural fit. Risk: scan unrolling could blow HLO size. Mitigation: use `jax.lax.scan` with `unroll=False` and verify HLO size.
- **TKE positivity**: TKE can numerically go negative under certain shear/buoyancy combinations. Mitigation: clamp at ε (small positive) like WRF does.
- **Surface flux stub vs WRF surface-layer**: bulk formula won't match WRF MYNN's surface-layer outputs exactly. Tier-1 parity may suffer; document residual fields as surface-stub-attributable per R-3 caveat from M5-S1 reviewer.
- **Length-scale formula**: WRF MYNN uses Nakanishi 2001 master length. Multiple length-scale variants exist (surface-layer, mixing-layer, BL-depth). Worker must transcribe WRF's exact formula, NOT a textbook variant.
- **MYNN2.5 vs MYNN-EDMF**: `bl_pbl_physics=5` is MYNN2.5; `=6` is MYNN-EDMF. M5-S2 implements `=5`. Fortran harness must invoke with `bl_pbl_physics=5` flag (or equivalent WRF subroutine select).

## Handoff Requirements

- Worker report + ADR-008.
- Tester (Claude Opus 4.7 xhigh) verifies per-AC, runs adversarial probe (pick one suspect WRF subroutine path, verify), recommends Accept/Reject + path.
- Reviewer (Claude Opus 4.7 xhigh) issues binding verdict.
- Manager closeout + merge.

## Dispatch Pattern (per current Gemini policy 2026-05-20 ~21:50)

- Primary worker: codex gpt-5.5 xhigh (frontrunner). Solo unless bug-chase triggers Gemini reactive-pair.
- Tester: Claude Opus 4.7 xhigh. Solo; authorized inline fixes for minor issues (user directive: bigger steps in debugging).
- Reviewer: Claude Opus 4.7 xhigh. Solo binding voice.
- Gemini: ONLY reactive — if worker/tester/reviewer hit a complex bug that codex+Claude cannot find, manager dispatches Gemini bug-chase side-runner. Otherwise no Gemini quota burn on this sprint.

## Bigger-steps authorization (user directive 2026-05-20 evening)

- Worker: fix minor inline issues you discover; file blocker only for substantive scope changes or true regressions.
- Tester: authorized to apply minor fixes inline and re-verify; you don't need to bounce back to the worker for a 1-line cleanup.
- Reviewer: if you find a R-class finding that's a 1-line fix and you're confident, apply it inline + note in your report. Otherwise file the R-finding for manager triage.
- Manager: skip tester/reviewer cycles when worker delivers cleanly and the work is mechanical (e.g. a small follow-up after a verified R-fix spec).

## Expected wall-time

Worker phase: 4-10 hours (Fortran harness + JAX kernel + tridiagonal solver + tests; bigger than Thompson because of the new tridiagonal solver dependency + surface stub).
Tester phase: 30-90 min.
Reviewer phase: 30-90 min.
Manager merge + closeout: 15-30 min.
Total: 5-13 hours wall-clock. Targets morning user availability for any blocker triage.

When done, commit + push to `worker/codex/m5-s2-mynn-pbl-column` + `/exit`.

## Attempt-2 AC6 Amendment — 2026-05-21

Reviewer finding R-4 required either raw HLO launch markers ≤5 or an explicit contract amendment. Attempt-2 implements the real dry MYNN2.5 path rather than the attempt-1 compact Louis/Blackadar proxy: WRF level-2 stability, option-2 MYNN length scale, five implicit vertical solves (`qke`, `u`, `v`, `theta`, `qv`), and WRF-style surface boundary terms. The raw HLO marker count is therefore permitted up to **35** for M5-S2 attempt-2, provided all of the following hold:

- `kernel_launches_per_step` and `raw_hlo_launch_marker_count` report the same unclamped raw marker count.
- HLO production text stays ≤300 KB.
- `temporary_bytes_per_step = 0` and `host_to_device_bytes_post_init = 0`.
- The worker report cites the profile artifact and does not claim the original ≤5 target was met.

This amendment is limited to M5-S2 attempt-2. A follow-up M5-S2.x/M6 optimization may reduce the five implicit solves or replace the XLA primitive if profiler evidence shows the raw marker count maps to real launch-bound cost.
