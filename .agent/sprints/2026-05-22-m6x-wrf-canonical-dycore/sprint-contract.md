# Sprint Contract — M6.x WRF-canonical Dycore Completion

**Sprint ID**: `2026-05-22-m6x-wrf-canonical-dycore`
**Created**: 2026-05-22 ~00:30
**Status**: ACTIVE — dispatching now per M6-S5 Opus binding §3
**Trigger**: M6-S5 ADR-007 verdict FAIL. Throughput 9.70× ≥ 4× target, but M4 reduced dycore not stable-grade at WRF-canonical 3km dt=10-12s. Per M6-S5 Opus §3: complete the M4-deferred dycore physics; preserve M4 architectural baseline (zero post-init transfers, JAX-XLA-resident state); NOT a full WRF Fortran port.

## Objective

Replace M4 reduced dycore proxy with WRF-canonical split-explicit dycore for the acoustic substep and `mu`-continuity, sufficient to run a 24h coupled forecast on real d02 (160×67×45) at `dt=10-12s` with Tier-2 invariants PASS and sanitize firing rate <5% of steps.

## Pre-dispatch decision

Manager will dispatch **Gemini architecture tiebreak in parallel** per M6-S5 Opus §4. Gemini opinion will land within ~10min; if it argues against option (a)-narrowed (e.g., says "this is 3-6mo of work; do option c re-architecture"), manager may revise scope mid-flight. Worker proceeds with current scope while Gemini opines.

## Acceptance

- **AC1 Physical sound speed in acoustic.py**: replace `c²=1.0` and `pressure_coupling=1.0e-3` proxies with physical values. Use WRF `dyn_em/module_small_step_em.F` formulation. Cite `:lineno`.
- **AC2 Per-cell CFL diagnostic**: per-grid-cell acoustic CFL diagnostic per substep. Bind `n_acoustic` from `time_step_sound` namelist semantics (typical 4-6 for 3km).
- **AC3 Canonical `mu`-continuity update**: implement WRF `dyn_em/module_em.F` mass tendency. State.mu evolves per dry-mass tendency from divergence. Validate against WRF fixture at `dt=10-12s`.
- **AC4 Tier-2 invariants under lifted cap PASS**: re-run M6-S5 verdict harness. `tier2_lifted_cap_invariants.json` must show:
  - NaN/Inf count = 0
  - sanitize firing rate < 5% of total steps
  - final state away from clip bounds (theta in [200, 350]K not [150, 550]; qv well below 0.05; w well below 50 m/s)
- **AC5 24h forecast finite + physically valid**: full 24h on d02, dt=10s, dycore actually integrating. No silent saturation.
- **AC6 Speedup preserved**: re-run `m6_full_domain_batching.py`; speedup must remain ≥4× (target ≥6× since infra still in place). End-to-end wall is binding numerator.
- **AC7 Post-init transfer regression resolved**: M6-S5 F-3 noted 164 KB H2D on warmed audit. Find source (likely boundary-replay step gather or sanitize scalar sync). Either fix to 0 or document explicit exception to M4's hard-zero rule with rationale.
- **AC8 No physics-kernel changes**: only `dynamics/{acoustic, rk3, step, tendencies}.py` + `contracts/state.py` (mu, mut, msft if needed) + driver wiring. Physics (Thompson/MYNN/RRTMG/sfclay/Noah-MP) FROZEN.
- **AC9 ADR-007 status update**: amend from FAIL to PASS-with-evidence after AC4+AC5+AC6 all green.
- **AC10 New ADR**: NEW `ADR-015-m6x-wrf-canonical-dycore.md` documenting acoustic substep formulation, mu-continuity choice, CFL diagnostic, sound speed source, n_acoustic binding rule. Cite WRF source lines.

## Files Worker May Modify

- `src/gpuwrf/dynamics/acoustic.py` (REWRITE: physical sound speed + CFL)
- `src/gpuwrf/dynamics/step.py`, `rk3.py`, `tendencies.py` (mu-continuity wiring; minimal changes)
- `src/gpuwrf/contracts/state.py` (add mu_tendency / dry-mass diagnostic state if needed; preserve SoA)
- `src/gpuwrf/coupling/driver.py` (no dycore cap anymore; tighter integration)
- `scripts/m6_full_domain_batching.py` (rerun verdict harness)
- `tests/test_m4_dycore_step.py`, `tests/test_m6_dycore_cap_lift.py`, `tests/test_m6_4x_verdict.py`, NEW `tests/test_m6x_dycore_completion.py`, `tests/test_m6x_cfl_diagnostic.py`, `tests/test_m6x_mu_continuity.py`
- `.agent/decisions/ADR-007-precision-policy.md` (Status amendment)
- `.agent/decisions/ADR-015-m6x-wrf-canonical-dycore.md` (NEW)
- `artifacts/m6/performance/full_domain_batching_verdict.json` (regenerated)
- `artifacts/m6/performance/tier2_lifted_cap_invariants.json` (regenerated)
- Worker report

## Files Worker Must NOT Modify

- `src/gpuwrf/physics/**` (all FROZEN — Thompson/MYNN/RRTMG/sfclay/Noah-MP)
- `src/gpuwrf/coupling/{physics_couplers,boundary_apply}.py` body (only consume new dycore outputs)
- `src/gpuwrf/io/**` (frozen)
- `src/gpuwrf/validation/**` body (only re-run; do not change kernels)
- Other ADRs (modulo cross-ref updates)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY per sprint-lifecycle; this is the biggest sprint of the project)
- Wall-time: **16-32h** (real dycore work; may extend)
- Worktree: `/tmp/wrf_gpu2_m6x` (NEW)
- Branch: `worker/codex/m6x-wrf-canonical-dycore`

## HARD RULES

1. **NO physics-kernel changes** — physics is frozen
2. **NO `min(raw, cap)` fudge in any budget**
3. **NO sanitize_state masking of broken dynamics** — sanitize firing rate <5% is the test
4. Cite WRF `dyn_em/module_small_step_em.F:lineno` for every acoustic-substep formula
5. Cite WRF `dyn_em/module_em.F:lineno` for mu-continuity
6. Verify physical constants by computation (sound speed = sqrt(γRT) at reference state)
7. M6-S5 H2D regression (F-3) must be resolved or explicitly documented
8. `/exit` slash-command; watchdog + multi-Enter

## End-goal context

This is the CRITICAL-PATH sprint for the entire project. If M6.x PASSES, M7 dispatch authorized, M6-S8 operational closeout can land, end-goal (Canary daily forecast) is in reach. If M6.x FAILS, project goes to option (c) re-architecture (Klemp-Skamarock alternative, semi-implicit, or hybrid ML emulator). User has full Gemini + Opus budget on call.
