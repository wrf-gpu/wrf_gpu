# Sprint Contract — F1: Real WRF Fortran Savepoint Comparator (Rewrite m6b6)

**Sprint ID**: `2026-05-28-f1-wrf-fortran-savepoint-comparator`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/f1-real-wrf-savepoint-comparator`
**Worktree**: `/tmp/wrf_gpu2_f1`
**Wall-time**: 3-5 days (best effort; longer OK if WRF build is hairy)
**GPU usage**: NO for build; YES for final JAX comparison run
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

Per [`agy review`](../2026-05-28-agy-dycore-deep-review/findings.md): the
`scripts/m6b6_coupled_step_compare.py` "100-step bitwise WRF parity" test is a
**JAX-vs-JAX self-compare tautology** — `emit_tier` writes JAX outputs and
`compare_tier` reads them back from the same path. Every "PASS" since M6 has
verified consistency with itself, **not parity with WRF Fortran**. This single
gap is why three serious dycore bugs (advection deleted, mu wrong, theta
decouple wrong) survived dozens of test runs.

Until this test is rebuilt against ACTUAL WRF Fortran savepoints, no dycore
claim can be trusted. This sprint is the **most fundamental test infrastructure
work in the project right now**.

## Binding goal (universal)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72h RMSE on T2/U10/V10
**statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins
on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md` — primary directive
2. `scripts/m6b6_coupled_step_compare.py` — the tautological comparator (study `emit_tier` and `compare_tier`)
3. `tests/savepoint/test_dycore_100_steps.py` — the test that wraps the tautology
4. `/home/enric/src/wrf_gpu/` — the LEGACY/old project, DO NOT EDIT — but may contain WRF build paths to reference
5. Any CPU WRF build artifacts on disk — search via `find /home -name "wrf.exe" -o -name "ideal.exe" 2>/dev/null` (limit to first 2-3 hits)
6. `src/gpuwrf/dynamics/core/` — the JAX dycore being tested
7. `tests/savepoint/` — existing savepoint infrastructure to extend
8. `proofs/m11p3/` — recent state showing the broken baseline

## Approach

### Phase 1 — Instrument WRF Fortran (1-3 days)

a. Locate or build CPU WRF v4 binary on this workstation (`ideal.exe` preferred — B6 case is an ideal run).
b. Identify the WRF Fortran subroutines that own the same state transitions as our JAX dycore:
   - `dyn_em/module_em.F:rk_scalar_tend` (advection)
   - `dyn_em/module_small_step_em.F:advance_w`, `advance_uv`, `advance_mu_t` (acoustic substep)
   - `dyn_em/module_em.F:rk_phys_tend` (physics-as-tendency)
c. Add **minimal NetCDF or raw binary dumps** at:
   - End of each RK stage (3 dumps per timestep)
   - End of each acoustic substep (variable count: ns_acoustic per RK stage)
   - End of full timestep
d. Run B6 ideal case for 100 timesteps. Capture per-stage savepoints to disk under `tests/savepoint/fixtures/wrf_b6_100step/`.
e. Schema: one NetCDF per dump containing U, V, W, theta, mu, mut, p, pb, ph, phb, qv, **plus the same field names our JAX dycore uses**, with explicit dimension order.

**If WRF build/instrumentation proves too hairy in 2 days**: stop instrumentation, write a fallback (Phase 1b) that uses any existing CPU-WRF wrfout NetCDFs already on disk (`/home/enric/data/wrf_baseline/` or similar) as imperfect-but-real Oracle. Document the limitation.

### Phase 2 — Rewrite Comparator (1 day)

a. Rewrite `scripts/m6b6_coupled_step_compare.py`:
   - `emit_tier` should ONLY emit JAX outputs (rename to `emit_jax_savepoints`).
   - NEW function `load_wrf_savepoints(step, stage)` reads the WRF NetCDFs from Phase 1.
   - NEW function `compare_jax_vs_wrf(step, stage)` does the actual JAX-vs-WRF Fortran diff.
   - Preserve the existing call sites so `tests/savepoint/test_dycore_100_steps.py` still drives it.
b. Add per-field tolerance with explicit rationale (bitwise vs EQUIVALENCE_TIGHT vs EQUIVALENCE_LOOSE).
c. Add per-step + per-stage breakdown — show WHICH stage of WHICH timestep first diverges.

### Phase 3 — Run + Honest Report (1 day)

a. Run new comparator on current `manager-2026-05-23` HEAD (which has M11.3 applied).
b. Emit `proofs/f1/m6b6_real_wrf_comparison.json` with:
   - per-field per-step max-diff + mean-diff
   - first stage where JAX diverges from WRF Fortran
   - histogram of divergence vs WRF reference
c. Emit `proofs/f1/honest_dycore_position.md` — written for the principal:
   - "Bitwise WRF parity at how many steps actually?"
   - "Which JAX function first diverges?"
   - "Is M11.3's coordinated fix closer to or farther from WRF Fortran than baseline?"
   - "What's the next concrete sprint target?"

### Phase 4 — Hand-off

Update `tests/savepoint/test_dycore_100_steps.py` to use the new real-WRF
comparator. Document in `tests/savepoint/README.md` that the old self-compare
behavior is RETIRED and any claim of "100-step parity" must reference real WRF
Fortran savepoints, not the old tautology.

## Acceptance

- **AC1**: WRF Fortran B6 100-step run produces savepoints on disk under `tests/savepoint/fixtures/wrf_b6_100step/` (or, fallback, a documented real-WRF Oracle is wired up).
- **AC2**: `scripts/m6b6_coupled_step_compare.py` reads WRF Fortran outputs (not JAX echoes) for ground truth.
- **AC3**: `tests/savepoint/test_dycore_100_steps.py` produces an HONEST verdict: e.g. "PASS through step N, FAIL at step N+1 with first divergence in operator X" rather than a tautological PASS.
- **AC4**: `proofs/f1/honest_dycore_position.md` written — tells the principal where the dycore actually stands.
- **AC5**: Speedup numbers preserved (this sprint doesn't optimize, but ensure JAX run time on B6 hasn't regressed).

## Hard rules

1. **CPU pinning**: `taskset -c 0-3` for all Python/JAX work; WRF Fortran can use 4-31 if needed (offline build only, no parallel with workers).
2. **GPU usage**: only Phase 3 needs GPU.
3. **Files writable**: `scripts/m6b6_coupled_step_compare.py`, `tests/savepoint/test_dycore_100_steps.py`, `tests/savepoint/README.md`, `tests/savepoint/fixtures/wrf_b6_100step/`, `proofs/f1/**`, `.agent/sprints/2026-05-28-f1-.../**`.
4. **Files NOT writable**: any JAX dycore source code (`src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`), governance, plan, ADRs.
5. **DO NOT MODIFY the legacy repo** `/home/enric/src/wrf_gpu/`. Reference only.
6. **No remote push.**
7. **Manager repo ONLY**.
8. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: f1 DONE exit=$?" Enter`.
9. **End with verdict**: `F1_COMPLETE` if AC1-AC5 all pass; `F1_PARTIAL` with explicit gaps otherwise.

## Out of scope

- Don't fix dycore bugs (M11.3 manager sprint territory).
- Don't expand to per-operator parity tests (that's a follow-up sprint).
- Don't touch idealized cases (F2 territory).
- Don't touch oracle suite YAML (that's the Oracle sidecar sprint).
