# Sprint Contract Template — M6.x-FALLBACK-c1 Klemp-Skamarock Clean-room Vertical-Implicit Acoustic

**Sprint ID**: `<YYYY-MM-DD>-m6x-fallback-c1-klemp-skamarock` (fill date at dispatch time)
**Status**: TEMPLATE — dispatchable only if M6.x fails per ADR-017 invocation rule
**Trigger**: M6.x landed RED on at least one of: Tier-2 lifted-cap invariants FAIL, sanitize firing rate ≥5% of steps, 24h forecast NaN-explodes, end-to-end speedup <4×. Manager invokes (c1) per ADR-017 §Invocation.
**Parent**: M6.x (`2026-05-22-m6x-wrf-canonical-dycore`) — read its sprint-contract.md and worker-report.md as Reading B before starting.

## Objective

Replace the broken M4 reduced + M6.x partial dycore with a clean-room implementation of **Klemp, Skamarock, Dudhia (2007) §3a-c** split-explicit time integration: horizontal-explicit + vertical-implicit acoustic with physical sound speed; tridiagonal solve per column per substep; canonical μ-continuity from small-step mass-flux accumulator. Goal: pass M6.x ACs (Tier-2 lifted-cap, sanitize <5%, 24h finite, speedup ≥4×) **without** porting the full 2089-line WRF `module_small_step_em.F`. Trade fidelity (no hybrid-coord, no `epssm` off-centering, no msf complications) for implementation tractability.

## Acceptance

- **AC1 Physical sound speed**: `acoustic_once` uses `c² = γ R T̄` with γ=1.4, R=287.05, T̄=260K (or per-cell `c² = γ R T(x,y,z)` if cheap). Cite Klemp 2007 Eq. 6.
- **AC2 Vertical-implicit (w, φ') tridiagonal**: per-column Thomas solve replaces explicit `_grad_z_to_w` (`acoustic.py:60-77`). Coefficients (a, b, c) computed once per large RK step; back-substitute per acoustic substep. Cite Klemp 2007 Eq. 19-22.
- **AC3 Horizontal-explicit (u, v) momentum**: forward-Euler at small dt with `c² ∂p'/∂x`, `c² ∂p'/∂y` pressure gradient. Cite Klemp 2007 Eq. 16-17.
- **AC4 μ-continuity via small-step mass-flux accumulator**: small-step (ru, rv) fluxes accumulated through `lax.scan` of acoustic substeps; `∂μ/∂t = -∇·(μV)` applied at end of large RK stage. Cite Klemp 2007 Eq. 12 and §3c. Reference WRF `dyn_em/module_em.F:advance_mu_t` (canonical equation only, not implementation).
- **AC5 Per-cell CFL diagnostic per substep**: `c·dt_sub/dx ≤ 1` and `c·dt_sub/dz ≤ 1`. Bind `n_acoustic` from CFL: if max-cell CFL > 0.8, increase n_acoustic. Document binding in `ADR-018` (NEW).
- **AC6 Tier-2 lifted-cap invariants PASS**: re-run M6-S5 verdict harness (`scripts/m6_full_domain_batching.py`). `tier2_lifted_cap_invariants.json` must show NaN/Inf=0, sanitize firing rate <5%, final state away from clip bounds (theta ∈ [200,350]K, qv well below 0.05, w well below 50 m/s).
- **AC7 24h forecast on real d02 finite + physically valid**: full 24h at dt=10s on (160×67×45) d02 grid, dycore actually integrating, no silent saturation, final w⋅theta⋅qv fields look like a real WRF output (visual sanity: tester runs ncview on output and confirms ≥1 weather feature looks meteorological).
- **AC8 Speedup ≥4×**: rerun `m6_full_domain_batching.py`; end-to-end wall ≤ CPU-baseline/4. Target ≥6×.
- **AC9 Zero post-init H↔D transfer regression**: `artifacts/transfer_audit.json` after the new dycore: host_to_device_bytes_post_init = 0, device_to_host_bytes_post_init = 0 (or strictly ≤ M6.x F-3 baseline of 164 KB, with a documented rationale memo).
- **AC10 Tridiagonal backend choice documented**: a 1-page mini-ADR (`ADR-018-m6x-fallback-c1-tridiag-backend.md`) selects between `jax.lax.linalg.tridiagonal_solve` (c1.A) and hand-rolled vmapped Thomas with `lax.scan` over k (c1.B). Decision-grade artifact: benchmark JSON at (1, 160, 67, 45) batch shape comparing both; pick the faster; document the loser's failure mode.
- **AC11 NEW ADR**: `ADR-019-m6x-fallback-c1-klemp-skamarock-clean-room.md` documenting: clean-room scope (not a WRF port), Klemp 2007 equation provenance, where this differs from WRF dyn_em (no hybrid coord, no off-centering, no msf, no sumflux time-averaging), operational limits (gravity-wave phase speeds may be 1-3% off canonical).
- **AC12 ADR-007 status update**: amend ADR-007 from FAIL to PASS-with-evidence after AC6+AC7+AC8 all green.

## Files Worker May Modify

- `src/gpuwrf/dynamics/acoustic.py` (REWRITE per AC1-AC3, ~250 LoC up from 92).
- **NEW** `src/gpuwrf/dynamics/tridiag.py` (~50-100 LoC: Thomas solve, vmapped over (j, i), depending on AC10 outcome).
- `src/gpuwrf/dynamics/rk3.py` (sumflux accumulator threaded through `lax.scan`, ~20 LoC delta).
- `src/gpuwrf/dynamics/step.py` (no algorithmic changes; CFL diagnostic call-shape, ~10 LoC delta).
- `src/gpuwrf/dynamics/tendencies.py` (μ-continuity Euler update path, ~20 LoC delta).
- `src/gpuwrf/contracts/state.py` (add optional `mu_tendency` diagnostic; preserve SoA per ADR-002; ~10 LoC delta).
- `src/gpuwrf/coupling/driver.py` (no dycore cap; tighter integration; ~20 LoC delta).
- `scripts/m6_full_domain_batching.py` (rerun verdict harness, no logic change).
- `tests/test_m6x_fallback_c1_*.py` (NEW: tridiag correctness, manufactured sound-wave phase speed, μ-continuity, 24h smoke).
- `.agent/decisions/ADR-007-precision-policy.md` (status amend after AC6+AC7+AC8).
- `.agent/decisions/ADR-018-m6x-fallback-c1-tridiag-backend.md` (NEW per AC10).
- `.agent/decisions/ADR-019-m6x-fallback-c1-klemp-skamarock-clean-room.md` (NEW per AC11).
- `artifacts/m6x-fallback-c1/`: `tier2_lifted_cap_invariants.json`, `full_domain_batching_verdict.json`, `transfer_audit.json`, `cfl_diagnostic.json`, `tridiag_benchmark.json`.

## Files Worker Must NOT Modify

- `src/gpuwrf/physics/**` (Thompson/MYNN/RRTMG/sfclay/Noah-MP — FROZEN per M6.x AC8).
- `src/gpuwrf/coupling/{physics_couplers,boundary_apply}.py` body (only consume new dycore outputs).
- `src/gpuwrf/io/**` (frozen).
- `src/gpuwrf/validation/**` body (only re-run; do not change kernels).
- ADR-001, ADR-002, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009, ADR-010, ADR-011, ADR-012, ADR-013, ADR-014 (only cross-ref updates).

## Dispatch

- Worker: **codex gpt-5.5 xhigh** (large dycore work — codex's strength).
- Reviewer: **Claude Opus 4.7 xhigh** (cross-AI; this is the project-pivot sprint).
- Wall-time budget: **5-9 wall-days** (3-5 days implementation; 2-3 days CFL + μ-continuity; 1-2 days validation).
- Worktree: `/tmp/wrf_gpu2_m6x_c1` (NEW; not the same as M6.x worktree).
- Branch: `worker/codex/m6x-fallback-c1-klemp-skamarock`.

## HARD RULES

1. **Clean-room from Klemp 2007 paper equations** — do NOT line-by-line port `module_small_step_em.F`. M6.x already failed that route.
2. **NO physics-kernel changes** — physics is frozen.
3. **NO `min(raw, cap)` fudge in any budget**.
4. **NO sanitize_state masking of broken dynamics** — sanitize firing rate <5% is the test.
5. Cite Klemp et al. 2007 equation numbers in every dynamics docstring touched.
6. Verify physical constants by computation (sound speed = √(γRT̄) at reference state; M6.x AC1 same rule).
7. M6-S5 H2D regression (F-3) must be resolved or explicitly documented; preserve M4 zero-transfer constitutional gate.
8. Tridiagonal backend choice (AC10) requires a real benchmark JSON, not "I think A is faster."
9. Manufactured sound-wave phase-speed test is binding: implement and pass before claiming acoustic correctness.
10. `/exit` slash-command; watchdog + multi-Enter.

## End-goal context

If M6.x failed, the project's pivot decision tree per ADR-017 invokes (c1) FIRST because it has the lowest architectural risk (well-established algorithm, published reference equations, preserves M4 SoA pytree + zero-transfer invariants). Success here unblocks M7 dispatch. Failure escalates to (c2) semi-implicit per ADR-017 §Invocation.

The reason (c1) is cleaner than a faithful M6.x WRF port: it omits hybrid sigma-pressure coordinate, omits `epssm` time off-centering, omits map-scale-factor plumbing in every flux, omits top-lid boundary distinctions, omits sumflux time-averaging gymnastics. These are operational refinements WRF accumulated over 20+ years; they're not required for a 24h Tier-2 forecast on a single d02 grid. Trade fidelity for tractability.

## Open questions for manager at dispatch time

1. Does the reviewer want a manufactured gravity-wave phase-speed test in addition to acoustic? (Adds 1 day; defensible if M7 starts caring about gravity-wave-resolved physics.)
2. If the M6.x worker produced partial code that's salvageable (e.g., physical sound-speed constant binding), do we cherry-pick or start from M4 main? Default: start from M4 main; cherry-pick risks importing the M6.x failure mode.
3. Per-cell c² (AC1 stretch) vs reference-state T̄ c² (AC1 baseline)? Baseline is safer for first pass; per-cell adds ~2× memory traffic on c²a array but is more physical at the stratosphere.
