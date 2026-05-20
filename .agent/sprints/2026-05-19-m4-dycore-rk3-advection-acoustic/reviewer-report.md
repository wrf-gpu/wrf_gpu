# Reviewer Report — M4 Dycore RK3 Advection Acoustic

Role: reviewer (opus-reviewer / Codex gpt-5.5). Branch: `reviewer/opus/m4-dycore-rk3-advection-acoustic`. Review target: attempt 2, after worker fix-cycle and tester branch.

## Findings

1. **major — debug snapshots do not implement the contracted JAX-side ring buffer.** AC 2.2 requires `snapshot(..., enabled=True)` to write to an in-JIT ring buffer carried only in debug traces (`.agent/sprints/2026-05-19-m4-dycore-rk3-advection-acoustic/sprint-contract.md:172`). The implementation instead stores host-side globals in `_SNAPSHOTS` and writes through `jax.debug.callback` (`src/gpuwrf/debug/snapshots.py:14`, `src/gpuwrf/debug/snapshots.py:17`, `src/gpuwrf/debug/snapshots.py:32`). It also keys by stage name, so a multi-step debug run records only `rk1`, `rk2_acoustic`, and `rk3_acoustic`, not a last-N temporal ring. Production HLO stripping is clean, but the debug feature itself is weaker than the constitutional AC.

2. **major — acoustic correctness is still only a reduced proxy, with no manufactured sound-wave oracle.** AC 1.3 requires a forward-backward acoustic substep (`sprint-contract.md:151`), and AC 11.1 requires operator tests to pass small manufactured-solution checks (`sprint-contract.md:296`). The acoustic update uses proxy constants `c2 = 1.0` and `pressure_coupling = 1.0e-3` (`src/gpuwrf/dynamics/acoustic.py:69`, `src/gpuwrf/dynamics/acoustic.py:70`), while `tests/test_m4_acoustic.py:16` only checks shapes and `tests/test_m4_acoustic.py:26` only checks finiteness. The maintainability note is honest that this remains a reduced proxy (`artifacts/m4/maintainability.md:11`), but physics claims require analytic/fixture evidence.

3. **minor — Tier-2 mass evidence is a tracer-total surrogate, not WRF-canonical mass.** The fix-cycle allowed a simpler tracer-translation path, and it now produces a nontrivial trajectory. Still, the invariant is `theta_total` (`src/gpuwrf/validation/tier2.py:93`, `src/gpuwrf/validation/tier2.py:114`; `artifacts/m4/tier2_invariants.json:6`), not `mu`/density mass. This is acceptable for the amended M4 proof but must not be carried forward as dycore mass-continuity evidence.

4. **note — the M5 dry-run trips on launch count as expected.** The artifact reports `kernel_launches_per_step = 24` against threshold 10 and records register/local-memory metrics as unknown JSON nulls (`artifacts/m4/m5_gate_dryrun.json:2`, `artifacts/m4/m5_gate_dryrun.json:4`, `artifacts/m4/m5_gate_dryrun.json:6`). Per AC 6.4 this is not a sprint failure, but it is a manager decision before M5 performance work.

## Contract Compliance

The attempt-1 blockers are resolved: RK3 constant tendencies integrate to one `dt`; Tier-1 now compares `advect_mass_scalar` against the `analytic-stencil-3d-upwind5-v1` sibling fixture (`artifacts/m4/tier1_advection_parity.json:2`); Tier-2 is nontrivial and passes (`artifacts/m4/tier2_invariants.json:4`, `artifacts/m4/tier2_invariants.json:10`); Tier-3 uses public `run(...)` and passes (`src/gpuwrf/validation/tier3.py:14`, `artifacts/m4/tier3_convergence.json:24`); the hand-stripped HLO diff is empty.

File ownership is acceptable for attempt 2. The diff includes manager/tester lifecycle changes (`sprint-contract.md`, `scripts/dispatch_role.sh`, tester reports/tests) plus the manager-approved fixture extension. I did not modify source files.

I read every line of `src/gpuwrf/dynamics/step.py` and `src/gpuwrf/debug/asserts.py` and found 0 simplification opportunities in those two files.

## Correctness Risks

The remaining risks are bounded but real: debug snapshots are not the specified JAX-side ring, acoustic dynamics are not validated as sound-wave dynamics, Tier-2 mass is a tracer surrogate, and the Tier-3 case is 1D advection with no velocity cross-term convergence oracle. These do not invalidate the repaired advection/RK proof objects, but they should be fixed before using M4 as evidence for coupled physics behavior.

## Performance Risks

Zero post-init transfer and zero temporary-byte artifacts are present, and my HLO identity rerun reproduced the empty diff. No register, local-memory, occupancy, or bandwidth evidence exists because `ncu/nsys` counters were not collected. The 24-launch dry-run trip should either open an M4.x fusion sprint or be explicitly accepted by the M5 manager gate.

## Independent Spot-Checks Run

- `git diff --stat main...HEAD`, `git diff --check main...HEAD`, and `git diff --name-status main...HEAD`.
- `JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 python scripts/m4_run_validation.py` -> Tier-1/2/3 all `pass: true`; reproduced Tier-3 observed order `4.65287662292045`.
- `JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 python scripts/m4_hlo_diff.py` -> diff size 0, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 python scripts/m4_m5_gate_dryrun.py` -> `gate_status: trip`, launch count 24, register/local metrics null.
- Focused pytest: `tests/test_m4_rk3.py tests/test_m4_tier1.py tests/test_m4_tier2_invariants.py tests/test_m4_tier3_convergence.py tests/test_m4_debug_hooks.py tests/test_m4_tester_adversarial_attempt2.py` -> `45 passed in 93.52s`.
- Independent probes: constant theta tendency residual `1.15e-14`; 5-step non-noop theta delta `0.5089`; debug snapshot dump after a 2-step debug run contained only three stage keys.
- `python scripts/validate_agentos.py` -> ok.

## Required Fixes

1. Either implement AC 2.2 literally with a debug-only JAX-side last-N snapshot ring, or amend the sprint contract/goal to bless the current host-callback snapshot behavior. Add a test that a multi-step debug run retains temporal snapshots, not only one entry per stage name.
2. Add a small-amplitude acoustic manufactured-solution/phase-speed test, or explicitly document the M4 acoustic module as call-shape-only proxy evidence and defer physical acoustic validation to a named M4.x/M5 gate. Do not treat the current acoustic artifact as physics-valid sound-wave evidence.
3. Before M5 closure, add a true mass-continuity diagnostic and a 2D cross-term convergence oracle, or keep the current Tier-2/Tier-3 evidence scoped to tracer advection only.

Decision: Accept with required fixes
