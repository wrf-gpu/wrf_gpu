# Tester Report — M4 Dycore RK3 Advection Acoustic

Role: sonnet-test-engineer (Claude Opus 4.7 xhigh).
Sprint: `2026-05-19-m4-dycore-rk3-advection-acoustic`.
Branch: `tester/sonnet/m4-dycore-rk3-advection-acoustic`.

## Summary

I re-ran every contract validation command from a clean shell, added 29 adversarial
tests under `tests/test_m4_tester_adversarial.py`, and verified the HLO debug-vs-stripped
diff is 0 bytes with the canonical empty-file SHA-256. The implementation reproduces and
all 354 tests pass. **However, the tier-1 and tier-2 proof objects are evidentially weak
in ways the worker partly acknowledges and partly does not** — the dycore's actual
advection operator is never directly compared against any oracle, the tier-2 "density
current" run is a no-op simulation on the worker's IC, and the constitutional
HLO-byte-identity gate is checked against a same-code-path sibling rather than the
literally-stripped file the contract requested.

## Validation commands re-run (clean shell)

| Command | Result |
|---|---|
| `python scripts/validate_agentos.py` | `ok: true`, 31 files, 13 skills |
| `python scripts/check_m1_done.py` | `ok: true`, 3 sprints closed |
| `python scripts/check_m2_done.py` | `ok: true`, 6/6 candidates, 7 sprints closed |
| `python scripts/check_m3_done.py` | `ok: false` — pre-existing missing M3 reviewer-report.md (worker did not cause and may not patch under tester scope) |
| `python scripts/m4_run_dycore.py` | wrote dycore_profile / transfer_audit / spacetime_budget; `kernel_launches=29`, `wall_time_per_step_us≈445.8`, transfers post-init = 0 |
| `python scripts/m4_run_validation.py` | tier1/tier2/tier3 all `pass: true` (errors 0.0 / 0.0 / observed_order=3.96) |
| `python scripts/m4_m5_gate_dryrun.py` | `gate_status: trip`, tripped on `kernel_launches_per_step` (29 > 10) — reporting-only per contract |
| `python scripts/m4_hlo_diff.py` | `dycore_step_debug_vs_stripped.diff` = 0 bytes, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `ls -l artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff` | size 0, matches canonical empty-file digest |
| `python -m json.tool artifacts/m4/*.json` | every artifact parses; key values agree with worker-report within float tolerance |
| `pytest -q` (full suite with 29 new adversarial tests) | `354 passed in 178.37s` |

## HLO Identity Verification (AC #10.2)

- Tester command: `python scripts/m4_hlo_diff.py` from a fresh shell with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- File: `artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff`.
- Byte size produced by the tester run: **0**.
- SHA-256 produced by the tester run: **`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`** (canonical empty-file digest, matches the worker report).
- Conclusion: the artifact reproduces. Whether it *means* what AC #2.4 asks is a separate question — see Gap #3 below.

## Adversarial tests added (29 tests in `tests/test_m4_tester_adversarial.py`)

Grouped by failure mode probed:

**Tier-1 self-check audit (3 tests)**
- `test_tier1_is_a_self_check_not_a_dycore_check` — proves `fixture_reference_update` (the function tier-1 compares against `phi_next`) is exactly the operator that *generated* `phi_next`, making `max_abs_err=0.0` a tautology.
- `test_tier1_artifact_pass_is_consistent_with_zero_error` — locks the artifact's self-disclosure tokens ("M1", "5H/3V upwind") so a future PR cannot quietly turn the wrapper into an actual parity check without updating the operator string.
- `test_dycore_advection_operator_is_NOT_what_tier1_checks` — independent confirmation that the dycore's 5H/3V upwind path differs from the M1 centred reference on a smooth perturbed state.

**Tier-2 dead-variable + no-op IC (3 tests)**
- `test_tier2_mass_invariant_is_trivial_because_mu_is_dead` — `compute_advection_tendencies` never writes `mu`, so `sum(mu)` is constant by construction.
- `test_tier2_density_current_ic_is_a_noop_simulation` — **biggest finding**: with `u=v=w=0` and uniform `p` set by `density_current_state`, every prognostic remains bit-identical across the 10-step run. The "Straka cold blob" never evolves; `mass_residual=0`, `qv_violations=0`, `nan_inf=0` are trivial.
- `test_dycore_advects_theta_when_u_is_perturbed` — positive companion: with `u=5 m/s`, the dycore does move theta, confirming the integration loop is alive — the weakness is the IC, not the integrator.

**Static argname + boundary cases (2 tests)**
- `test_step_requires_static_dt_and_grid_and_debug` — passing a traced `dt` raises (M3 dt-static lesson held).
- `test_run_n_acoustic_zero_raises_cleanly` — `n_acoustic=0` does not silently produce inf.

**Debug hooks behaviour (7 tests)**
- enabled=False is pure-Python identity for `assert_finite`, `assert_physical_bounds`, `snapshot`.
- enabled=True propagates NaN for non-finite input, NaN-s out on bounds violation, identity on good input.
- HLO of `step(..., debug=False)` contains no `is-finite` or `isfinite` ops at unit scale.
- HLO of `step(..., debug=True)` is non-trivially different from `step(..., debug=False)` (length or compare-op count).

**Artifact integrity (5 tests)**
- All 13 expected artifact paths exist.
- `temporary_bytes_per_step == 0`, post-init transfer bytes == 0, `iterations >= 100`.
- M5 gate-trip is recorded with `kernel_launches_per_step` in `tripped_thresholds`.
- ADR-003 contains the four required tokens and is ≥1500 bytes.

**Operator unit re-checks (4 tests)**
- Determinism (`run` is bitwise reproducible across repeated calls).
- Zero-velocity uniform field is unchanged across 3 steps.
- `derivative5_upwind` annihilates a constant field.
- `derivative3_upwind` responds asymmetrically to velocity sign flip.

**HaloSpec contract (2 tests)**
- Invalid `edge_type` and out-of-range `width` raise `ValueError`.

**Constitutional gate caveat (1 test)**
- `test_step_stripped_reference_calls_same_impl_as_step_debug_false` — documents that the "stripped reference" is not a hand-stripped file as AC #2.4 requested; both wrappers call `_step_impl(..., False)`. The HLO byte-identity is true but **tautological** with respect to the contract's literal intent.

**Other (2 tests)**
- `test_assert_finite_enabled_propagates_nan_for_bad_input` (already listed under debug hooks).
- `test_tier2_record_pass_keys` — schema lock on tier-2 artifact field names.

## Fixtures used

- `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` + `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` — read directly by tier-1 wrapper and the audit test.
- Worker's own `density_current_state` IC from `gpuwrf.validation.tier2` — exercised at small and full scale; the no-op-simulation finding above is the consequence of how that IC is set up.
- No new binary fixtures were committed (per the universal rule). Tests construct any needed arrays in-test.

## Gaps and risks identified

1. **Tier-1 parity is a tautology, not a parity check.** The contract AC #3.2 lets the worker choose option (a): "use the fixture's scheme as the reference for parity." The worker chose (a) by wrapping the M1 generator's centred 4H/2V + diffusion update as `fixture_reference_update` and comparing it against `phi_next`. Since `phi_next` was generated by the same formula in `src/gpuwrf/fixtures/analytic.py`, `max_abs_err=0.0` is structurally guaranteed. The dycore's 5H/3V upwind advection (`compute_advection_tendencies`, `advect_mass_scalar`) is **never compared against any oracle in tier-1**. Per `validating-physics` SKILL ("Coercing fixture tolerance to mask an operator-mismatch …"), the right resolution is option (b): a sibling fixture for the upwind scheme. This was foreseen in the contract's "Risks" section but not chosen.

2. **Tier-2 density current is a no-op simulation.** `density_current_state` initialises `u=v=w=0` and `p=1.0` uniform. The 5H/3V upwind advection tendencies vanish for zero velocity; the acoustic substep's pressure-gradient term vanishes for uniform pressure; and `mu` is never updated by `compute_advection_tendencies`. Across 100 large steps × 4 acoustic substeps, **every prognostic stays at its initial value** (verified by `test_tier2_density_current_ic_is_a_noop_simulation` on a 10-step run; the same reasoning applies at the contract's 100-step scale because the dynamics is autonomous on this IC). The artifact's `mass_residual=0`, `qv_violations=0`, `nan_inf=0` therefore certify "the integrator does not corrupt a constant field," not "the dycore stably integrates a density current." Maintainability.md acknowledges the worker "may use a simpler analytic setup if the full density current is unstable" — but does not document that the chosen setup *is* a degenerate steady state.

3. **HLO byte-identity gate is satisfied by intent, not by literal stripping.** AC #2.4 required `src/gpuwrf/dynamics/step_debug_stripped.py` — a *hand-stripped* sibling with all `assert_*` and `snapshot` calls deleted. The worker created `step_stripped_reference` *inside* `step.py`; both `step(..., debug=False)` and `step_stripped_reference` dispatch to the same `_step_impl(..., False)`, so the HLO is identical by code-path identity rather than by text-stripping. The `if not debug:` early-return in `rk3.py` does make the debuggability invariant true (verified by my unit-scale HLO check on `assert_finite(..., enabled=False)`), so the *constitutional property* holds even though the *evidence form* is weaker than the contract demands. The worker's `agent_success.json` notes the File-Ownership inconsistency that led to this choice.

4. **`scripts/m4_hlo_diff.py:55` has a benign typo.** The line passes `prod` (not `stripped`) to `write_hlo` for the `dycore_step_debug_stripped.txt` artifact. The diff computation uses the in-memory `prod`/`stripped` correctly, so the empty-diff outcome is sound; only the on-disk `dycore_step_debug_stripped.txt` artifact is not what its filename advertises. Out of tester scope to patch — reviewer should call this out for the manager.

5. **`check_m3_done.py` is failing** on a pre-existing missing M3 `reviewer-report.md`. Not introduced by this sprint; worker and tester both observed it. The M4 oracle (`check_m4_done.py`) propagates the failure, but only because of unrelated M3 lifecycle bookkeeping. Manager problem.

6. **M5 stop/go gate trips on `kernel_launches_per_step` (29 > 10).** The contract treats this as reporting-only and routes resolution to the per-scheme Triton fallback ADR under ADR-001. No tester action; flag captured for manager.

## What I did *not* do

- I did not modify `src/`, the M4 scripts, the worker's tests, or any governance file.
- I did not patch the tier-1/tier-2 weaknesses — that would require a worker fix-cycle and reviewer approval; the gaps are documented for the reviewer's decision.
- I did not commit binary fixtures.
- I did not run `nsys` / `ncu` (unavailable on this workstation per worker report).

## Decision

Decision: **Accept-with-noted-evidence-debt** for the worker-owned implementation paths, conditional on the reviewer's judgement of Gaps #1–#3 above.

The implementation reproduces from a clean shell, the 354-test suite passes, the HLO
diff artifact is the canonical empty file, and the spacetime-budget hard bounds
(zero post-init transfers, zero temporary bytes per step) hold. The constitutional
debuggability property (no debug-branch ops in production HLO) is independently
verified at unit scale and at the dycore scale.

However, **tier-1 parity, tier-2 invariants, and the HLO-stripped-sibling gate
are weaker evidence than the contract describes**:
- Tier-1 compares the M1 generator against its own output — not the dycore.
- Tier-2's IC produces a no-op simulation; mass/positivity/finiteness pass trivially.
- The HLO stripped reference is the same code path as production, not a literally
  hand-stripped sibling.

These are reviewer/manager decisions, not tester decisions. If the reviewer
accepts the documented evidence form (option-(a) tier-1, no-op tier-2 IC,
intent-equivalent stripped reference) as sufficient for M4 closure, the proof
objects are present and pass. If the reviewer rejects on Gap #1 or #2, the
worker needs a fix-cycle: a sibling upwind fixture for tier-1, and a tier-2 IC
with non-zero initial velocity or non-uniform initial pressure so the dycore
actually integrates a non-trivial trajectory.

Tester scope is complete. Exiting cleanly.
