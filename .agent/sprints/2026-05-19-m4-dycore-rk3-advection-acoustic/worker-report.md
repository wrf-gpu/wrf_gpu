# Worker Report — M4 Dycore RK3 Advection Acoustic

Summary: Attempt 2 fixes the reviewer-blocking M4 issues inside the worker-owned scope: RK3 constant-tendency scaling, real Tier-1 dycore upwind oracle, nontrivial Tier-2 trajectory, Tier-3 through public `run(...)`, velocity cross-advection, real hand-stripped HLO sibling, HLO artifact typo, JSON-null M5 dry-run unknown metrics, and no-wrap vertical advection boundaries. The M5 dry-run still trips on `kernel_launches_per_step=24`; per contract this is reporting-only.

## Objective

Deliver the reduced split-explicit dry dycore and proof objects required by `.agent/sprints/2026-05-19-m4-dycore-rk3-advection-acoustic/sprint-contract.md`, without editing reviewer/tester/manager reports, memory patches, or governance files.

## Files changed

- `src/gpuwrf/dynamics/{advection.py,rk3.py,step.py,step_debug_stripped.py}`
- `src/gpuwrf/validation/{tier1.py,tier2.py,tier3.py}`
- `scripts/{generate_analytic_fixtures.py,m4_run_dycore.py,m4_run_validation.py,m4_m5_gate_dryrun.py,m4_hlo_diff.py}`
- `fixtures/manifests/analytic-stencil-3d-upwind5-v1.yaml`
- `fixtures/samples/analytic-stencil-3d-upwind5-v1.npz`
- `tests/test_m4_{advection,debug_hooks,rk3,tier1,tier2_invariants,tier3_convergence}.py`
- `tests/test_m4_tester_adversarial.py` updated to the contract’s required inverted form.
- `artifacts/m4/{dycore_profile.json,transfer_audit.json,spacetime_budget.json,tier1_advection_parity.json,tier2_invariants.json,tier3_convergence.json,m5_gate_dryrun.json,maintainability.md,agent_success.json,hlo_dump/*}`
- `.agent/decisions/ADR-003-dycore-precision.md`
- This worker report.

## Commands run + output

- `python scripts/validate_agentos.py`
  Output: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`

- `python scripts/check_m1_done.py`
  Output: `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`

- `python scripts/check_m2_done.py`
  Output: `{"candidates_satisfied": 6, "candidates_total": 6, "errors": [], "ok": true, "sprints_closed": 7}`

- `python scripts/check_m3_done.py`
  Output: failed only on existing lifecycle closeout: `sprint 2026-05-19-m3-state-grid-halo-skeleton not closed: missing reviewer-report.md`.

- `JAX_ENABLE_X64=True ... python scripts/m4_run_dycore.py`
  Output summary: `host_to_device_bytes_post_init=0`, `device_to_host_bytes_post_init=0`, `temporary_bytes_per_step=0`, `kernel_launches_per_dycore_step=24`, `wall_time_per_step_us=634.0713601093739`.

- `JAX_ENABLE_X64=True ... python scripts/m4_run_validation.py`
  Output summary: Tier-1 `pass=true max_abs_err=0.0 max_rel_err=0.0 fixture=analytic-stencil-3d-upwind5-v1`; Tier-2 `pass=true mass_residual_relative=1.937334197150901e-16 final_state_differs_from_initial=true`; Tier-3 `pass=true observed_order=4.65287662292045 expected_order=3.0`.

- `JAX_ENABLE_X64=True ... python scripts/m4_m5_gate_dryrun.py`
  Output: `gate_status=trip`, `kernel_launches_per_step=24`, `local_memory_bytes_per_kernel=null`, `registers_per_kernel=null`, `tripped_thresholds=["kernel_launches_per_step"]`.

- `JAX_ENABLE_X64=True ... python scripts/m4_hlo_diff.py`
  Output: `artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff bytes=0 sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

- `ls -l artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff`
  Output: `-rw-rw-r-- 1 enric enric 0 May 20 01:18 artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff`.

- `python -m json.tool` on all seven M4 JSON proof objects
  Output: `all M4 JSON proof objects parse`.

- `JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -q`
  Output: `359 passed in 226.76s (0:03:46)`.

- `JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/check_m4_done.py`
  Output: failed on lifecycle gates outside worker scope: prior `check_m3_done.py` reports missing M3 `reviewer-report.md`; current sprint missing `manager-closeout.md` and `memory-patch.md`; missing `.agent/decisions/MILESTONE-M4-CLOSEOUT.md`; M4 milestone reviewer decision not yet Accepted.

Note: `check_m2_done.py` rewrote several tracked M2 fallback profile JSONs as a side effect of its internal checks. I restored those unrelated M2 artifact changes before handoff.

## Proof objects

- `artifacts/m4/tier1_advection_parity.json`: dycore 5H horizontal periodic + 3V no-wrap upwind sibling fixture, `pass=true`.
- `artifacts/m4/tier2_invariants.json`: nontrivial tracer translation, `final_state_differs_from_initial=true`, `pass=true`.
- `artifacts/m4/tier3_convergence.json`: public `run(...)` convergence, `observed_order=4.65287662292045`, `pass=true`.
- `artifacts/m4/transfer_audit.json`: zero post-init H2D/D2H bytes over 100 iterations.
- `artifacts/m4/spacetime_budget.json`: `temporary_bytes_per_step=0`.
- `artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff`: empty, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `.agent/decisions/ADR-003-dycore-precision.md`: updated draft precision evidence.

## Spacetime Budget

| metric | value | justification |
|---|---:|---|
| `state_bytes` | 14540800 | eight fp64 prognostic leaves at 40x80x80 C-grid shapes |
| `tendency_bytes` | 14540800 | one matching fp64 tendency tree |
| `temporary_bytes_per_step` | 0 | production scan body has no `jnp.array/zeros/empty`; proof HLO emitted |
| `halo_buffer_bytes` | 0 | M3 single-GPU halo remains no-op call shape |
| `total_persistent_bytes` | 29081600 | state + tendencies + halo buffers |
| `kernel_launches_per_step` | 24 | HLO-derived estimate; trips M5 dry-run threshold |
| `wall_time_per_step_us` | 634.0713601093739 | median cached JAX dycore timing |

## Allocation Audit

- `src/gpuwrf/validation/tier1.py`: `jnp.asarray` fixture loads are init-only validation setup.
- `src/gpuwrf/validation/tier2.py`: `jnp.zeros`, `jnp.ones`, `jnp.asarray` are init-only validation-grid/state/counter setup.
- `src/gpuwrf/validation/tier3.py`: `jnp.zeros_like` / `jnp.ones_like` build the convergence initial condition before public `run`.
- `src/gpuwrf/debug/snapshots.py`: `jnp.asarray` is debug-only and not traced when `debug=False`.
- `src/gpuwrf/dynamics/rk3.py`: `jnp.asarray` is in `rk3_scalar_decay`, a test helper not called by hot-path `step/run`.
- Test-file `jnp.asarray/zeros/ones` calls are test-only.
- No `jnp.array`, `jnp.zeros`, or `jnp.empty` occurs in `step()` or functions transitively called by production `run(..., debug=False)`.

## Risks

- Acoustic constants remain reduced-proxy constants (`c2=1.0`, pressure coupling `1e-3`), and the acoustic substep still does not take a separate slow-mode source term. Documented in `maintainability.md`; not expanded in this fix cycle.
- Tier-2 uses the contract-approved simpler tracer-translation setup rather than a full Straka density current.
- Tier-3 timestep levels are 2/1/0.5 s instead of the suggested 20/10/5 s because the unfiltered upwind proxy is unstable at the larger CFL.
- `JAX_ENABLE_X64=True` is required for stable convergence evidence; M4 scripts now set it before importing JAX.
- M5 gate dry-run trips on kernel launches (`24 > 10`); local/register metrics remain unavailable from HLO and are JSON null.

## Handoff

Objective complete for worker-owned M4 implementation and proof objects. Next decision needed: reviewer/tester should verify attempt 2, and manager should decide whether the M5 launch-count trip requires an M4.x fusion sprint or can be handled by the M5 fallback gate. Manager-owned closeout files remain intentionally untouched.
