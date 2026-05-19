# Tester Report

- Sprint: `2026-05-19-m3-state-grid-halo-skeleton`
- Role: sonnet-test-engineer (Claude Opus 4.7 xhigh)
- Branch: `tester/sonnet/m3-state-grid-halo-skeleton`
- Environment: `data/scratch/m2-jax-venv` (the pinned `jax[cuda13]==0.10.0` venv), `env -i` clean shell, `CudaDevice(id=0)` (RTX 5090 / cc120).

## Tests Added Or Run

I re-ran every command in the sprint contract from a `env -i` clean shell with only the `m2-jax-venv` on `PATH`, and added a focused adversarial test module `tests/test_m3_edge_cases.py` (45 new tests). The new module covers:

### Allocation discipline (AC §4.3, §8.2, §8.3)
- `test_hot_path_has_no_allocator_tokens` — static scan of `src/gpuwrf/timestep/dummy_loop.py` for `jnp.array|asarray|zeros|empty|ones|full|linspace|arange` and `jax.device_put`. Hard regression guard.
- `test_init_path_allocations_are_bounded_and_listed` — pins the exact set of allocators to grid.py + state.py only; expansion fails the test (so a future worker can't silently add `jnp.zeros` to halo.py or precision.py).

### GridSpec invariants (AC §1)
- Invalid projection kind, non-`hybrid_eta` vertical, `halo_width` below 1, `halo_width` above 4, terrain-provenance shape mismatch, terrain_height shape mismatch, eta_levels length mismatch, fp32 eta rejected, non-`c-grid` staggering rejected, eta monotone descending check, Canary template dimensions sanity, hash stability across rebuilds.

### State / Tendencies invariants (AC §2)
- `replace()` returns new object, leaves original untouched (immutability check); cross-field non-update preservation on Tendencies; pytree round-trip rebuilds typed `State` / `Tendencies` with 8 leaves each; `State.from_init(grid, Path)` matches `State.zeros(grid)`; per-field State↔Tendencies shape equality (scan-carry compatibility).

### Precision registry
- Unknown field raises `KeyError`; every named state field maps to fp64; factory is pure.

### HaloSpec / apply_halo (AC §3)
- Width 0 and width 5 rejected; all three documented edge types accepted; empty `fields_to_exchange` still allowed; `apply_halo` is object-identity for empty fields.

### Dummy step / loop (AC §4)
- `dummy_step` with zero tendencies is bitwise identity on θ; same `Tendencies` object returned (no reallocation).
- `run_dummy_loop` is deterministic across two calls; `n_steps=0` is a state passthrough; outputs stay on GPU.
- Different `n_steps` produce different compiled HLO → `n_steps` is genuinely static.
- HLO has exactly one `while(` op → exactly one JITed scan, matching AC §4.2.

### Proof-object schema (AC §5–§6, §7)
- `spacetime_budget.json`: `total_persistent_bytes == state_bytes + tendency_bytes + halo_buffer_bytes`; `state_bytes` / `tendency_bytes` match a freshly built `State.bytes()` and `Tendencies.bytes()`; `wall_time_per_step_us < 100`; persistent bytes under 1 GiB.
- `transfer_audit.json`: `method` is non-empty; `jax_version == jax.__version__`; `gpu_name` contains "cuda".
- HLO dump: present, size in (100 B, 100 KB), contains `HloModule jit_run_dummy_loop` and a `while(`.
- `agent_success.json` records the JAX pin.
- ADR-002 ≥ 1500 bytes and contains every required token (`Decision:`, `Layout:`, `Staggering:`, `Halo packing:`).
- `kernel_launches_per_step` helper is clamped to `[1, 5]` for empty input and saturates correctly.

### Validation commands re-run from clean shell
| Command | Result | Notes |
|---|---|---|
| `bash scripts/m2_run_jax.sh` | exit 0; `gpu [CudaDevice(id=0)]` | M2 JAX backend still loads cleanly. |
| `python -m gpuwrf.contracts.state --self-test` | exit 0; `ok state_bytes=38656 tendency_bytes=38656 device=gpu` | Matches worker's reported numbers. |
| `python scripts/m3_run_audits.py` | exit 0 | Idempotently regenerated both JSONs; new run produced `wall_time_per_step_us=2.819`, transfer bytes stayed zero. |
| `python -m json.tool artifacts/m3/transfer_audit.json` | exit 0 | Valid JSON; H2D=0, D2H=0, iterations=1000. |
| `python -m json.tool artifacts/m3/spacetime_budget.json` | exit 0 | Valid JSON; all six required keys present. |
| `head -50 artifacts/m3/hlo_dump/dummy_loop.txt` | exit 0 | Contains `HloModule jit_run_dummy_loop`, one `while.5`, and `%loop_subtract_fusion` calling `%fused_subtract` (broadcast / multiply / multiply / add / subtract — full theta no-op chain fused in one kernel). |
| `pytest -q tests/test_m3_*.py` | **58 passed in 1.41 s** | 13 worker tests + 45 new tester tests; no flakes. |
| `pytest -q` (full repo) | **231 passed, 8 skipped, 11 failed** | All 11 failures are M2 environment-state-dependent (Triton cuobjdump, Kokkos build, CUDA tile build, JAX edge cases tied to dirty M2 profile JSONs); none touch M3 code. See "Gaps" below. |
| `pytest --collect-only` count | 295 (was 250 before this sprint) | Comfortably above the AC §10.6 "≥ 250" floor *with this tester's contribution*. |
| `python scripts/check_m1_done.py` / `check_m2_done.py` / `check_m3_done.py` | non-ok | Failures are lifecycle / regression-from-environment, not M3 code; see "Gaps". |

## Allocation Audit

Static scan of every worker-owned file for allocator tokens (`jnp.array|asarray|zeros|empty|ones|full|linspace|arange`, `jax.device_put`):

| File | Line | Allocator | Grade | Justification |
|---|---:|---|---|---|
| `src/gpuwrf/contracts/grid.py` | 186 | `jnp.linspace` | **necessary** | Init-time eta-level coordinate in `canary_3km_template`; constructed exactly once and stored as a frozen `GridSpec` array leaf. |
| `src/gpuwrf/contracts/grid.py` | 187 | `jnp.zeros` | **necessary** | Init-time analytic terrain-height placeholder for the Canary template; stored as a frozen `GridSpec` leaf. Real ingestion is M5+. |
| `src/gpuwrf/contracts/state.py` | 32 | `jnp.zeros` | **necessary** | Single private `_zeros` helper used to allocate frozen `State` and `Tendencies` leaves at init only; never called from the scan body. |
| `src/gpuwrf/contracts/state.py` | 32 | `jax.device_put` | **necessary** | Forces each new array onto the resolved GPU device (`_gpu_device()` raises if none); satisfies AC §2.4. |
| `src/gpuwrf/timestep/dummy_loop.py` | — | none | **necessary** | Hot path: zero array constructors. Verified statically by `test_hot_path_has_no_allocator_tokens`. |
| `src/gpuwrf/contracts/halo.py` | — | none | **necessary** | Single-GPU no-op; nothing to allocate. |
| `src/gpuwrf/contracts/precision.py` | — | none | **necessary** | Pure metadata registry. |
| `src/gpuwrf/profiling/budget.py` | — | none | **necessary** | Pure JSON / regex / `statistics.median`. |
| `src/gpuwrf/profiling/transfer_audit.py` | — | none | **necessary** | Pure trace scanning. |
| `scripts/m3_run_audits.py` | — | none | **necessary** | Calls into the contracts; no direct array constructors. |

Conclusion: worker's `Allocation Audit` in `worker-report.md` is **accurate and complete** — every allocator I found matches the worker's list. No "could be eliminated" or "suspect" entries. Scan body is genuinely allocation-free, and the HLO carry I read confirms it: XLA prunes the unused-prognostic leaves and the body's only array work is the fused `theta` add/multiply/subtract chain (`%fused_subtract`), with the loop counter and predicate as the only other fusions.

## Fixtures Used

- `data/scratch/m2-jax-venv/` — pinned `jax[cuda13]==0.10.0` venv (M2 artifact).
- `artifacts/m3/{transfer_audit.json,spacetime_budget.json,hlo_dump/dummy_loop.txt,agent_success.json,maintainability.md}` — read for schema checks; regenerated once by `scripts/m3_run_audits.py` to confirm idempotency.
- `GridSpec.canary_3km_template()` — used in every test; no external IC/BC files needed at M3.

## Gaps

1. **Pre-existing M2 regression in `pytest -q`** — 11 failures (`test_m2_kokkos`, `test_m2_triton_edge_cases`, `test_m2_cuda_tile`, `test_m2_jax_edge_cases::test_ptxas_reports_no_spills`, `…test_cuobjdump_register_count_matches_profile`, `…test_profile_sanity_bounds_match_contract`). All depend on physical build / profile artifacts (Triton cubin cache, Kokkos `cmake --build`, CUDA tile `nvcc`, cuobjdump output, dirty M2 profile JSONs in the worktree). None of the failing tests touch M3 code paths or M3 artifacts. They reproduced for me on a clean shell, but worker reported 250 pass — almost certainly a hot build cache on worker's side that has since lapsed. This blocks `check_m1_done.py`, `check_m2_done.py`, and therefore `check_m3_done.py`. Recommend: **manager-owned environment refresh** (rebuild Triton cubin / Kokkos / CUDA tile) before milestone closeout; outside this sprint's role scope.
2. **`check_m3_done.py` lifecycle blockers** — reviewer-report.md (131 B), tester-report.md (was 91 B before this update), manager-closeout.md (114 B) all stubs at the time worker handed off; missing `.agent/decisions/MILESTONE-M3-CLOSEOUT.md`; reviewer Decision not yet `Accepted`. This report unblocks the tester slot; reviewer / manager remain.
3. **Transfer audit uses `jax.profiler.trace`, not CUPTI.** Worker's choice is documented in `maintainability.md` (workstation perfmon permission blocked, per project memory). The byte counts being zero is consistent with what CUPTI would report for an `XLA_PYTHON_CLIENT_PREALLOCATE=false` jitted scan that takes pre-staged inputs and returns leaves still on device. I treat this as **adequate evidence at M3**; a CUPTI cross-check belongs in a later sprint if/when perfmon is opened. Not a blocker.
4. **`kernel_launches_per_step = 3`** — strictly within the AC §6 cap of ≤ 5. The 3 fusions are: counter increment, predicate, and the fused theta `broadcast / mul / mul / add / sub` over `(10, 8, 8)`. Already ≤ 5 with margin; no action.
5. **`wall_time_per_step_us ≈ 2.82`** — well under the 100 µs soft bound at `(nz=10, ny=8, nx=8)`. Sanity OK; not a performance claim.
6. **Tests touching `data/scratch/m3/transfer_trace/`** — I did not assert anything about the on-disk trace files. The auditor scans them and reports byte totals; that's what matters at M3.
7. **HLO carry pruning** — XLA pruned the inactive prognostic carry leaves so the while-body operates only on `(counter, theta, dtheta, dt)`. This is correct optimisation but means the loop does not prove the *other* fields actually round-trip through the scan unchanged. The shape/dtype assertions in `test_1000_step_dummy_loop_preserves_shape_dtype` and my new `test_run_dummy_loop_zero_steps_is_state_passthrough` cover this at the API level; a deeper "every field bit-identical after the loop with zero tendencies" check would be a useful **next-sprint** addition once real physics starts touching more fields.

## Decision

**Decision: Accept-with-noted-environment-debt.**

The M3 worker patch satisfies every code-level acceptance criterion I can verify from this branch:

- All 58 M3-specific tests pass on a clean shell (13 worker + 45 new tester adversarial tests).
- Transfer audit is genuinely zero post-init (reproduced; trace files inspected).
- Spacetime budget is internally consistent and self-consistent with `State.bytes()` / `Tendencies.bytes()`.
- HLO dump shows a single fused `while` loop with the documented body shape; `n_steps` is static; loop is one `jax.jit`.
- Hot path is allocation-free (static guard added).
- Init-time allocations are bounded and traceable to frozen state / template leaves.
- GridSpec and HaloSpec invariants reject every adversarial input I threw at them.
- ADR-002 draft contains all required tokens and is well over the 1500-byte floor.
- `State.replace` is immutable; pytree round-trips for `State` and `Tendencies` preserve type and shape.

The only failures I observed (`check_m{1,2,3}_done.py` non-ok and 11 M2 pytest failures) are environment-state issues outside this sprint's worker-owned files: Triton/Kokkos/CUDA-tile build artifacts and dirty M2 profile JSONs that the worker explicitly noted touching when running `scripts/m2_run_jax.sh`. These are **manager-owned** to refresh before milestone closeout — I do not consider them M3-S1 reviewer-blockers.

Recommend the reviewer proceed with the Allocation Audit / per-line attestation (AC §8.5) on `src/gpuwrf/timestep/dummy_loop.py` and `src/gpuwrf/contracts/state.py`; my read of both files finds **zero** simplification opportunities — the dummy loop is already a single fused step and the State is a flat SoA pytree with one private allocator helper.
