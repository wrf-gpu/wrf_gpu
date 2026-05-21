# M6-S1 Worker Report — Coupled Interface and Precision Boundary Freeze

## objective

Freeze the first M6 coupled interface so M6-S2..S8 can run file-disjoint work: extend the ADR-002 SoA state pytree, encode ADR-007 precision boundaries, add adapters around the existing M5 physics kernels without modifying them, prove a 100-step coupled dummy carry with zero post-init transfers, and document the downstream ownership split in ADR-010.

## files changed

- `src/gpuwrf/contracts/state.py`
  - Extended `State` with separate SoA leaves for hydrometeors, number concentrations, MYNN `qke`, surface handles, and precipitation accumulators.
  - Added units and staggering documentation in the `State` docstring.
  - Added `_state_field_shapes(grid)` for the frozen M6 shape contract.
  - Kept `Tendencies` limited to the existing dycore fields; M6-S2 owns real coupled tendency routing.
  - Updated `State.replace` to preserve the existing field dtype when callers provide promoted expressions, which keeps dycore-only scans type-stable under mixed boundaries.
- `src/gpuwrf/contracts/precision.py`
  - Added `FP32_GATED`, `STATE_FIELD_ORDER`, and `PRECISION_MATRIX`.
  - Kept locked FP64 rows for `mu`, `p`, `ph/pgeop`, `w`, surface-stability handles, and precipitation accumulators.
  - Marked `u/v/theta/qv`, hydrometeors, number concentrations, and `qke` as gate-required FP32 rows.
- `src/gpuwrf/coupling/__init__.py`
- `src/gpuwrf/coupling/physics_couplers.py`
  - Added `thompson_adapter`, `mynn_adapter`, `rrtmg_adapter`, and `surface_adapter`.
  - All adapters slice SoA state into transient column-batched views, call existing physics kernels unchanged, and reassemble `State`.
  - The adapters cast updated fields back to the precision registry at the coupling boundary.
- `scripts/m6_run_dummy_coupled.py`
  - Builds the 16x16x30 dummy grid/state from `State.zeros`.
  - Runs a single jitted 100-step carry with dycore, Thompson, MYNN, surface, and RRTMG every tenth step.
  - Writes `artifacts/m6/coupled_dummy_carry.json` and `artifacts/m6/spacetime_budget.json`.
- `tests/test_m6_state_extension.py`
- `tests/test_m6_precision_matrix.py`
- `tests/test_m6_dummy_coupled.py`
- `.agent/decisions/ADR-010-coupled-state-extension.md`
- `src/gpuwrf/debug/snapshots.py`
  - Minimal compatibility fix: debug snapshots now record all current `State` leaves instead of assuming the old eight-leaf M4 state.
- `artifacts/m6/coupled_dummy_carry.json` and `artifacts/m6/spacetime_budget.json`
  - These are ignored by `artifacts/*`; they must be force-added with the commit.

## AC status

AC1 — PASS. `State.zeros(grid)` now allocates every required M6 leaf as its own JAX array. New 3D fields use `(nz, ny, nx)` and surface/accumulator fields use `(ny, nx)`. `tests/test_m6_state_extension.py` verifies presence, device residency, shape, and dtype.

AC2 — PASS. `PRECISION_MATRIX` is the machine-readable ADR-007 boundary registry. `tests/test_m6_precision_matrix.py` verifies every `State` field is covered and that gated vs locked rows match the intended classes.

AC3 — PASS. `src/gpuwrf/coupling/physics_couplers.py` defines all four adapter functions. The Thompson, MYNN, RRTMG SW/LW, and surface stub kernels are called unchanged. The only adapter-local assumptions are transient axis movement and the M6-S1 dummy `DEFAULT_DZ_M = 100.0` because the contracted adapter signature does not carry `GridSpec`.

AC4 — PASS. `python scripts/m6_run_dummy_coupled.py` generated `artifacts/m6/coupled_dummy_carry.json` for a 100-step 16x16x30 dummy carry. Transfer audit fields are hard zero:

- `host_to_device_bytes_post_init = 0`
- `device_to_host_bytes_post_init = 0`
- `temporary_bytes_per_step = 0`

The committed script avoids a device-side radiation-cadence predicate. An earlier local attempt used `lax.cond((step + 1) % 10 == 0, ...)` and the profiler caught repeated 1-byte D2H predicate transfers. The final script uses static nested 10-step scans instead.

AC5 — PASS. `artifacts/m6/spacetime_budget.json` contains per-kernel wall time, HLO bytes, and raw HLO-derived launches:

- dycore: 24 launches, 816323 HLO bytes, 0.30562200117856264 ms
- Thompson: 7 launches, 508811 HLO bytes, 0.4818330053240061 ms
- MYNN: 32 launches, 354926 HLO bytes, 0.7731560617685318 ms
- surface: 1 launch, 30290 HLO bytes, 0.18486205954104662 ms
- RRTMG: 170 launches, 1097136 HLO bytes, 4.346429952420294 ms, cadence 10 steps
- total coupled carry: 0.646770759485662 ms per step

No `min(raw, cap)` launch clamp is used; values come from `kernel_launches_per_step` over compiled HLO text.

AC6 — PASS. Added `.agent/decisions/ADR-010-coupled-state-extension.md`, cross-referencing ADR-002 and ADR-007 and pointing at the M6 proof objects.

AC7 — PASS. ADR-010 declares file ownership for M6-S2 through M6-S8, including forecast driver, surface/Noah-MP, Tier-2/Tier-3/Tier-4 validation, performance verdict, and operational comparison paths.

## commands run

- `pytest -q tests/test_m6_state_extension.py tests/test_m6_precision_matrix.py` -> `4 passed`
- `pytest -q tests/test_m6_dummy_coupled.py` -> initially failed on a shape broadcast bug and dtype promotion; fixed and reran -> `2 passed`
- `pytest -q tests/test_m6_*.py` -> `6 passed`
- `python scripts/m6_run_dummy_coupled.py` -> initially failed transfer audit with `h2d=0 d2h=30920`; root cause was the dynamic radiation-cadence predicate. After static nested-scan fix -> success with zero H2D and zero D2H.
- `pytest -q tests/test_m4_dycore_step.py` -> initially exposed mixed-dtype dycore scan instability after the State extension; fixed `State.replace` and debug snapshots -> `3 passed`
- `python scripts/validate_agentos.py` -> `{"ok": true, "errors": [], "required_files_checked": 31, "skills_checked": 13}`
- `cat artifacts/m6/coupled_dummy_carry.json`
- `cat artifacts/m6/spacetime_budget.json`

## proof objects produced

- `artifacts/m6/coupled_dummy_carry.json`
  - domain `[16, 16, 30]`
  - steps `100`
  - wall time per step `0.646770759485662 ms`
  - kernel launches per step `320`
  - HLO bytes `5193670`
  - H2D post-init `0`
  - D2H post-init `0`
  - temporary bytes per step `0`
- `artifacts/m6/spacetime_budget.json`
  - per-kernel budget table
  - total per-step wall time
  - host-device transfer bytes `0`
- Tests:
  - `tests/test_m6_state_extension.py`
  - `tests/test_m6_precision_matrix.py`
  - `tests/test_m6_dummy_coupled.py`
- ADR:
  - `.agent/decisions/ADR-010-coupled-state-extension.md`

## unresolved risks

- This sprint proves interface residency and dtype/shape stability, not forecast correctness. Conservation budgets, operational drift, and RMSE gates remain M6-S4/S6/S8 work.
- RRTMG is wrapped unchanged. M5-S3.y transfer-solver parity debt remains outside this sprint.
- The adapter signature is fixed as `(state, dt)`, so MYNN/RRTMG use `DEFAULT_DZ_M = 100.0` in M6-S1. M6-S2 must decide how real grid metrics enter the coupled forecast path without rewriting the frozen State layout.
- Surface fields are placeholders. M6-S3 must widen or replace the surface-layer protocol for real Monin-Obukhov/Noah-MP inputs such as roughness lengths, skin temperature provenance, and soil state.
- The trace raw files under `artifacts/m6/trace_dummy_coupled/` are ignored and not listed as deliverables. The committed JSON records the trace path and zero byte counts; reviewer can regenerate with the script if raw trace inspection is required.

## next decision needed

Mandatory fresh-context Claude Opus 4.7 reviewer pass per `.agent/rules/sprint-lifecycle.md`. If accepted, M6-S2 can take ownership of the coupled forecast driver while M6-S3..S8 start from ADR-010's file-disjoint ownership freeze.
