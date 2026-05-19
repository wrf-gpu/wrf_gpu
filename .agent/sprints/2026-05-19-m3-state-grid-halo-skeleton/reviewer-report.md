# Reviewer Report

Sprint: `2026-05-19-m3-state-grid-halo-skeleton`
Role: reviewer
Branch: `reviewer/opus/m3-state-grid-halo-skeleton`

## Findings

### Blocker — transfer audit undercounts real H2D copies

`artifacts/m3/transfer_audit.json:4` claims `host_to_device_bytes_post_init == 0`, but the recorded trace under `data/scratch/m3/transfer_trace` contains post-init `MemcpyH2D` events. My parser of the trace's `memcpy_details` fields found 16 H2D bytes, including `size:8` events. The audit code in `src/gpuwrf/profiling/transfer_audit.py:63-77` only counts sizes from top-level `args` keys such as `bytes` or `size`; it does not parse `args["memcpy_details"]`, so `MemcpyH2D ... size:8` is counted as zero. This invalidates AC 5.1/5.2 and the M3 hard rule of zero host/device transfers inside the audited loop. The likely source is `dt` remaining a dynamic Python scalar in `src/gpuwrf/timestep/dummy_loop.py:24-32`, but the required fix is to make the audit parse actual trace sizes first, then remove the transfer.

### Blocker — GridSpec cannot construct in a clean import

`GridSpec.canary_3km_template()` in `src/gpuwrf/contracts/grid.py:164-188` requests fp64 arrays, but `grid.py` never enables `jax_enable_x64`. In a clean process importing only `gpuwrf.contracts.grid`, JAX truncates `jnp.linspace`/`jnp.zeros` to fp32 and `GridSpec.__post_init__` raises `TypeError` at `src/gpuwrf/contracts/grid.py:94-95`. `pytest -q tests/test_m3_grid.py` fails 3/3 in isolation. The full suite passed only because other modules set global JAX config before these tests ran. This violates AC 1.2/1.3 and the fp64 default required by the contract.

### Major — GridSpec is not safe as a JIT static argument

AC 1.4 requires `GridSpec` to be hashable for `@jit` static use. `src/gpuwrf/contracts/grid.py:63-75` uses a frozen dataclass with JAX array fields, and `src/gpuwrf/contracts/grid.py:146-162` overrides `__hash__` but not `__eq__`. Two independently built equivalent `GridSpec` objects hash the same, but comparing them raises `ValueError: The truth value of an array with more than one element is ambiguous`. A direct `@jax.jit(static_argnames=("grid",))` spot-check succeeds on the first grid and fails on the second when JAX compares static cache keys. Implement an array-aware `__eq__` or keep arrays out of static-comparable identity.

### Major — kernel launch budget helper masks future violations

`src/gpuwrf/profiling/budget.py:46-52` clamps the launch estimate to at most 5, and `tests/test_m3_edge_cases.py:417-420` explicitly blesses that behavior. Since the contract requires the HLO-derived `kernel_launches_per_step` to be `<= 5`, clamping makes the proof object unable to fail if a later HLO has 20 fusions. The current HLO independently counted as 3 fusions, so this is not the immediate M3 failure, but the proof machinery needs to report the actual count.

## Contract Compliance

Pass: file ownership in the committed diff is within the sprint-owned M3 files plus reports; `State`/`Tendencies` shapes and fp64 behavior pass when x64 is configured; `apply_halo(state, halo) -> state` has the requested call shape; M3-specific tests pass as a group; artifacts and ADR-002 exist and meet size/token checks.

Fail: AC 5.2 transfer bytes are not proven zero and are contradicted by the trace; AC 1.3 fails in a clean `GridSpec` import; AC 1.4 static-argument comparability fails.

Blocked lifecycle: `python scripts/check_m2_done.py` is still non-ok due to the pre-existing M2 tester-provenance gate, and `python scripts/check_m3_done.py` remains non-ok because reviewer/manager closeout artifacts are not complete. Those lifecycle items are manager-owned, but the two M3 blockers above are implementation/proof blockers.

## Correctness Risks

The current test ordering hides the clean-import `GridSpec` failure, so downstream users can hit fp32 truncation or construction failure depending on import order. The static-argument equality issue will also surface once M4 dycore code starts passing `GridSpec` as a frozen compile-time contract, exactly the use AC 1.4 anticipated.

## Performance Risks

The transfer audit is currently a false-negative risk: the trace shows H2D events that the JSON proof object reports as zero. The launch-count proof is also too forgiving because it clamps violations to the passing threshold.

## Required Fixes

1. Enable x64 before `grid.py` creates or validates fp64 arrays, and add an isolated `pytest -q tests/test_m3_grid.py` regression check.
2. Make `GridSpec` static-argument comparable, not just hashable; add a JIT cache-key test using two independently constructed equivalent grids.
3. Parse `memcpy_details` sizes in the transfer audit, regenerate `artifacts/m3/transfer_audit.json`, and remove the actual post-init H2D transfer so the parsed trace total is zero.
4. Report the raw HLO-derived launch count without clamping to the acceptance threshold.

## Validation Commands

- `python -m gpuwrf.contracts.state --self-test` -> pass, `state_bytes=38656`, `tendency_bytes=38656`, GPU visible.
- `pytest -q tests/test_m3_*.py` -> pass, `58 passed`.
- Read-only artifact/HLO spot-check -> pass for schema and one `while`, but it used the existing JSON and did not validate trace byte parsing.
- `python scripts/check_m1_done.py` -> pass.
- `python scripts/check_m2_done.py` -> fail, M2 tester provenance gate.
- `python scripts/check_m3_done.py` -> fail, M2 gate plus lifecycle closeout stubs.
- `pytest -q` -> pass, `295 passed in 58.92s`.
- `pytest -q tests/test_m3_grid.py` in a clean process -> fail, 3 failures from fp64 config.
- Trace parser spot-check -> `count_transfer_bytes=(0, 0)` but parsed `memcpy_details` totals are H2D=16, D2H=0.

## Decision

Decision: Reject
