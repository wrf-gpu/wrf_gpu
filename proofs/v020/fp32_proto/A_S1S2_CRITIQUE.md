# Sprint-A S1/S2 Critique

Codex critic of Claude Sprint-A S1/S2 at commit `0ea4414b`
(`worker/claude/v020-s1s2-compaction`, worktree
`<USER_HOME>/src/wrf_gpu2_wt/v020-s1s2`). CPU-only review.

## Verdict

**APPROVE - merge-ready.**

I found no blocking correctness, bit-identity, dtype, or liveness issues in S1/S2.
The only notable gaps are coverage framing issues, not merge blockers:

- The `harness.py compare` path proves the 6-step fp64-default hot loop, but it is
  not itself a restart round-trip test. Restart round-trip coverage comes from the
  separate restart tests listed below.
- The `harness.py nested` path proves the fused nested cascade traces and runs on
  CPU with finite output. It is a trace/run coverage gate, not an output byte-diff
  gate for nested multilevel production fixtures.

## Verification Summary

### 1. fp64_default bit identity

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/v020-s1s2/src \
  python proofs/v020/s1s2/harness.py compare --steps 6
```

Result:

- `71/71 leaves BITWISE-EQUAL (6 steps)`
- Removed legacy duplicate leaves reported as `state.mu`, `state.p`, `state.ph`.

I also probed the EOS direct-total path in `rk_addtend_dry.py:168` by
instrumenting `_absolute_diagnostics` and `_inverse_density_from_theta_pressure`
during one `_advance_chunk`; it called the direct `state.p_total` EOS path 3 times.

Nested fused cascade:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/v020-s1s2/src \
  python proofs/v020/s1s2/harness.py nested
```

Result:

- Fused cascade default-enabled: `True`
- Fused cascade lowers to HLO and runs on CPU.
- Own-step schedule: `{'d01': 1, 'd02': 3, 'd03': 9}`
- `finite_after_run=True`

Restart coverage:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/v020-s1s2/src \
  python -m pytest tests/test_m3_state.py tests/test_m6_state_extension.py \
    tests/test_m7_restart_checkpoint_roundtrip.py tests/test_p0_5_restart_full_carry.py \
    tests/test_v0110_wrfrst_netcdf.py -q
```

Result: `15 passed, 4 skipped`.

Conclusion: the bit-identity claim is real for the exercised 6-step hot loop, EOS
direct-total reads are exercised, nested fused cascade traces/runs, and restart
round-trip is covered separately.

### 2. State pytree surgery

Audited the `State` changes and all `state.p` / `state.ph` / `state.mu` usage:

- `State.__slots__` drops only the duplicate legacy aliases, while retaining
  `p_total`, `ph_total`, `mu_total`, and their perturbation leaves.
- Read-only properties `p`, `ph`, and `mu` return `p_total`, `ph_total`, and
  `mu_total`.
- `State.replace(p=...)`, `replace(ph=...)`, and `replace(mu=...)` redirect to
  the authoritative total leaves and preserve the historical perturbation delta
  when only the total changes.
- Production direct assignment to `State.p`, `State.ph`, or `State.mu` was not
  found. The direct assignment hits are duck-typed test fixtures, not
  `gpuwrf.contracts.State`.

Targeted probe result:

- `state.p is state.p_total` is true.
- `state.replace(p=new_total)` updates `p_total`.
- The perturbation delta is preserved as `old_perturbation + (new_total - old_total)`.
- Direct `state.p = ...` raises `AttributeError`.

The remaining legacy writes in production are compatible `replace(...)` calls, for
example `dynamics/acoustic.py`, `runtime/operational_mode.py`, `validation/tier2.py`,
and coupling guards. The `State.replace` redirect handles those call sites.

### 3. S2 fp64-island behavior

`force_fp64_island` behaves as required:

- fp64 input returns by Python identity (`force_fp64_island(x64) is x64`), so the
  helper itself does not emit a conversion for fp64 inputs.
- fp32 input widens to `float64`.
- EOS helper `_inverse_density_from_theta_pressure` returns `float64` from fp32
  inputs.
- `_calc_al_p` returns `float64,float64` from fp32 inputs when called with the
  expected broadcast shapes.

Independent HLO comparison on the same warm-bubble `_advance_chunk` lowering:

| Worktree | carry leaves | active State leaves | classic `convert(` | `stablehlo.convert` |
| --- | ---: | ---: | ---: | ---: |
| clean main `d329ef2e` | 74 | 60 | 0 | 2 |
| Sprint-A `0ea4414b` | 71 | 57 | 0 | 2 |

Conclusion: S2 is a fp64 no-op on this fp64-default hot path and widens fp32
inputs inside the protected brackets.

### 4. dtype-stability gate

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/v020-s1s2/src \
  python -m pytest tests/test_v020_dtype_stability.py -q
```

Result: `8 passed`.

The test file explicitly covers:

- fp64-default `_advance_chunk` dtype stability.
- injected `float64 -> float32` downcast detection.
- injected `float32 -> float64` upcast detection.
- JAX fori-loop carry dtype rejection as a second layer.
- convert-counter direction parsing.

Conclusion: the dtype-stability test genuinely fires both directions.

### 5. Reported unrelated failures

I reran the named failing nodes on Sprint-A and clean `main` (`<USER_HOME>/src/wrf_gpu2`,
`d329ef2e`; tracked files clean, unrelated untracked proof artifacts present).

Memory audit:

```bash
python -m pytest \
  tests/test_m7_1km_memory_audit.py::test_static_model_tracks_state_contract_field_count \
  tests/test_m7_1km_memory_audit.py::test_static_model_uses_precision_registry_for_known_fields -q
```

Results:

- Sprint-A: both fail with `State field contract mismatch:
  STATE_FIELD_ORDER=64 slots=64 shapes=57`.
- main: both fail with `State field contract mismatch:
  STATE_FIELD_ORDER=67 slots=67 shapes=60`.

The exact counts differ because S1 removed three duplicate leaves, but the underlying
contract mismatch is already present on main.

Radiation:

```bash
python -m pytest \
  'tests/test_v013_operational_smoke.py::test_ra_lw_operational_runs_and_cools[1]' \
  tests/test_v013_operational_smoke.py::test_sw_and_lw_are_selected_independently -q
```

Results:

- Sprint-A: both fail with `FileNotFoundError` for
  `.../data/wrf_pristine/WRF/phys/module_ra_rrtm.F`.
- main: both fail with the same missing file under the main worktree.

Conclusion: these failures are pre-existing and not introduced by S1/S2.

### 6. Efficiency and hot-path regression

The 74 -> 71 leaf reduction is real in the executable warm-bubble operational carry:

- main: `carry_leaves=74`, `state_active_count=60`
- Sprint-A: `carry_leaves=71`, `state_active_count=57`

The removed leaves are the legacy duplicate total aliases only. Totals remain stored
in fp64-default, consistent with the D critique contract. The same lowered hot path
has unchanged float-convert counts (`classic_convert=0`, `stablehlo_convert=2`) and
slightly smaller HLO text (`4002061` chars on main, `3999657` on Sprint-A), so I did
not find a hot-path regression.

## Commands Run

- `python proofs/v020/s1s2/harness.py compare --steps 6`
- `python proofs/v020/s1s2/harness.py liveness`
- `python proofs/v020/s1s2/harness.py nested`
- `python -m pytest tests/test_v020_dtype_stability.py -q`
- `python -m pytest tests/test_m3_state.py tests/test_m6_state_extension.py tests/test_m7_restart_checkpoint_roundtrip.py tests/test_p0_5_restart_full_carry.py tests/test_v0110_wrfrst_netcdf.py -q`
- `python -m pytest tests/test_noahmp_coupler.py -q`
- Targeted EOS/direct-total, State alias/replace, S2 fp32-widening, and main-vs-Sprint-A HLO/liveness probes.
- Named failing memory-audit and radiation tests on both Sprint-A and clean main.

All execution used CPU-only environment variables (`CUDA_VISIBLE_DEVICES=''`,
`JAX_PLATFORMS=cpu`, `JAX_PLATFORM_NAME=cpu`, `JAX_ENABLE_X64=1`) and
`PYTHONPATH` pointed at the reviewed worktree source when reviewing Sprint-A.
