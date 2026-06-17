# v0.18 Schema Harvest Report

Branch: `worker/gpt/v018-schemes` from `worker/opus/v018-trunk` at `fbec3544`.

Objective: validate and group the already-green Phase-1 schemes requested for the
v0.18 schema harvest, with per-scheme oracle/smoke/gate evidence and no MP24/MP26
harvest work added in this branch.

## Result

The trunk already contained the target scheme implementations. This branch adds
one correctness fix exposed by the coupled gate and records the grouped proof
objects.

| Scheme | Status | Oracle / wiring | GPU smoke | Coupled real-case gate |
| --- | --- | --- | --- | --- |
| `mp_physics=13` SBU-YLin | PASS | `tests/test_sbu_ylin_savepoint_parity.py` | PASS | PASS: `proofs/v016/coverage/mp13_gate.json`, 637.769 s, 3188.85 ms/step, 7.317 GiB, worst dyn/limit 0.0735 |
| `mp_physics=28` Thompson aerosol-aware | PASS | `tests/test_v016_thompson_aero_oracle.py` | PASS | PASS: `proofs/v016/coverage/mp28_gate.json`, 543.536 s, 2717.68 ms/step, 7.339 GiB, worst dyn/limit 0.0335 |
| `mp_physics=97` Goddard-GCE | PASS | `tests/test_goddard_savepoint_parity.py` | PASS | PASS: `proofs/v016/coverage/mp97_gate.json`, 453.302 s, 2266.51 ms/step, 7.081 GiB, worst dyn/limit 0.0948 |
| `ra_lw_physics=31` Held-Suarez | PASS | `tests/test_v017_ra_lw_hs.py` | PASS | PASS: `proofs/v016/coverage/lw31_gate.json`, with `ra_sw_physics=0`, 426.331 s, 2131.66 ms/step, 3.487 GiB, worst dyn/limit 0.0256 |
| `sf_surface_physics=1` slab LSM | PASS | `proofs/v013/t3_surface_lsm_oracle.json` + dispatch test | PASS | Not run by generic L2 sweep: operational path requires explicit `SlabStaticBundle`; see risk below |
| `sf_surface_physics=7` Pleim-Xiu LSM | PASS | `proofs/v017/pxlsm_savepoint_parity_report.json` + parity test | PASS | Not run by generic L2 sweep: operational path requires explicit `PleimXiuStaticBundle`; see risk below |

All produced coupled gate JSONs have `verdict: PASS`, `all_finite: true`, empty
`bounds_violations`, empty `hard_gate_fails`, and empty `review_flags`.

## Fixes

`thompson_aero_coldstart_init` no longer branches through `float(jnp.max(...))`.
The previous host scalar conversion failed when the operational coupled forecast
staged MP28 cold-start under JIT. The replacement uses a scalar JAX predicate and
`jnp.where`, preserving restart-provided aerosol fields without host transfer.

`proofs/v016/coupled_coverage_gate.py --family lw --option 31` now sets
`ra_sw_physics=0`, matching the operational fail-closed rule for Held-Suarez as a
combined idealized LW+SW scheme.

## Commands Run

CPU oracle/wiring group:

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true pytest -q tests/test_v016_thompson_aero_oracle.py tests/test_goddard_savepoint_parity.py tests/test_sbu_ylin_savepoint_parity.py tests/test_v017_ra_lw_hs.py tests/test_v017_lsm_pleim_xiu.py tests/test_v060_physics_dispatch.py::test_slab_lsm_is_operational_and_gpu_gate_ready tests/test_v013_operational_smoke.py::test_sf_surface_slab_operational_runs_and_advances_land
```

Result: `61 passed in 9.14s`.

Focused MP28 JIT regression:

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true pytest -q tests/test_v016_thompson_aero_threading.py::test_coldstart_matches_frozen_numpy_climatology tests/test_v016_thompson_aero_threading.py::test_coldstart_is_jittable_and_preserves_seeded_aerosols tests/test_v016_thompson_aero_oracle.py
```

Result: `5 passed in 4.85s`.

GPU smoke group:

```bash
scripts/with_gpu_lock.sh --label gpt-schemes -- bash -lc 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader && taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=/mnt/data/gpuwrf_jax_cache pytest -q "tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[13]" "tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[28]" "tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates[97]" "tests/test_v013_operational_smoke.py::test_ra_lw_operational_runs_and_cools[31]" "tests/test_v013_operational_smoke.py::test_sf_surface_slab_operational_runs_and_advances_land" "tests/test_v013_operational_smoke.py::test_sf_surface_pleim_xiu_operational_runs_and_advances_land"'
```

Result: `6 passed in 7.71s` on `NVIDIA GeForce RTX 5090`.

Coupled gates:

```bash
scripts/with_gpu_lock.sh --label gpt-schemes -- bash -lc 'taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=/mnt/data/gpuwrf_jax_cache python proofs/v016/coupled_coverage_gate.py --family mp --option 13 --hours 1 --refresh-baseline'
scripts/with_gpu_lock.sh --label gpt-schemes -- bash -lc 'taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=/mnt/data/gpuwrf_jax_cache python proofs/v016/coupled_coverage_gate.py --family mp --option 28 --hours 1'
scripts/with_gpu_lock.sh --label gpt-schemes -- bash -lc 'taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=/mnt/data/gpuwrf_jax_cache python proofs/v016/coupled_coverage_gate.py --family mp --option 97 --hours 1'
scripts/with_gpu_lock.sh --label gpt-schemes -- bash -lc 'taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=/mnt/data/gpuwrf_jax_cache python proofs/v016/coupled_coverage_gate.py --family lw --option 31 --hours 1'
```

Results: `mp13`, `mp28`, `mp97`, and `lw31` all PASS.

Coverage bookkeeping:

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python proofs/v016/coverage_map.py
taskset -c 0-3 env OMP_NUM_THREADS=4 PYTHONPATH=src JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python proofs/v016/coverage/rollup.py
```

Result: v0.16/v0.18 rollup is `IN_PROGRESS` overall because trunk has
out-of-scope pending targets `mp24`, `mp26`, and `pbl3`; the requested harvest
targets `mp13`, `mp28`, `mp97`, and `lw31` are green.

## Unresolved Risks

The generic coupled L2 sweep cannot currently run slab/Pleim-Xiu by family/option
alone because the operational scan correctly requires WRF-derived static bundles.
Both LSM schemes passed their pristine-WRF oracle and GPU operational smoke tests,
but this branch does not add real-case static extraction for `lsm1`/`lsm7`; that
work is handed off to the Opus LSM worker and does not block the PBL-family
batch. The follow-on boundedness check is recorded in
`proofs/v018/phase2_survey_report.md`; the available real-case wrfinput lacks the
slab/PX static fields needed for an honest generic `SlabStaticBundle` /
`PleimXiuStaticBundle` extraction.

`worker/opus/v018-trunk` already includes MP24/MP26 schema rows. They remain
visible in the recomputed coverage/rollup artifacts, but were not harvested or
validated here because the contract explicitly said to skip them.

## Phase-2 Continuation

This branch also ports `bl_pbl_physics=11` Shin-Hong and `bl_pbl_physics=12`
GBM in the PBL-family batch. Both are now operational+oracle:

- PBL11 Shin-Hong: dynamics-green vs the v090 WRF-backed host reference,
  GPU-smoke PASS, coupled gate PASS/finite. TKE_PBL/EL_PBL remain an honest
  non-driving diagnostic caveat vs the PARTIAL/fp32-sensitive reference
  (TKE rel ~=0.285, EL rel ~=0.013); no tolerance was widened.
- PBL12 GBM: fp64 pristine-WRF module oracle PASS, GPU-smoke PASS, coupled gate
  PASS/finite.

The rest of the PBL family is classed without silent gaps: PBL4 QNSE, PBL10
TEMF, PBL16 EEPS, and PBL17 KEPS are accepted reference-only with real
pristine-WRF module oracles and operational fail-close; PBL9 CAM-UW is proven
irrelevant to the standalone v0.18 PBL matrix from WRF source evidence and
remains recognized fail-closed. `pbl_family_ship_gate=true` and scoped
`full_ship_gate=true` for this PBL worker. RRTMG14/24 is radiation scope and is
handed to the RA worker in `/tmp/v018_rrtmg1424_handoff.md`.
