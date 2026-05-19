# Tester Report

Sprint: `2026-05-19-m2-kokkos-stencil-column`
Role: tester (Claude Opus 4.7 acting as sonnet-test-engineer, cross-AI verification of the gpt-5.5 worker).
Branch: tester scope on `worker/gpt/m2-kokkos-stencil-column` (no source edits — only `tests/` and this report).

## Tests Added Or Run

### Re-run of contract validation commands (clean shell after `source env_wrf_gpu.sh`)

- `bash src/gpuwrf/backends/kokkos/build.sh` — idempotent rerun: cached Kokkos install
  used, only the bench was relinked. Exit 0, ending `[100%] Built target bench`.
- `cuobjdump --dump-sass data/scratch/kokkos/bench | grep -m1 'arch'` → `arch = sm_120`
  (matches contract AC #3).
- `bash scripts/m2_run_kokkos.sh` — exit 0. Fresh stencil run: `wall_time_s=0.000223531`,
  `kernel_launches=1`, `host_device_transfer_bytes=118272`, `kokkos_execution_space=Cuda`,
  `runtime_compute_capability=12.0`. Fresh column run: `wall_time_s=0.000134431`,
  `kernel_launches=1`, `host_device_transfer_bytes=2560`.
- `compare_fixture --pass=true` against the analytic-stencil and analytic-column manifests
  — both pass at tier 1. `mse_delta` and `qv_next` show only floating-point round-off
  (`max_rel_diff` ≤ 1.8e-16; well inside the manifested tolerances).
- `cuobjdump --dump-resource-usage data/scratch/kokkos/bench` — `StencilAdvdiffKernel`
  reports `REG:64 LOCAL:0`, `ColumnThermoKernel` reports `REG:40 LOCAL:0` (column also
  uses 1032 B `SHARED` for Kokkos TeamPolicy scaffolding, but `local_memory_bytes`
  is the AC and that is zero).
- Both Function symbols embed `Kokkos::Cuda` (mangled `4Cuda`), confirming the handoff
  requirement that View allocations target CudaSpace, not HostSpace/Serial.
- `pytest -q` — **146 passed in 42.04s** (114 pre-existing + 32 new tester tests).
- `python scripts/validate_agentos.py` → ok.
- `python scripts/check_m1_done.py` → ok, 3 sprints closed.
- `python scripts/check_m2_done.py` → exits 1 as expected (3/6 candidates satisfied;
  remaining errors are out-of-scope: JAX/Triton/GT4Py artifacts, ADR-001, and unfinished
  closeout stubs for *this* sprint that downstream roles must fill).

### New edge-case tests (added under tester scope only)

File: `tests/test_m2_kokkos_edge_cases.py` (32 tests, all pass).

Coverage groups:

1. **Profile JSON rigor** — schema keys/types, numeric consistency, bandwidth
   = transfer/wall identity, artifact paths relative+existent, profile vs.
   `*_run.json` agreement, `profiler_limitation` only when `.ncu-rep` is missing.
2. **Contract sanity bounds** — `registers_per_thread` ≤ 64 stencil / ≤ 128 column,
   `kernel_launches` ≤ 5, occupancy ≥ 25 % stencil / ≥ 20 % column, `local_memory_bytes`
   == 0 for *both* kernels, exec space pinned to `Cuda`, CC 12.0, Kokkos version 40701.
3. **Resource-usage cross-check** — re-parses `data/scratch/kokkos/resource_usage.txt`
   and confirms the worker's profile JSON values *come from* that file (catches
   hand-edited fabricated register/local numbers).
4. **CUDA-space allocation evidence** — verifies the mangled functor names for
   `StencilAdvdiffKernel` and `ColumnThermoKernel` contain `Kokkos::Cuda` and do
   *not* fall back to `Serial` (the cross-AI handoff requirement).
5. **Bench malformed-input behavior** — unknown problem, missing `--input`,
   `--input` without value, missing input file, garbage bytes, truncated EOCD-only
   zip, wrong fixture for problem (column→stencil). All must fail loudly with a
   non-empty diagnostic, not silently produce garbage output.
6. **End-to-end determinism** — fresh-tempdir bench run on the analytic fixture
   round-trips through `compare_fixture --pass=true`, and two consecutive bench runs
   produce **bitwise-identical** output bytes (would catch a future warp-shuffle
   atomicAdd nondeterminism regression).
7. **Build pipeline integrity** — `bench config` advertises Cuda + BLACKWELL120 +
   `KOKKOS_ENABLE_CUDA: yes`; `KokkosConfig.cmake` and `nvcc_wrapper` present;
   `build.sh` rerun does not re-clone or re-install (AC #1 idempotency).
8. **Maintainability / agent-success** — `maintainability.md` ≤ 300 words and
   covers all four AC #11 topics; `agent_success.json` schema valid.
9. **Deliberate-bug capture** — `deliberate_bug_stderr.txt` contains a real
   compiler diagnostic with the documented `phi_nxt` undefined-identifier error
   and a non-zero exit code (maintainability evidence is real, not fabricated).

## Results

All contract acceptance criteria verified:

- AC #1 (idempotent build.sh) — verified by `test_build_is_idempotent_on_rerun`.
- AC #2 (bench exits cleanly with no args, prints usage) — verified.
- AC #3 (`sm_120` SASS or PTX-fallback runtime CC) — verified; both paths covered.
- AC #4, #5 (stencil/column fixture round-trip pass) — verified end-to-end on
  fresh-tempdir output, not just the worker's cached artifact.
- AC #6 (profile JSON schema) — schema-key set audited; both profiles validate.
- AC #7 (`kernel_launches` ≤ 5) — both = 1.
- AC #8 (`local_memory_bytes` = 0 on column) — verified for *both* kernels.
- AC #9 (`registers_per_thread` ≤ 64 stencil, ≤ 128 column) — observed 64 / 40.
  Stencil sits *exactly at the limit*; the worker flagged this as a regression
  risk and I confirm it: any future kernel edit that bumps register count by 1
  will silently violate the contract unless this test fires.
- AC #10 (numbers derived from cuobjdump + bench output; ncu fallback documented)
  — verified, including the `profiler_limitation` text matching the
  `ERR_NVGPUCTRPERM` known limitation.
- AC #11 (maintainability narrative ≤ 300 words, all four topics) — verified.
- AC #12 (`agent_success.json` populated) — verified, schema-checked.
- AC #13, #14 (tests + `pytest -q` green) — 146 passed.
- AC #15–18 (hygiene, gitignored caches) — `validate_agentos.py` ok, no new
  >100 KB files committed, `data/scratch/kokkos-{src,install}/` are under the
  gitignored `data/` symlink.

## Fixtures Used

- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (manifest:
  `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml`) — M1-S2 deliverable,
  unchanged on main.
- `fixtures/samples/analytic-column-thermo-v1.npz` (manifest:
  `fixtures/manifests/analytic-column-thermo-v1.yaml`) — M1-S3 deliverable,
  unchanged on main.
- Synthetic malformed inputs created in pytest `tmp_path` for the negative tests
  (garbage bytes, truncated EOCD-only zip, wrong-fixture-for-problem).

No new fixtures introduced. No binary data added to git (only `tests/` and
this report touched in tester scope).

## Gaps

These are explicitly out of tester scope but worth recording so the reviewer
and the M2 closeout/ADR-001 author can take them into account:

1. **`ncu` performance counters remain blocked by `ERR_NVGPUCTRPERM`** on this
   workstation across all M2 candidates so far. Numbers are derived correctly
   from `cuobjdump` + Kokkos runtime + bench wall-time, matching the manager-
   approved M2 fallback pattern, but ADR-001 must compare candidates knowing
   that none of the M2-S2/S3/S4 rows have raw hardware counters.
2. **Stencil registers = 64 is at the AC ceiling.** Headroom = 0. Recorded as a
   surfaced risk in the worker report; the edge-case suite will catch a future
   regression by failing `test_profile_sanity_bounds_match_contract`.
3. **Column kernel uses Kokkos TeamPolicy `SHARED:1032` bytes** for runtime
   scaffolding. The AC asks about `local_memory_bytes` (zero), not shared memory,
   so this is acceptance-compliant. ADR-001 should note the asymmetry: cuda_tile
   declares the manual tile in shared explicitly, Kokkos pays the cost implicitly
   for TeamPolicy launch overhead.
4. **Kokkos TeamPolicy uses one team of 64 threads** for a 40-level column
   (24 idle threads per launch). For the bakeoff this is fine — the contract
   said "idiomatic MDRangePolicy / team_policy patterns first". Whoever writes
   the maintainability comparison in ADR-001 should note that this is exactly
   the kind of decision a real port would revisit.
5. **`achieved_bandwidth_gbps` ≈ 0.5 GB/s** on the stencil. This is fallback-
   derived from host↔device transfer bytes ÷ wall-time and **does not represent
   on-GPU streaming bandwidth** (these problems are kernel-launch-overhead
   bound, not bandwidth bound, on 32×16×8 and 40-level test sizes). All M2
   candidates inherit this caveat — ADR-001 cannot rank candidates on this field
   alone.
6. **Out-of-scope working-tree noise.** `git status` at sprint entry showed
   `artifacts/m2/{cuda_tile,cupy_or_numba}/*_profile.json` as modified. These
   are *not* in worker file ownership for this sprint, but they were regenerated
   as a benign side effect of `pytest -q`: `tests/test_m2_cupy.py` and
   `tests/test_m2_cuda_tile.py` invoke `scripts/m2_run_{cupy,cuda_tile}.sh`,
   which rewrite their respective profile JSONs with fresh timings. Not a
   worker scope violation, but worth flagging to the manager: those test files
   should arguably stop overwriting committed artifacts on every pytest run.
7. **No multi-device, no mixed precision, no UVM exercised** — explicitly out
   of scope per the contract Non-Goals; flagged here so ADR-001 doesn't credit
   Kokkos for portability evidence that the M2-S4 sprint did not produce.

## Decision

**Decision: Accept (no fixes required).**

Rationale: every numbered acceptance criterion in the sprint contract is
independently verified by re-running the worker's commands from a clean shell
and by the 32-test edge-case suite added under `tests/test_m2_kokkos_edge_cases.py`.
The Kokkos build is reproducible, the bench targets `sm_120`, View allocations
live in `Kokkos::CudaSpace` (per the mangled symbols in `resource_usage.txt`),
both kernels round-trip the analytic fixtures at tier 1, both kernels report
`local_memory_bytes = 0`, and the JSON numerics are internally consistent
(bandwidth = transfer ÷ wall; profile values match `*_run.json`). The deliberate-
bug evidence is real (compiler diagnostic with non-zero exit code) and the
maintainability narrative covers all four contract topics within the 300-word
budget. The known `ERR_NVGPUCTRPERM` limitation is honestly recorded with the
manager-approved M2 fallback text. Stencil registers at the 64-thread limit and
the cuda_tile/cupy artifact noise are *risks/observations*, not defects, and
the new edge-case tests guard against regressions on the former. ADR-001 has
the kokkos row it needs.
