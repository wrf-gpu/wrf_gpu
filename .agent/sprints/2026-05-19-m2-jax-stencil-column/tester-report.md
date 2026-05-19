# Tester Report

Sprint: `2026-05-19-m2-jax-stencil-column`
Role: tester / Claude Opus 4.7 `xhigh` (cross-AI verification of the
gpt-5.5 worker output)
Branch: `tester/sonnet/m2-jax-stencil-column`

## Tests Added Or Run

- New file `tests/test_m2_jax_edge_cases.py` — 39 parametrized cases.
  Coverage (organised by intent):
  1. **Profile JSON schema rigor.** Required-key set, value types,
     `backend == "jax"`, `hardware == "RTX 5090 32GB"`, pinned
     `jax_version == "0.10.0"`, `jax_backend == "gpu"`, CUDA device in
     `jax_devices`, `achieved_bandwidth_method == "fallback-derived"`,
     `warmup_pattern` documents compile exclusion + multi-run statistic,
     `wall_time_s < 0.5s` (compile is ~120 ms — would dominate if
     leaking), bandwidth = transfer / wall, relative artifact paths
     resolve on disk, `run.json ↔ profile.json` numeric agreement,
     `profiler_limitation` present iff the `.ncu-rep` is absent/empty.
  2. **Contract sanity bounds.** `kernel_launches ∈ [1, 5]`,
     `registers_per_thread ≤ 64` (stencil) / `≤ 128` (column),
     `occupancy_pct ≥ 25` / `≥ 20`. `local_memory_bytes == 0` for
     **both** the stencil and the column kernels — the contract's
     load-bearing ADR-001 signal.
  3. **Independent XLA-dump cross-checks** (not relying on the bench
     self-report). Parses
     `data/profiler_artifacts/jax/{stencil,column}_cuobjdump_resource_usage.txt`
     and asserts `0 bytes spill stores / 0 bytes spill loads` and
     `LOCAL:0`. Re-parses `REG:N` and demands equality with the profile
     JSON. Reads the captured `thunk_sequence.txt` and confirms exactly
     one `kKernel` thunk and no `kCustomCall` / `kAllReduce` /
     `kCollectivePermute` / `kAllToAll`. Grep on
     `{stencil,column}_compiled_hlo.txt` requires exactly one `fusion(`
     site, no `all-reduce`, no `collective-permute`, and an `f64`
     entry-computation return type (no covert mixed precision).
  4. **Backend witness.** `data/scratch/m2-jax/jax_backend.json` claims
     `default_backend == "gpu"`, `jax_version == "0.10.0"`, a CUDA
     device, and that `XLA_FLAGS` carried `--xla_dump_to=` so the dump
     tree is the run-time output (not a checked-in stub).
  5. **Maintainability / agent_success / contract negatives.** Word
     budget ≤ 300, topics covered, and `maintainability.md` mentions
     neither `pallas` nor `mixed precision` (contract Non-Goals).
     `agent_success.json` schema check.
  6. **Bench CLI behaviour** (invoked inside the worker's
     `data/scratch/m2-jax-venv/`). Rejects unknown `--problem`, missing
     `--stencil-fixture`, malformed `.npz`, and the column fixture
     handed to the stencil problem (wrong-key failure). End-to-end
     stencil and column reproduction against the reference fixtures.
     Bitwise reproducibility across back-to-back stencil runs.
  7. **Venv idempotency / pin enforcement.** `python -c "import jax"`
     in the venv prints `0.10.0` / `gpu`. `pip_freeze.txt` pins
     `jax==0.10.0`, `jaxlib==0.10.0`, and contains no `triton` extras.

- Re-ran every command listed in the sprint contract from a clean shell:
  - `bash scripts/m2_run_jax.sh` → exit 0, `gpu [CudaDevice(id=0)]`.
  - `compare_fixture --manifest analytic-stencil-3d-advdiff-v1.yaml …`
    → `pass: true`, all variables max abs diff `0.0`.
  - `compare_fixture --manifest analytic-column-thermo-v1.yaml …` →
    `pass: true`, all variables max abs diff `0.0`.
  - `python -m json.tool` on both profile JSONs → valid JSON.
  - `pytest -q` → 187 passed (148 prior + 39 new edge cases) in ~73 s.
  - `python scripts/validate_agentos.py` → `{ok: true, errors: []}`.
  - `python scripts/check_m1_done.py` → `{ok: true}`.
  - `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`
    → top entry unchanged (`WRF GPU Porting_ Architecture &
    Verification.pdf`, 1.5 MB); no new tracked file > 100 KB.

- Independent timing reproduction (outside the bench, executed from
  `data/scratch/m2-jax-venv/bin/python` against the column fixture):
  `compile_s = 0.124 s`, `first_call_after_compile_s = 0.0019 s`,
  five-run warm median ≈ `2.75e-4 s`. Confirms the bench's
  `wall_time_s ≈ 7.4e-5 s` (column) / `3.4e-5 s` (stencil) excludes
  compile time and one post-compile warmup call.

## Results

- **Re-run of the contract validation suite: clean.** All commands
  match the worker's claims (correctness pass, single fusion, zero
  spills, `kernel_launches = 1` confirmed by `thunk_sequence.txt` and
  HLO grep, `LOCAL:0` confirmed by `cuobjdump`).
- **All 39 new edge-case tests pass; full suite 187/187 passes.**
- **Cross-checks against the XLA dump tree corroborate the profile
  numbers**: registers (`stencil` 48, `column` 22), `LOCAL:0`,
  `0 bytes spill stores/loads`, one `kKernel` thunk per problem.
- **GPU backend is real**: `jax_backend.json` and the profile JSONs
  both declare `gpu` with `cuda:0`; the venv resolves the
  jax_cuda13_plugin and `jax.default_backend() == "gpu"`.
- **Compile time is honestly excluded.** Independent measurement shows
  compile ≈ 120 ms, first-call ≈ 1.9 ms, warm runs in the
  10–100 microsecond range — well separated from the reported
  `wall_time_s`.

## Fixtures Used

- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (M1 frozen
  stencil reference; also acts as the bench input).
- `fixtures/samples/analytic-column-thermo-v1.npz` (M1 frozen column
  reference; also acts as the bench input).
- `fixtures/manifests/analytic-{stencil,column}-*.yaml` (tolerance
  manifests for `compare_fixture`).
- `data/profiler_artifacts/jax/{stencil,column}_xla_dump/` (XLA dump
  tree, including `*.ptx`, `*.ptxas.cubin`, `thunk_sequence.txt`).
- `data/profiler_artifacts/jax/{stencil,column}_cuobjdump_resource_usage.txt`
  (independent ptxas + cuobjdump record).
- `data/scratch/m2-jax/{stencil,column}_run.json`,
  `jax_backend.json`, `pip_freeze.txt` (bench witness files).
- `data/scratch/m2-jax-venv/` (jax[cuda13]==0.10.0 venv).

## Gaps

- **ncu performance-counter permission is unavailable on this
  workstation.** The runner correctly captures `ERR_NVGPUCTRPERM` and
  falls back to ptxas+cuobjdump for registers and `LOCAL:N`; bandwidth
  is fallback-derived from `host_device_transfer_bytes / wall_time_s`.
  This matches the M2-S2/S3/S4 candidates and is properly disclosed in
  each profile's `profiler_limitation` field. It is **not** a worker
  defect, but the reviewer should treat `achieved_bandwidth_gbps` as
  H↔D throughput, not effective DRAM throughput.
- **`kernel_launches = 1` counts the compute kernel only.** The
  column problem's `thunk_sequence.txt` records `1 kKernel` plus
  `1 kCopy` (the `pressure_next = pressure_initial + 0` aliasing
  emitted as a copy thunk). Both candidates' totals are ≤ 5 either way,
  but the reviewer should note that XLA still emits the trivial
  alias-copy thunk rather than folding it into the fusion. This is a
  minor finding, not a contract violation.
- **Stencil ptxas reports a 40-byte stack frame** (`__internal_trig_reduction_slowpathd`,
  invoked from `jnp.sin` for the diffusivity profile). It is *not* a
  spill (`0 bytes spill stores / 0 bytes spill loads`) and the cubin
  resource summary shows `LOCAL:0`. The contract's local-memory
  invariant therefore holds, but a future worker swapping the
  diffusivity formula to one without trig should expect this stack
  frame to disappear.
- **Wall time is measured with `time.perf_counter_ns()` on the host
  bracketing a `block_until_ready()`.** This is the same methodology
  as the prior candidates, so cross-candidate comparison is fair, but
  the absolute number includes host launch latency on top of GPU
  execution. The reviewer should not treat the `wall_time_s` number as
  pure GPU time; it is the right number for ADR-001's "how long does
  one jit call take in practice?" question.
- **Maintainability narrative is honest but minimal.** All four
  required topics (install, error legibility, debugger, agent
  friction) are covered in ≤ 300 words, but the document does not
  quote the actual XLA broadcast traceback length captured in
  `deliberate_jax_bug.txt` (~58 bytes — JAX 0.10 truncates to the
  primitive-level shape error). Reviewer may want a richer error
  excerpt before merging the ADR-001 row.

## Decision

Decision: **Accept.** The JAX bakeoff candidate satisfies every Acceptance
Criterion in the sprint contract: (a) the venv is created and idempotent
under `data/scratch/m2-jax-venv/` with `jax==0.10.0` / `jaxlib==0.10.0`
and `jax.default_backend() == "gpu"`; (b) both `compare_fixture` round
trips pass with zero max-abs-diff; (c) both profile JSONs validate
schema, declare `fallback-derived` bandwidth + `profiler_limitation`,
report `kernel_launches = 1` (corroborated independently by the HLO
fusion count, the `thunk_sequence.txt` kKernel count, and the
`cuobjdump` resource dump), `local_memory_bytes = 0` on **both**
problems (the contract's primary ADR-001 signal — JAX does **not**
spill on this column kernel), `registers_per_thread = 48 / 22` within
the 64 / 128 limits, occupancy ≥ 25 / 20 %; (d) `wall_time_s` honestly
excludes the ~120 ms XLA compile time and one post-compile warmup
call, verified by an independent timing reproduction outside the bench;
(e) `pytest -q` is 187/187 green including the 39 new tester edge-case
checks; (f) `validate_agentos.py` and `check_m1_done.py` both report
`ok: true`; (g) no committed file exceeds 100 KB beyond pre-existing
content. Open the sprint to reviewer for blind code/artifact review,
then to manager for ADR-001 entry and M2-S6 (triton) launch.
