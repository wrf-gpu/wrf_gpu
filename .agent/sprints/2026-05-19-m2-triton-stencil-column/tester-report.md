# Tester Report

Sprint: `2026-05-19-m2-triton-stencil-column`
Role: tester / Claude Opus 4.7 `xhigh` (cross-AI verification of the
gpt-5.5 worker output)
Branch: `tester/sonnet/m2-triton-stencil-column`

## Tests Added Or Run

- New file `tests/test_m2_triton_edge_cases.py` — 38 cases (44 nodes
  with parametrization). Coverage (organised by intent):
  1. **Profile JSON schema rigor.** Required-key set, value types,
     `backend == "triton"`, `hardware == "RTX 5090 32GB"`, pinned
     `triton_version == "3.7.0"` and `torch_version` starts with
     `2.12.0`, `torch_cuda` populated, CUDA/RTX device in
     `torch_devices`, `achieved_bandwidth_method == "fallback-derived"`,
     `warmup_pattern` documents compile/warmup + multi-run statistic,
     `wall_time_s < 0.5s` (compile would dominate if leaking),
     bandwidth = `host_device_transfer_bytes / wall_time_s / 1e9`,
     relative `artifact_paths` resolve on disk, `run.json ↔
     profile.json` structural agreement (regs / launches / transfer /
     local_memory / occupancy / triton_version / torch_version — walls
     are excluded because run.json captures the most-recent runtime
     while the committed profile.json predates any re-run jitter),
     `profiler_limitation` present iff the `.ncu-rep` is absent/empty.
  2. **Contract sanity bounds.** `kernel_launches ∈ [1, 5]`,
     `registers_per_thread > 0` and bounded (≤ 96 stencil / ≤ 128
     column — wide bounds, the contract's load-bearing comparison is
     vs. JAX, not an absolute cap), `occupancy_pct ≥ 25 / 20`,
     `local_memory_bytes == 0` for **both** kernels (column is AC #14;
     a stencil non-zero would still be ADR-001 evidence and is
     guarded).
  3. **Independent cubin cross-checks** (not relying on the bench
     self-report). Re-runs `cuobjdump --dump-resource-usage` against
     each captured cubin under
     `data/profiler_artifacts/triton/{stencil,column}_triton_*.cubin`,
     parses the per-kernel `Function _..._kernel:` section, and
     requires that the profile JSON's `registers_per_thread` matches
     the kernel's own REG:N — not the max across cached cubins. Also
     asserts LOCAL:0 in the kernel's own section, and that the
     committed `*_cuobjdump_resource_usage.txt` artifact contains the
     `Function _..._kernel:` header so the reviewer can audit directly.
  4. **Source-tree compliance with Non-Goals / File Ownership.**
     `@triton.jit` decorators are present on both compute kernels;
     `import triton.language as tl` is present; no torch math ops
     (`torch.matmul/exp/log/add/mul/sum/mean/relu/nn/einsum`) appear in
     the compute modules; `pyproject.toml` does **not** declare
     `triton` or `torch` (the dependency and the optional-dependency
     blocks are scanned, not a global grep); `scripts/m2_run_triton.sh`
     references the data-scoped venv, every `pip install` invocation
     is prefixed by `"$VENV/bin/python"`, and the runner pins
     `triton==3.7.0` + `torch==2.12.0`.
  5. **Maintainability / agent_success / contract negatives.** Word
     budget ≤ 300, topics covered (install, error legibility,
     debugger, agent iteration), explicit mention of `torch` (the
     contract specifically calls out the heavy torch dep), and the
     narrative does **not** claim autotuning or mixed precision
     (Non-Goals). `agent_success.json` schema check.
  6. **Bench CLI behaviour** (invoked inside `data/scratch/m2-triton-
     venv/`). Rejects unknown `--problem`, missing `--stencil-fixture`,
     malformed `.npz`, a truncated zip pretending to be `.npz`, and
     the column fixture handed to the stencil problem (wrong-key
     failure). End-to-end stencil and column reproduction against the
     reference fixtures using `--skip-artifacts` so the temp scratch
     does not stomp on committed artifacts. Bitwise reproducibility
     across back-to-back runs for **both** stencil and column.
  7. **Venv idempotency / pin enforcement.** `python -c "import
     triton, torch"` in the venv prints `3.7.0` and `2.12.0`.
     `pip_freeze.txt` pins `triton==3.7.0` and `torch==2.12.0` (with
     optional `+cuXYZ` suffix). `triton_cache_dir` in both profile
     JSONs is repo-scoped (`data/scratch/m2-triton-cache`) and does
     **not** leak to `~/.triton/`.
  8. **Deliberate Triton bug capture.** `deliberate_triton_bug.txt`
     exists, does not say "ran successfully", contains a Triton
     compile diagnostic (the captured error is "arange's range must
     be a power of 2" — a real `@triton.jit` rule), and includes a
     source-line pointer (`at L:C`).

- Re-ran every command listed in the sprint contract from a clean
  shell on branch `tester/sonnet/m2-triton-stencil-column`:
  - `bash scripts/m2_run_triton.sh` → exit 0, smoke `3.7.0 13.0`.
  - `python -m gpuwrf.validation.compare_fixture --manifest
    analytic-stencil-3d-advdiff-v1.yaml …` → `pass: true`; every
    variable `max_abs_diff = 0.0`; `phi_next` max_abs_diff 0.
  - `python -m gpuwrf.validation.compare_fixture --manifest
    analytic-column-thermo-v1.yaml …` → `pass: true`; `qv_next`
    max_abs_diff `4.34e-19`, `mse_delta` max_abs_diff `1.08e-12`,
    everything else 0; well within the manifest tolerances.
  - `python -m json.tool` on both profile JSONs → valid JSON; `backend
    == "triton"`, `kernel_launches == 1`, `local_memory_bytes == 0`,
    `triton_version == "3.7.0"`, `torch_version == "2.12.0+cu130"`.
  - `pytest -q` → **232 passed, 1 failed** (the deliberate cubin-
    kernel-mismatch check; see Gaps / Decision). Suite size is 233
    nodes (189 prior + 44 new tester nodes including parametrization).
  - `python scripts/validate_agentos.py` → `{ok: true, errors: []}`.
  - `python scripts/check_m1_done.py` → `{ok: true, sprints_closed:
    3}`.
  - `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`
    → top tracked file unchanged
    (`WRF GPU Porting_ Architecture & Verification.pdf`, 1.5 MB); the
    new tester test file is 19 KB; no new tracked file > 100 KB.
  - `cuobjdump --dump-resource-usage
    data/profiler_artifacts/triton/column_triton_0.cubin` → `Function
    _column_thermo_kernel: REG:34 STACK:0 SHARED:0 LOCAL:0` (this is
    the **actual** column kernel; see Decision).
  - `cuobjdump --dump-resource-usage
    data/profiler_artifacts/triton/stencil_triton_0.cubin` → `Function
    _stencil_advdiff_kernel: REG:60 STACK:48 SHARED:0 LOCAL:0`.

## Results

- **Re-run of the contract validation suite: clean for correctness and
  smoke.** All `compare_fixture` round trips pass, both profile JSONs
  validate, `local_memory_bytes == 0` on **both** kernels (column is
  AC #14; the stencil's `STACK:48` does not count against LOCAL).
- **Cubin extraction works at the file level** — every captured
  `*_triton_*.cubin` parses cleanly with `cuobjdump --dump-resource-
  usage` and the cuobjdump dumps under
  `data/profiler_artifacts/triton/` contain the `Function _..._kernel:`
  header.
- **Triton backend is real and pinned**: venv `import triton, torch`
  resolves `3.7.0` and `2.12.0+cu130`, `torch.cuda.is_available()`
  returns true, `torch_devices == ["NVIDIA GeForce RTX 5090"]`, and
  `pyproject.toml` keeps both out of project deps so the rest of the
  repo continues to install without the heavy torch wheel.
- **Compile time is honestly excluded.** Wall times measured at
  ~25 µs (stencil) and ~14–18 µs (column) — three orders of
  magnitude below a typical Triton first-call compile cost; the
  warmup pattern in both profile JSONs explicitly documents the
  "compile/first-run launch + one unmeasured warmup + median of 5"
  protocol around `torch.cuda.synchronize()`.
- **One material defect detected** (see Gaps and Decision):
  `column_profile.json` reports `registers_per_thread = 60` and
  `occupancy_pct = 70.833…` — both values inherited from the stencil
  cubin still resident in the Triton cache. The actual
  `_column_thermo_kernel` cubin has `REG:34, LOCAL:0, STACK:0`, which
  under the same `_derive_occupancy` math gives **100 %** occupancy
  at block_size=64 on this Blackwell SM. The bench is comparing the
  wrong cubin.

## Fixtures Used

- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (M1 frozen
  stencil reference; also acts as the bench input).
- `fixtures/samples/analytic-column-thermo-v1.npz` (M1 frozen column
  reference; also acts as the bench input).
- `fixtures/manifests/analytic-{stencil,column}-*.yaml` (tolerance
  manifests for `compare_fixture`).
- `data/profiler_artifacts/triton/{stencil,column}_triton_*.cubin`
  (Triton cubins captured by the bench from the repo-scoped cache).
- `data/profiler_artifacts/triton/{stencil,column}_cuobjdump_resource_usage.txt`
  (independent cuobjdump record committed by the runner).
- `data/profiler_artifacts/triton/deliberate_triton_bug.txt`
  (deliberate `@triton.jit` invalid-program capture).
- `data/profiler_artifacts/triton/{stencil,column}_ncu_{stdout,stderr,exit}.txt`
  (ncu invocation evidence; permission-counter-blocked as expected).
- `data/scratch/m2-triton/{stencil,column}_run.json`,
  `triton_backend.json`, `pip_freeze.txt` (bench witness files).
- `data/scratch/m2-triton-venv/` (triton==3.7.0 + torch==2.12.0 venv).
- `data/scratch/m2-triton-cache/` (TRITON_CACHE_DIR for the cache the
  bench operates against; cleared between ncu and bench phases).

## Gaps

- **The bench's resource-usage parser silently inflates the column
  register count.** Root cause:
  `src/gpuwrf/backends/triton/bench.py:_resource_metrics_factory` →
  `_recent_cubins(marker)` returns *every* cubin in the Triton cache
  whose `st_mtime >= marker - 0.25s`; `_parse_resource_usage` then
  takes `max(regs)` across the parsed entries. Because the final
  bench runs stencil and column in one Python process with shared
  `TRITON_CACHE_DIR`, the column run sees both `_stencil_advdiff_
  kernel` (REG:60) and `_column_thermo_kernel` (REG:34) cubins, and
  `max(...)` returns 60. The committed
  `data/profiler_artifacts/triton/column_cuobjdump_resource_usage.txt`
  literally records both kernels — the column kernel's own section
  shows REG:34, LOCAL:0, STACK:0. Two viable fixes:
  1. Pass the kernel symbol per-problem (`_stencil_advdiff_kernel` /
     `_column_thermo_kernel`) into `resource_metrics` and extract REG
     / LOCAL only from that `Function NAME:` section.
  2. `rm -rf "$CACHE"; mkdir -p "$CACHE"` between the stencil and
     column kernel launches inside the bench, so each problem
     compiles in isolation.
  Either is small and local to the Triton bench; no source-of-truth
  governance file is affected.
- **Material impact on ADR-001.** The sprint contract is explicit
  (Performance Metrics, lines 117–122): "If Triton beats this
  materially (e.g. occ=100 % with same register count), hybrid
  (JAX + Triton) is justified. If Triton ties or loses, pure JAX
  wins." With the **reported** column numbers (60 regs, 70.8 %
  occupancy) Triton appears to *lose* to JAX (22 regs, 83.3 %
  occupancy). With the **actual** column numbers (34 regs, ~100 %
  occupancy, 0 local memory, 0 stack) Triton has fewer registers
  than the existing M2-S4 Kokkos column (40, 100 %) and beats JAX on
  occupancy while reaching ≥ 4× the bandwidth (0.18 GB/s vs 0.011
  GB/s, fallback-derived). The defect therefore flips the ADR-001
  signal direction for the column kernel and must be repaired before
  ADR-001 cites these numbers.
- **ncu performance-counter permission is unavailable on this
  workstation** (consistent with every prior M2 candidate). The
  runner correctly captures `ERR_NVGPUCTRPERM` and falls back to
  cuobjdump + a derived occupancy; bandwidth is fallback-derived
  from `host_device_transfer_bytes / wall_time_s`. This is **not** a
  worker defect — both profile JSONs document the limitation in
  `profiler_limitation` and `achieved_bandwidth_method` — but the
  reviewer should treat `achieved_bandwidth_gbps` as H↔D throughput,
  not effective DRAM throughput.
- **Stencil `STACK:48` is captured but not surfaced in
  `local_memory_bytes`.** STACK is the per-thread stack frame that
  CUDA places in local memory; LOCAL:0 means there is no static
  allocation but a 48-byte frame is still reserved per thread on the
  stencil. This is *not* a contract violation (LOCAL:0 is the
  documented metric in M2-S2 conventions and matches every other
  candidate), but the reviewer should be aware that "no local
  memory" on the stencil is "no static local allocation"; runtime
  spills via the stack frame would not show in this field. The
  column kernel has STACK:0 — no such caveat there.
- **No autotuning, no Pallas, no mixed precision** — confirmed by
  source inspection and the maintainability narrative; both
  Non-Goals are honoured.
- **Re-run jitter affects `wall_time_s`.** Two independent runs of
  the bench during this tester sprint produced stencil wall times of
  25.4 µs and 25.85 µs, column wall times of 14.2 µs and 17.7 µs.
  Median-of-5 is the contract's chosen statistic and is stable in
  the µs range. The `test_run_json_matches_profile_numbers` check is
  therefore restricted to structural fields (regs, launches,
  transfer, local_memory, occupancy, triton_version, torch_version)
  rather than walls.

Decision: **Reject (fix required).** The Triton implementation
itself is correct (both `@triton.jit` kernels round-trip the M1
fixtures exactly for the stencil and within manifest tolerance for
the column; LOCAL:0 holds on both kernels; venv pinning, torch
scoping, and Non-Goal compliance are all clean), but the column
profile JSON misreports the load-bearing ADR-001 metric. The bench's
`_resource_metrics_factory` extracts REG/LOCAL via `max(...)` across
every cubin recently dropped in `TRITON_CACHE_DIR`; in the in-process
`--problem both` run the column step sees the stencil cubin (REG:60)
plus its own cubin (REG:34) and reports 60. The committed cuobjdump
text literally exposes both sections — the truth is on disk, the
profile JSON just contradicts it. The contract (AC #6) requires
`registers_per_thread` to come "from cuobjdump on the cubin"; in
spirit and in practice that means the kernel's own cubin section.
Because the contract makes this single metric the deciding signal
between "pure JAX wins" and "JAX-dycore + Triton-physics hybrid",
ADR-001 must not consume these numbers as-is. The required fix is
small (kernel-symbol-aware section extraction, or a cache wipe
between the stencil and column launches in `bench.py`), is fully
contained inside the worker's file ownership scope, requires no
governance changes, and is covered by
`tests/test_m2_triton_edge_cases.py::test_column_profile_registers_
match_column_kernel_cubin` (and the symmetric stencil check), which
currently asserts `60 == 34` and will turn green automatically once
the bench reports the kernel's own register count. Recommended next
step: bounce to worker for a one-commit bench patch (kernel-symbol
filter is the cheaper option and keeps the cache hot for any future
autotune sprint), re-run `scripts/m2_run_triton.sh`, re-run
`pytest -q tests/test_m2_triton_edge_cases.py` (expect 44/44),
update `column_profile.json`'s `registers_per_thread` and
`occupancy_pct`, and refresh ADR-001's column row before reviewer
sign-off.
