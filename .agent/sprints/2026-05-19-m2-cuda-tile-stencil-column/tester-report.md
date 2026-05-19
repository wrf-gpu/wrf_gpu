# Tester Report

Role: tester (Claude Opus 4.7 acting as `sonnet-test-engineer`, cross-AI verification).
Branch: `tester/sonnet/m2-cuda-tile-stencil-column`.
Sprint: `2026-05-19-m2-cuda-tile-stencil-column`.

## Tests Added Or Run

### Re-run of every contract validation command from a clean shell

All commands run from the project root with a fresh source of
`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`.

| Step | Command | Outcome |
| --- | --- | --- |
| 1 | `rm -rf data/scratch/cuda_tile/bench && bash src/gpuwrf/backends/cuda_tile/build.sh` | nvcc primary path fails on the CUDA 13.1 + glibc `rsqrt`/`rsqrtf` exception-spec conflict (as the worker reported); the wrapper's documented fallback `nvc++ -cuda -gpu=cc120 -O3 -std=c++17` succeeds and produces `data/scratch/cuda_tile/bench`. |
| 2 | `cuobjdump --dump-sass data/scratch/cuda_tile/bench \| grep '^arch ='` | Single hit: `arch = sm_120`. `cuobjdump --list-elf` shows one cubin (`*.sm_120.cubin`); no lower-CC fallback in the binary. `cuobjdump --list-ptx` reports no PTX (so the runtime can't JIT to anything other than sm_120). |
| 3 | `bash scripts/m2_run_cuda_tile.sh` | Idempotent: build no-ops on the second invocation, both bench runs print well-formed JSON to stdout, both `compare_fixture` invocations pass, both profile JSONs regenerate. |
| 4 | `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/cuda_tile/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` | `pass: true`; every variable (`phi_initial`, `u_face`, `v_face`, `w_face`, `phi_next`) reports `max_abs_diff = 0.0`. |
| 5 | `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/cuda_tile/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz` | `pass: true`; only deltas are `qv_next max_abs_diff = 4.34e-19` and `mse_delta max_abs_diff = 1.08e-12`, both well inside the manifest tolerances (1e-13 abs / 1e-12 rel and 1e-10 abs respectively). |
| 6 | `python -m json.tool artifacts/m2/cuda_tile/{stencil,column}_profile.json` | Both JSONs parse cleanly. |
| 7 | `pytest -q` | 86 passed (65 pre-existing + 21 new tester edge cases). 0 failures, 0 errors. |
| 8 | `python scripts/validate_agentos.py` | `ok: true`, 31 required files checked, 13 skills checked. |
| 9 | `python scripts/check_m1_done.py` | `ok: true`, `sprints_closed: 3`. No regression. |
| 10 | `python scripts/check_m2_done.py` | Exits 1 as expected for an open M2 (`candidates_satisfied: 1/6`; the `cuda_tile` row is satisfied, the rest of the milestone is out of scope for this sprint). The script also flags that the reviewer/tester/closeout reports for this sprint are still stub-sized — this report when written replaces my stub. |
| 11 | `git ls-files -z \| xargs -0 stat -c '%s %n' \| sort -nr \| head -5` | Top tracked files remain pre-existing PDFs and the 65 KB M1 sample; no new tracked file exceeds 100 KB. |

### Cross-check: profile JSON values are not fabricated

The contract's AC #10 explicitly requires the JSON numbers to come from a
profiler artifact rather than a hand-edited estimate. I re-derived each
field independently:

- `registers_per_thread` and `local_memory_bytes`: parsed
  `data/scratch/cuda_tile/resource_usage.txt` produced by
  `cuobjdump --dump-resource-usage` for both kernel symbols and confirmed
  bit-for-bit equality with the profile JSONs
  (stencil `REG:58 LOCAL:0`, column `REG:24 LOCAL:0`). New test
  `test_resource_usage_matches_profile_register_counts` enforces this.
- `kernel_launches`, `host_device_transfer_bytes`, `wall_time_s`,
  `occupancy_pct`: re-read from `data/scratch/cuda_tile/{stencil,column}_run.json`
  (the JSON the bench binary itself prints) and matched against the profile
  JSON. New test `test_run_json_matches_profile_numbers` enforces this.
- `achieved_bandwidth_gbps`: recomputed as `transfer / wall / 1e9` and
  matched against the profile to within 1e-3 relative tolerance. New test
  `test_achieved_bandwidth_is_consistent_with_transfer_and_wall`.
- `artifact_paths`: every entry is relative and resolves to an existing
  file under `ROOT`. New test
  `test_profile_artifact_paths_are_relative_and_exist`.
- I also verified the hardcoded diffusivity values in `stencil.cu`
  (`{18.0, 19.414213562373095, 20.0, ...}`) reproduce the analytic
  generator's `18.0 + 2.0 * sin(2π·k/8)` for `k=0..7` to fp64 precision —
  consistent with `phi_next max_abs_diff = 0.0`.

### Cross-check: kernel really targets sm_120

- `cuobjdump --dump-sass` shows exactly one architecture tag (`arch = sm_120`)
  and the disassembly's `.target sm_120` directive.
- `cuobjdump --list-elf` lists a single `*.sm_120.cubin` ELF.
- `cuobjdump --list-ptx` reports no PTX in the binary, ruling out silent
  JIT to a lower CC at runtime.
- New test `test_bench_binary_targets_only_sm120` asserts the set of
  observed SASS architectures is exactly `{sm_120}`.

### Adversarial probes of the bench binary

I tried to break `data/scratch/cuda_tile/bench` with several malformed
invocations; every case exits non-zero with a diagnostic on stderr:

| Probe | Outcome |
| --- | --- |
| `bench` (no args) | `usage: bench stencil\|column --input path --output path` |
| `bench foo --input ... --output ...` | `unknown problem: foo` |
| `bench stencil` (missing `--input/--output`) | `--input and --output are required` |
| `bench --input ... --output ...` (no positional problem) | `expected --key value argument` |
| `bench stencil --input /nonexistent.npz --output ...` | `cannot open input file: /nonexistent.npz` |
| `bench stencil --input /tmp/garbage.npz --output ...` | `zip too small` |
| `bench stencil --input <column-fixture> --output ...` | `missing array: phi_initial` (correctly refuses to silently produce garbage) |

All of these are covered by new tests
`test_bench_reports_usage_on_missing_args`,
`test_bench_rejects_unknown_problem`,
`test_bench_rejects_missing_required_flag`,
`test_bench_rejects_missing_input_file`,
`test_bench_rejects_malformed_npz`,
`test_bench_detects_wrong_fixture_for_problem`.

### Other invariants added

- `test_profile_schema_keys_and_types` — every required key present with
  the correct primitive type (and rejects accidental `bool` for integer
  fields).
- `test_profile_sanity_bounds_match_contract` — the contract's
  "Performance Metrics" bounds (`wall<=5s`, `launches in [1,10]`,
  `occupancy_pct` floors per problem, `registers_per_thread` ceilings per
  problem) are now hard-asserted, so a future regression that blows
  through them fails CI.
- `test_column_profile_has_zero_local_memory` and the stencil counterpart
  enforce AC #7 directly.
- `test_correctness_json_passes_both_problems` walks each variable record
  in `correctness.json` and rejects any partial failure.
- `test_agent_success_log_is_well_formed` pins the keys the manager
  closeout consumer expects.
- `test_maintainability_markdown_is_within_budget` enforces the ≤300-word
  budget and the four-section coverage (build, error, debugger,
  iteration friction) by keyword.
- `test_profiler_limitation_field_only_when_ncu_report_missing` makes the
  honesty rule structural: if `*.ncu-rep` exists, the JSON must not claim
  a profiler limitation; if it doesn't exist, the JSON must declare one.

All edge-case tests live in `tests/test_m2_cuda_tile_edge_cases.py` and
skip cleanly (instead of failing) on a host without the toolchain or the
built artifacts.

## Results

- Pre-existing pytest suite: **65/65 pass** unchanged.
- New tester edge-case suite: **21/21 pass**.
- Full `pytest -q`: **86 passed in 8.73 s**.
- `validate_agentos.py`, `check_m1_done.py`: ok.
- `check_m2_done.py`: cuda_tile row satisfied; remaining errors are
  expected and out of scope for this sprint (other M2 candidates, ADR-001,
  milestone closeout).

## Fixtures Used

- `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` (32×16×8 staggered grid, fp64 reference + fp32 face velocities).
- `fixtures/samples/analytic-column-thermo-v1.npz` (40-level column, fp64 reference).
- Worker-produced candidate outputs `data/scratch/cuda_tile/{stencil,column}_out.npz`.
- Worker-produced cuobjdump resource dump `data/scratch/cuda_tile/resource_usage.txt`.
- Worker-produced bench self-report `data/scratch/cuda_tile/{stencil,column}_run.json`.
- `ncu` stdout/stderr/exit logs under `data/profiler_artifacts/cuda_tile/` (referenced by both profile JSONs).
- Two ad-hoc adversarial inputs created in `tmp_path` (`garbage.npz`, `does_not_exist.npz`) for the malformed-input probes; not committed.

## Gaps

These are observations the reviewer/manager should weigh; none individually
block the candidate from going forward as a bakeoff baseline.

1. **nvcc primary path is broken on this workstation.** The contract's
   build step §125 calls for `nvcc -arch=sm_120`; what actually works is
   the worker's documented `nvc++ -cuda -gpu=cc120` fallback inside
   `build.sh`. The resulting SASS is still sm_120-only, so the bakeoff
   numbers are valid for the target hardware, but the project memory
   `project_target_hardware.md` is now out of sync with the toolchain
   actually in use. Updating it via the patch protocol is a manager
   call — worth doing before S3 starts so the next candidate inherits
   the working flags.

2. **No `.ncu-rep` produced.** `ncu` is invoked but exits with
   `ERR_NVGPUCTRPERM`, so `data/profiler_artifacts/cuda_tile/*.ncu-rep`
   never materializes. The contract's Risks section explicitly anticipates
   this and authorizes the "document the limitation and emit best-effort
   numbers with a `profiler_limitation` field" workaround the worker
   took. AC #9 / #10 are therefore satisfied in spirit but not in letter;
   the reviewer/manager must consciously accept this. If the cross-
   candidate comparison in ADR-001 needs counter-level metrics, the
   manager should re-run this branch with NVIDIA performance-counter
   permissions enabled before locking in the cuda_tile baseline numbers.

3. **Column kernel parallelizes within the column rather than across
   columns.** `column.cu` launches `<<<1, 64>>>` with `threadIdx.x` indexing
   levels. For the single-column M1 fixture this is functionally identical
   to the contract's stated "every cell's vertical column is processed by
   a single block, with thread-local prognostics in registers" design and
   the correctness oracle confirms equivalence. But it is not the same
   physical layout, and applied to a multi-column problem this kernel
   would launch 1 block total instead of 1-per-column. Worth flagging in
   the cross-candidate comparison; not a correctness violation on the M1
   fixture so I am not failing the candidate on it.

4. **stencil kernel reports `STACK:64`.** Not the same as `LOCAL` spills
   (which are 0 and satisfy AC #7), but it does mean each thread has a
   64-byte stack frame. Probably the `wrap_index` modulo on negative ints
   and/or the constant-array `diffusivity_for_level` lookup. Worth noting
   when comparing register pressure across candidates.

5. **Contract paths under `src/gpuwrf/backends/cuda_tile/` include a
   `Makefile` only, not the contract's "Makefile OR CMakeLists.txt".**
   That is allowed by the contract — listing both options means either
   alone is sufficient — but I am noting that the file ownership list is
   in line with the actual changes.

## Decision

**Accept-with-notes.**

Correctness passes for both bakeoff problems against the M1 analytic
oracles with zero or noise-floor deltas. The built binary genuinely
targets sm_120 with no PTX/lower-CC fallback. Every profile JSON field
agrees with an independently re-derived value from `cuobjdump`,
`compare_fixture`, or the bench binary's own per-run JSON; my new
`tests/test_m2_cuda_tile_edge_cases.py` (21 tests) enforces this
non-fabrication invariant in CI going forward. The bench binary handles
every adversarial input I tried with a non-zero exit and a useful
diagnostic.

Two contract deviations exist (`nvc++ -cuda` instead of `nvcc`, and no
`.ncu-rep` due to `ERR_NVGPUCTRPERM`) and both are inside the explicitly
anticipated risk fallbacks; the worker documented them honestly in
`worker-report.md`, `maintainability.md`, and a `profiler_limitation`
field in each profile JSON. I am calling this Accept-with-notes (rather
than a blocking Reject) because:

- the deviations are within the contract's own Risks-section
  authorization,
- they are reported transparently rather than papered over,
- the produced numbers are internally consistent and reproducible,
- the implementation is robust against the malformed-input probes,
- and the test suite I added will catch any future regression of the
  honesty rules.

The reviewer/manager should explicitly weigh in on (a) whether the
absence of an `.ncu-rep` is acceptable for ADR-001's cross-candidate
comparison and (b) whether `project_target_hardware.md` should be patched
to reflect the working `nvc++ -cuda` toolchain before S3 opens.

Decision: Accept-with-notes.
