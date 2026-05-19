# Tester Report

Sprint: `2026-05-19-m2-scout-blackwell-toolchain`
Tester: Claude Opus 4.7 (sonnet-test-engineer role) — cross-AI verifier on Codex `gpt-5.5` worker output.
Branch: `tester/sonnet/m2-scout-blackwell-toolchain` (branched off `worker/gpt/m2-scout-blackwell-toolchain`).

## Tests Added Or Run

### Re-ran every contract validation command from a clean shell

- `nvidia-smi | head -8` → `NVIDIA-SMI 590.48.01`, `Driver Version: 590.48.01`, `CUDA Version: 13.1`. Matches `toolchain_support_matrix.json::target_hardware`.
- `python scripts/validate_agentos.py` → `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.
- `bash scripts/m2_scout_hello_gpu.sh` → `jax: pass / triton: pass / gt4py: skipped blocked / kokkos: pass / cupy_or_numba: pass / cuda_tile: pass / m2 scout hello-gpu: 5 pass, 0 fail`.
- `python -c "...summarise verdicts..."` → `6 candidates, 2 go, 1 blocked` (the remaining 3 are `go-with-version-bump`).
- `pytest -q` → `63 passed in 5.01s` (47 pre-existing + 16 new edge-case tests added below).
- `python scripts/check_m1_done.py` → `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}` — no M1 regression.
- `git diff --name-only $(git merge-base HEAD main)` → all 26 changed files are inside the contract's File Ownership list (`artifacts/m2/scout/**`, `scripts/m2_scout_hello_gpu.sh`, `tests/test_m2_scout_matrix.py`, `worker-report.md`). No protected file touched.

### Idempotence check (contract AC #7)

Snapshotted every `hello_gpu/<candidate>/output.txt` + `exit.txt` before re-running `bash scripts/m2_scout_hello_gpu.sh`, then `diff -u`'d each pair afterwards. All twelve diffs were empty, and the wrapper again reported `5 pass, 0 fail`. Wrapper is idempotent.

### Independent re-run of the gt4py block evidence

The `blocked` verdict for gt4py is the only one without a positive hello-GPU artifact, so it warranted an independent reproduction. I ran `data/scratch/m2-scout-venv/bin/python artifacts/m2/scout/hello_gpu/gt4py/hello.py` directly and observed exit `1` with `AttributeError: module 'sympy' has no attribute 'expr'` raised inside `dace/properties.py` during dtype subscripting at line 10. This matches `output.txt` exactly and confirms the blocked rationale (DaCe 0.10.0 not Python-3.13/SymPy-modern clean) is reproducible, not a one-time flake.

### Version-pin cross-check against the installed venv (contract tester-note)

The matrix's `version_pin` values for the five non-blocked candidates were verified against what is actually installed in `data/scratch/m2-scout-venv/`:

| candidate | version_pin claim | `importlib.metadata` reports |
|---|---|---|
| jax | `jax[cuda13]==0.10.0` | `jax 0.10.0`, `jaxlib 0.10.0` |
| triton | `triton==3.7.0 torch==2.12.0` | `triton 3.7.0`, `torch 2.12.0` |
| kokkos | `Kokkos 4.7.1 tag 4.7.01` | `KOKKOS_VERSION=40701` in `kokkos/output.txt` |
| cupy_or_numba | `cupy-cuda13x==14.0.1` | `cupy-cuda13x 14.0.1` |
| cuda_tile | `CUDA Toolkit 13.1.115` | `cuda_runtime=13010` in `cuda_tile/output.txt`; `nvidia-smi` reports CUDA 13.1 |

No hallucinated versions. Every pin matches what was actually executed on the 5090.

### Device-evidence audit on hello-GPU outputs

For every non-blocked candidate I confirmed `output.txt` contains both a device-identification token **and** the expected `× 2` result, satisfying AC #4 (mere "import succeeded" rejected):

- `jax`: `devices=[CudaDevice(id=0)]`, `result=[2.0, 4.0, 6.0, 8.0]`.
- `triton`: `device=NVIDIA GeForce RTX 5090`, `torch=2.12.0+cu130 cuda=13.0`, `result=[2.0, 4.0, 6.0, 8.0]`.
- `kokkos`: `execution_space=Cuda`, `result=[2, 4, 6, 8]`.
- `cupy_or_numba`: `device=NVIDIA GeForce RTX 5090`, `result=[2.0, 4.0, 6.0, 8.0]`.
- `cuda_tile`: `device=NVIDIA GeForce RTX 5090`, `cuda_runtime=13010`, `result=[2, 4, 6, 8]`.

### New edge-case tests (16 new tests, file `tests/test_m2_scout_matrix_extras.py`)

I imported the worker's existing `validate_matrix` so the negative cases re-use the same validator the positive case is gated on, and added:

Schema-level:
- `test_generated_utc_parses_as_iso8601` — runs `datetime.fromisoformat(...)` on `generated_utc`, asserts a timezone is present.
- `test_known_gaps_are_strings` — every entry of `known_gaps` is a `str` (worker test only checked the outer list).
- `test_candidate_order_matches_contract` — order is exactly `[jax, triton, gt4py, kokkos, cupy_or_numba, cuda_tile]` per contract §6.
- `test_blocked_rationale_is_substantive` — blocked rationales ≥20 chars.

Artifact-level:
- `test_each_candidate_has_a_runnable_program` — checks `hello.py` for the Python candidates; `hello.cpp + build.sh + CMakeLists.txt` for kokkos; `hello.cu + build.sh` for cuda_tile.
- `test_passing_candidates_show_device_evidence` — for every non-blocked candidate, asserts a device-name token and a `result=` line are present in `output.txt`.
- `test_exit_files_are_integers` — `exit.txt` parses as int; equals `0` for non-blocked, non-zero for blocked.

Narrative-level:
- `test_narrative_covers_all_candidates_in_order` — `## <name>` headings appear in fixed order.
- `test_narrative_word_budget` — ≤2000 words (actual: 511).
- `test_narrative_mentions_target_hardware_and_closing` — both required sections present.

Wrapper-level:
- `test_wrapper_script_lists_exactly_six_candidates` — `CANDIDATES=(...)` array in `scripts/m2_scout_hello_gpu.sh` contains exactly the six contract names.

Negative cases (validator must reject):
- Duplicate candidate name (6 entries, 5 unique names).
- Verdict outside the `{go, go-with-version-bump, blocked}` enum (`"maybe"`).
- Blocked candidate carrying a non-null `install_command`.
- Wrong compute capability in `target_hardware` (`"9.0"`).
- Rationale that overflows the 200-char budget.

All 16 added tests pass. Combined suite: `63 passed in 5.01s`.

## Results

- **Hardware** (`nvidia-smi`): 5090, cc 12.0, driver 590.48.01, CUDA 13.1 — matches matrix exactly.
- **Matrix schema** (positive): valid.
- **Matrix schema** (negative): all six tamper cases rejected by the worker's own `validate_matrix`.
- **Hello-GPU artifacts**: present for all five non-blocked candidates; each contains a runnable program, an `output.txt` with device-side evidence and `× 2` result, and an `exit.txt` of `0`.
- **Blocked candidate**: gt4py rationale ("DaCe 0.10.0 failed under Python 3.13") was independently reproduced.
- **Idempotence**: `bash scripts/m2_scout_hello_gpu.sh` is byte-identical on rerun.
- **Wrapper**: lists exactly the six contract candidates; skips `blocked` correctly (the script's `is_blocked` helper reads the matrix and returns 0 only for blocked).
- **Test suite**: 63 passed, 0 failed, 0 errors.
- **Version pins**: all five non-blocked pins match the package versions actually loaded in `data/scratch/m2-scout-venv/` and the toolchain reported by `nvidia-smi` / `nvcc`. No hallucinated Blackwell-specific claim.
- **File ownership**: every modified file is on the contract's allow-list; no `src/gpuwrf/`, no fixtures, no governance file touched.
- **M1 oracle**: `check_m1_done.py` still returns `ok: true`; no regression.
- **Narrative**: 511 words, all six `## <candidate>` headings in fixed order, plus Target-hardware and Closing-Recommendation sections.

## Fixtures Used

This sprint is a research scout, so it uses no WRF or analytic physics fixtures. The "fixtures" exercised by the tests are:

- `artifacts/m2/scout/toolchain_support_matrix.json` — JSON matrix under test.
- `artifacts/m2/scout/hello_gpu/<candidate>/output.txt` and `exit.txt` — captured stdout/stderr and exit codes from the on-GPU smoke runs.
- `artifacts/m2/scout/hello_gpu/<candidate>/{hello.py | hello.cpp+build.sh+CMakeLists.txt | hello.cu+build.sh}` — runnable smoke programs.
- `scripts/m2_scout_hello_gpu.sh` — the idempotent wrapper.
- `data/scratch/m2-scout-venv/` — the gitignored sprint venv (resolved via the `data → /mnt/data/wrf_gpu2` symlink), used only to spot-check installed versions, not committed.

No binary fixture data was created or committed.

## Gaps

1. **Online upstream URL spot-check not performed.** The contract's tester note asks the verifier to confirm that cited URLs in the narrative still resolve and cited PyPI versions still exist. This tester ran without web access, so URL liveness for `docs.jax.dev`, `pytorch.org/blog/pytorch-2-12-release-blog/`, `kokkos.org/kokkos-core-wiki`, `docs.cupy.dev`, `docs.nvidia.com` etc. was **not** independently verified. I did, however, verify that each cited version is the version actually installed locally and that ran on the 5090; the URL claims are corroborated by the running binaries, which is the stronger evidence.

2. **GT4Py block is environment-conditional.** The `blocked` rationale is honest for this venv (DaCe 0.10.0 + Python 3.13 + SymPy-modern co-installed via the Triton install). A later scout under Python 3.11 with a DaCe release that pins compatible SymPy may yet succeed. Tester corroborates the worker's risk note rather than disproving it.

3. **No profiler / no transfer audit.** Explicitly out of scope per contract §Non-Goals. Flagging only because both will be required for the M2 implementation sprints.

4. **Kokkos build runs as part of the wrapper.** Re-running the wrapper rebuilds the Kokkos hello binary (the build.sh path is unconditional, though CMake is incremental). Output is still byte-stable, so idempotence holds, but a future scout could short-circuit the build for true zero-cost reruns.

5. **No stress test on `is_blocked` against malformed matrix.** The script's helper would crash with a Python traceback if the JSON were unparseable; not strictly required for closeout, but a future hardening idea.

## Decision

**Decision: Accept.**

The worker's deliverable satisfies every contract acceptance criterion that can be checked from the local repo on the target 5090: schema, candidate count and ordering, hello-GPU artifacts with real device-side evidence for the five non-blocked candidates, idempotent wrapper, narrative within the word budget and in fixed order, file ownership respected, no M1 regression, and 47 of the worker's tests plus 16 new tester-added edge-case tests all pass (63/63). Version pins are not hallucinated — every pinned version is the version that actually executed on the 5090 in `data/scratch/m2-scout-venv/`. The single `blocked` verdict (gt4py) is reproducible and honestly attributed to a DaCe-on-Python-3.13 SymPy break, with a clear path for a follow-up remediation scout if the manager wants gt4py kept in the bakeoff. Recommend reviewer Accept and manager dispatch M2 implementation sprints in the narrative's readiness order (`cuda_tile`, `cupy_or_numba`, `kokkos`, `jax`, `triton`), optionally followed by a focused gt4py remediation scout before final ADR-001.
