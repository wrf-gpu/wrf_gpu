# Sprint Contract

Sprint ID: `2026-05-19-m2-scout-blackwell-toolchain`
Milestone: M2 — Backend Bakeoff
Sequence: S1 (M2's first sprint — research-scout to fail-fast on cc120/Blackwell support)
Reviewer: opus-reviewer (via codex)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (Claude Opus 4.7 — cross-AI verification per 2026-05-19 directive)
Approval status: opened 2026-05-19 by manager after M1 closeout.

## Objective

Produce an evidence-based **toolchain readiness matrix** for the six M2 candidate families on the project's target hardware (NVIDIA RTX 5090, compute capability 12.0 = Blackwell, 32 GB VRAM). The matrix tells the manager which candidates can be implementation-sprinted as-is, which need a version bump or workaround, and which should be excluded from the bakeoff with a documented `candidate-failure.json`.

Six candidate families:
1. **jax** (Python, JAX + XLA, GPU via jaxlib-cuda)
2. **triton** (Python, OpenAI Triton)
3. **gt4py** (Python, GridTools for Python + DaCe backend)
4. **kokkos** (C++, Kokkos performance-portability library targeting CUDA backend)
5. **cupy_or_numba** (Python, CuPy raw CUDA kernels or Numba CUDA — worker picks one based on Blackwell readiness)
6. **cuda_tile** (C++, explicit CUDA C++ with shared-memory tile resident kernels; the previous wrf_gpu attempt's recommended path)

For each family, answer:
- **Blackwell (cc120) support today?** What version supports it, when was support added, are there known gaps (e.g. specific CUDA features not yet supported on Blackwell)?
- **Install command** that produces a working build on the target machine (Ubuntu 24.10, CUDA 12.x, NVHPC 26.3 available per Gen2 env at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`).
- **"Hello, GPU"** smoke test: tiny program that allocates a device array, multiplies by 2, copies back, verifies. Worker actually runs this on the local RTX 5090 (`nvidia-smi` reports a 5090) and records exit code + the parsed output.
- **Verdict**: `go` | `go-with-version-bump` | `blocked` (with one-line rationale citing real evidence).

## Non-Goals

- No implementation of either bakeoff problem (stencil / column). That is M2-S2..S7.
- No performance measurement; no profiler invocation. This sprint just answers "does it build & run hello-GPU on cc120?"
- No backend ranking, no preference statement. The scout reports facts. ADR-001 (M2-S8) integrates everything.
- No installation of system-wide deps that would conflict with Gen2's env. All Python deps go into a sprint-local venv at `data/scratch/m2-scout-venv/` (gitignored).
- No purchase of cloud GPU instances. RTX 5090 is the only target.

## File Ownership

Worker may create or edit only these paths:

- `artifacts/m2/scout/toolchain_support_matrix.json` (new — the structured matrix)
- `artifacts/m2/scout/toolchain_report.md` (new — human-readable narrative ≤2000 words)
- `artifacts/m2/scout/hello_gpu/<candidate>/` (new — one directory per candidate, each containing the smoke-test program, build command, run output captured in `output.txt`, and exit-code captured in `exit.txt`)
- `scripts/m2_scout_hello_gpu.sh` (new — wrapper that runs every candidate's smoke test idempotently; writes pass/fail counts to stdout)
- `tests/test_m2_scout_matrix.py` (new — validates the matrix JSON schema, asserts each candidate has a verdict, asserts exit codes parsed correctly)
- `pyproject.toml` (edit only if a research-time dep like `requests` is needed; explain in worker report)

Anything outside this list requires manager approval. Specifically: **do NOT modify** `src/gpuwrf/`, `fixtures/`, governance files, the M1 oracle, or other sprints' folders.

## Inputs

- Target hardware: NVIDIA RTX 5090 (cc120, 32 GB), driver verifiable with `nvidia-smi`.
- Existing env script: `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh` (CUDA + NVHPC paths).
- `PROJECT_PLAN.md §5` (candidate descriptions).
- `.agent/milestones/ROADMAP.md M2` (candidate-failure schema; the scout uses the same schema for `blocked` verdicts).
- `PERFORMANCE_TARGETS.md` (informational — this sprint produces no profiler artifacts, but matrix entries should reference the metrics that *will* be needed in S2-S7).

## Acceptance Criteria

All must hold for closeout.

### Matrix structure

1. `artifacts/m2/scout/toolchain_support_matrix.json` exists and validates against this schema:
   ```json
   {
     "generated_utc": "ISO-8601",
     "target_hardware": {"gpu_model": "...", "compute_capability": "12.0", "driver_version": "..."},
     "candidates": [
       {
         "name": "jax|triton|gt4py|kokkos|cupy_or_numba|cuda_tile",
         "verdict": "go|go-with-version-bump|blocked",
         "rationale": "≤200-char one-liner with primary evidence citation (URL or filename)",
         "version_pin": "PyPI/conda spec or git ref; null if blocked",
         "install_command": "shell string; null if blocked",
         "hello_gpu_passed": true,
         "hello_gpu_artifact_dir": "artifacts/m2/scout/hello_gpu/<candidate>/",
         "known_gaps": ["short bullet strings; empty list OK"]
       }
     ]
   }
   ```
2. Exactly six entries in `candidates[]`, one per family.

### Hello-GPU evidence

3. For every candidate with verdict ∈ {`go`, `go-with-version-bump`}: directory `artifacts/m2/scout/hello_gpu/<candidate>/` contains: a runnable program (`.py` or `.cpp`+`build.sh`), `output.txt` (stdout+stderr of the run), `exit.txt` (single integer exit code).
4. For every `go`/`go-with-version-bump` candidate: `output.txt` contains evidence the device array was actually computed on the GPU (e.g. `array([2.0, 4.0, 6.0, ...])` from device after `× 2`). Mere "import succeeded" is not enough.
5. For every `blocked` candidate: no hello_gpu/<candidate>/ directory required, but the matrix entry's `rationale` must cite a specific reason (e.g. "jaxlib 0.4.X cuda120 wheels not yet published; only 0.4.Y supports cc<120" with URL).

### Narrative report

6. `artifacts/m2/scout/toolchain_report.md` ≤2000 words covers:
   - Target hardware summary (driver, CUDA, NVHPC).
   - Per-candidate (in this fixed order: jax, triton, gt4py, kokkos, cupy_or_numba, cuda_tile): version pinned, install command, hello-GPU result, gaps, verdict.
   - Closing recommendation: which order should M2-S2..S7 dispatch in, given readiness?

### Idempotence

7. `bash scripts/m2_scout_hello_gpu.sh` re-runs every candidate's smoke test from the existing install. Output unchanged on the second run (compare via `diff` of output.txt before and after).

### Test suite

8. `tests/test_m2_scout_matrix.py`:
   - Validates the matrix JSON against the schema (positive test).
   - Asserts each of the six candidates appears exactly once.
   - For every `go` candidate, asserts `exit.txt` contains `0` and `output.txt` is non-empty.
   - Negative test: corrupted matrix JSON is rejected.
9. `pytest -q` passes overall.

### CI / hygiene

10. `python scripts/validate_agentos.py` passes.
11. `python scripts/check_m1_done.py` returns `ok: true` (no regression).
12. No file outside the File Ownership list modified.
13. No file >100 KB committed beyond pre-existing PDFs.

## Validation Commands

```bash
nvidia-smi | head -8                                             # confirm hardware
python scripts/validate_agentos.py
bash scripts/m2_scout_hello_gpu.sh                              # idempotent
python -c "import json; m=json.load(open('artifacts/m2/scout/toolchain_support_matrix.json')); print(len(m['candidates']),'candidates,', sum(1 for c in m['candidates'] if c['verdict']=='go'),'go,', sum(1 for c in m['candidates'] if c['verdict']=='blocked'),'blocked')"
pytest -q
python scripts/check_m1_done.py
git diff --stat $(git rev-parse HEAD)
```

## Performance Metrics

Not applicable. This is a research-scout sprint. Hello-GPU is a *functional* smoke test, not a performance benchmark.

## Proof Object

- Diff limited to File Ownership paths.
- `toolchain_support_matrix.json` covering all 6 candidates.
- `toolchain_report.md` narrative.
- Per-candidate hello_gpu/ directories (where applicable).
- `m2_scout_hello_gpu.sh` idempotent CLI.
- Tests.
- Standard lifecycle reports: worker, tester (Claude Opus), reviewer, manager-closeout, memory-patch.

## Risks

- **Blackwell may have spotty support across some candidate families.** That's exactly what this sprint is *measuring*. A `blocked` verdict is a valid outcome — it produces a candidate-failure-equivalent artifact that ADR-001 will consume.
- **Some candidates require root install or system libraries** (Kokkos via apt, HDF5 for some build paths). Worker uses user-mode installs only (pip/conda/local CMake build); if root is needed, candidate gets `blocked` with rationale "requires system-wide installation outside sprint sandbox."
- **CuPy vs Numba choice.** Worker decides between CuPy and Numba CUDA based on which has cleaner Blackwell wheels. Either is fine for M2.
- **Disk usage.** Per-candidate venvs may total several GB. They live under `data/scratch/m2-scout-venv/` (gitignored, on `/mnt/data`). Worker monitors `df -h /mnt/data` and writes a BLOCKER if free space drops below 50 GB.
- **Time budget.** Codex `high` should fit in 30 min for this scout. If it stalls trying every variant, worker should produce a partial matrix with `unknown` verdicts and reviewer escalates.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m2-scout-blackwell-toolchain`.
- After reviewer Accept, manager merges branch → main, then opens the highest-priority `go`-verdict candidate's S2 sprint per the matrix's closing recommendation.
- If three or more candidates are `blocked`, manager calls a `critical-review` to decide whether to escalate to user or proceed with a shrunken bakeoff. Per manager-autonomy directive, default is proceed with a documented shrunken bakeoff.

## Note: cross-AI tester

This sprint's **tester is Claude Opus 4.7** (not codex), per 2026-05-19 cross-AI verification directive. The tester's job: independently re-run `m2_scout_hello_gpu.sh` from a clean shell, spot-check the cited version pins against current upstream releases (does the cited URL still resolve? does the cited version still exist on PyPI?), and verify that no Blackwell-specific claim is hallucinated.
