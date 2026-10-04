# v0.23-close Validation Scripts — prep (GPT, CPU-ONLY authoring)

Read `proofs/v023/validation/V0230_CLOSE_VALIDATION_PLAN.md` fully. Author the **turnkey
validation scripts** it lists (A1–A4, B1–B5, C1–C2) under `proofs/v023/validation/`, so the
whole gauntlet runs unattended the moment the GPU frees.

## CPU-ONLY authoring — do NOT run any GPU job
`taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES=`. You **WRITE**
the scripts (they RUN on GPU later, wrapped in `scripts/with_gpu_lock.sh`). Dry-run-lint on CPU
only (`bash -n`, `python -c "compile(open(f).read(),f,'exec')"`, `--help`), and a CPU smoke of
the pure-CPU pieces (e.g. the field-compare tool on two tiny local files) — but do NOT execute
GPU forecasts. Cores 0-3 are 0:2's.

## Reuse, don't reinvent
Grep the repo for existing paired-baseline / field-compare / canary-gate tooling
(`proofs/v0*/…`, `scripts/…`, `docs/assets/…`) and reuse it. The CORE reusable piece = the
**paired-baseline field-compare tool** (A1/A2/A4): field-tolerance vs a **FRESH same-env paired
baseline**, NOT a stored digest (the stored-digest canary gate is autotune-broken — that is
exactly what A2 fixes). Parameterize every script (`CASE`, `BASE_REF`, `HEAD`). Each script:
clear header + usage, `with_gpu_lock` wrapper, warm-vs-warm, honest bands, writes a JSON verdict.

## Rules
No masking. Commit on branch `worker/gpt/v023-validation-scripts` (off `origin/main`).

## Deliverable
The scripts + `proofs/v023/validation/VAL_SCRIPTS_REPORT.md` (what each does, reuse notes,
dry-run-lint results, what remains GPU-only) + `touch proofs/v023/validation/VAL_SCRIPTS_DONE`.
Report to `0:1`. Blocked → `VAL_SCRIPTS_QUESTION.md` + ping `0:1`. Begin now.
