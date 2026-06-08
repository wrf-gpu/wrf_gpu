# v0.14 Exact-Branch Memory Preflight

Date: 2026-06-08
Worker: GPT xhigh
Branch inspected: `worker/gpt/v013-close-manager`
Write scope: proof script/report JSON/MD plus this review only.

## Objective

Implement and run the v0.14 exact-branch memory preflight before any long
validation. Confirm the current branch contains RRTMG leading-column tiling and
the nested allocator/segmentation controls, then attempt only a short bounded
memory exercise through `scripts/run_gpu_lowprio.sh`.

## Outcome

Static exact-branch audit passes:

- RRTMG SW/LW leading-column tiling is present and default-enabled with
  `*_COLUMN_TILE_COLS=16384`.
- Nested allocator controls are present: CLI nested re-exec to
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, nested pipeline `setdefault`, and
  output-interval segmentation with resumable carries/own-steps.
- Prior proof artifacts are readable, including
  `proofs/v013/rrtmg_column_tile_vram_suite.json` and
  `proofs/v0120/nested_oom_fix.json`.

Short GPU nested smoke did not complete inside the 600 s cap. It was launched
via `scripts/run_gpu_lowprio.sh`; no TOST or long validation was started. The
run reached the cap with no wrfout/proof payload and no OOM marker observed.
External polling during the attempt saw peak total VRAM about `3204 MiB` from a
baseline of `1539 MiB`, so this did not expose a memory-pressure failure. The
first attempt also exposed and fixed a timeout-handler bug in the preflight
script.

Formal verdict in the proof JSON is `NO_RUN_PLAN`, not a completed
`PASS_SHORT_GPU_PREFLIGHT`, because there is no `nested_pipeline_run.json` and
no output file from the nested smoke.

## Files Changed

- `proofs/v014/exact_branch_memory_preflight.py`
- `proofs/v014/exact_branch_memory_preflight.json`
- `proofs/v014/exact_branch_memory_preflight.md`
- `.agent/reviews/2026-06-08-v014-exact-branch-memory-preflight.md`

## Commands Run

- `sed -n` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`,
  `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`,
  `docs/GPU_RUNBOOK.md`, `scripts/run_gpu_lowprio.sh`, and current v0.14
  contracts/reviews.
- `python -m json.tool proofs/v013/rrtmg_column_tile_vram_suite.json`
- `rg`/`sed` audits of RRTMG tiling and nested allocator/segmentation source.
- `nvidia-smi` and `ps` GPU-idle/process checks.
- `python -m py_compile proofs/v014/exact_branch_memory_preflight.py`
- `python proofs/v014/exact_branch_memory_preflight.py`
- `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v014/exact_branch_memory_preflight.py --run-gpu --timeout-s 600`
- `scripts/run_gpu_lowprio.sh --cores 0-23 -- python proofs/v014/exact_branch_memory_preflight.py --run-gpu --timeout-s 300`
- `python proofs/v014/exact_branch_memory_preflight.py --observed-timeout-run-root /mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260608T223250Z --observed-timeout-s 600 --observed-timeout-peak-total-vram-mib 3204 --observed-timeout-baseline-total-vram-mib 1539`

## Proof Objects Produced

- `proofs/v014/exact_branch_memory_preflight.json`
- `proofs/v014/exact_branch_memory_preflight.md`

Key proof facts:

- `branch_controls.ok: true`
- `rrtmg_column_tiling_present: true`
- `nested_allocator_controls_present: true`
- observed timed-out GPU attempt:
  - command used `scripts/run_gpu_lowprio.sh`
  - run root `/mnt/data/wrf_gpu_validation/v014_exact_branch_memory_preflight_20260608T223250Z`
  - 600 s cap reached
  - peak total VRAM observed `3204 MiB`
  - baseline total VRAM observed `1539 MiB`
  - output count `0`
  - no OOM observed

## Risks

- The short nested smoke did not complete, so this is not a full exact-branch
  memory-fit proof for the representative nested configuration.
- No full transfer audit was run. Hourly wrfout payload preparation is expected
  to move output data to host; no no-transfer-in-loop claim is made here.
- The nested platform allocator path does not provide useful JAX
  `memory_stats`; peak VRAM relies on `nvidia-smi` sampling.
- A retry immediately after the timeout did not start because a single
  `nvidia-smi` sample reported transient GPU utilization above the conservative
  idle threshold, despite no non-desktop compute process.

## Next Memory-Measurement Sprint

Run `V014-MEM-1` as the next memory sprint: empirical memory map on the exact
post-grid-parity branch. Measure MYNN BouLac liveness, non-radiation column
physics peaks, post-physics merge transients, and moisture limiter/advection
scratch. Keep it measurement-only; no semantic memory fixes in that sprint.

No current evidence shows a memory OOM blocker for short targeted validation,
but a completed exact-branch short nested preflight is still required before any
long validation campaign is used as evidence.
