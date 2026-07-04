# Release Notes — v0.23.2

A small **operational ergonomics fix** to the F1 batched-ensemble distinct-init path.
It changes **no dynamics, no physics, and no default numerical result** — the batched
math is unchanged; only the parsing of the input-dir list and its error message change.

## The fix — `GPUWRF_BATCH_INPUT_DIRS` separator

The distinct-init batched ensemble (`GPUWRF_BATCH_ENSEMBLE=B` with a list of `B` input
directories in `GPUWRF_BATCH_INPUT_DIRS`) originally split the list **only on the OS path
separator (`:`)**. An operator's natural **comma**-separated list therefore parsed as a
single directory and failed with the confusing `requires N input dirs, got 1` — which forced
a live run (ALISIOS) to fall back to sequential day segments.

- `GPUWRF_BATCH_INPUT_DIRS` now accepts a **comma**, a newline, **or** `:` between the
  directories (backward-compatible: the old `:`-only form still works).
- A wrong count now fails closed with an **actionable** message that names the accepted
  separators and shows a copy-paste example.

## The distinct-init CLI contract (now documented + verified)

Batched ensembles run on the **nested path** (`--max-dom > 1`), **one-way**:

```bash
export GPUWRF_BATCH_ENSEMBLE=4
export GPUWRF_BATCH_INPUT_DIRS="/cases/2025-01-21,/cases/2025-01-22,/cases/2025-01-23,/cases/2025-01-24"
python -m gpuwrf.cli run \
    --input-dir   /cases/2025-01-21 \
    --output-dir  runs/batch \
    --max-dom     3 --hours 24 \
    --scratch-dir /fast/nvme/scratch
```

- Provide **exactly `B`** directories.
- All must be the **same grid, physics suite and timestep** — only the initial/boundary
  **data** (the day) differs. A heterogeneous lane fails closed with a named reason.
- If `GPUWRF_BATCH_INPUT_DIRS` is unset, all `B` lanes reuse the single `--input-dir` (a
  replicated batch, useful for perturbation / bit-identity work).
- The run payload records the resolved `input_dirs` and per-lane `run_starts`.

Full guidance: [User's Guide → Performance → Batched ensemble → Distinct-init batches](https://wrf-gpu.github.io/wrf_gpu/performance.html#batched-ensemble-distinct-init).

## Validation

- New CPU unit tests (`tests/test_v0232_batch_input_dirs.py`): comma / newline / `:` /
  whitespace all resolve to the right dirs; count-mismatch raises an actionable message;
  unset replicates the input dir; `B=1` is a single dir. All pass.
- Only `src/gpuwrf/integration/nested_pipeline.py::_resolve_batch_input_dirs` changed in
  `src/`; the batched vmap orchestration, dynamics, and physics are untouched. No masking.
