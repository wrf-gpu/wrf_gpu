# Worker Report

Summary:

Produce an implementation-ready empirical/static memory map for the remaining
non-radiation memory risks on the exact current branch.

Files changed:

- `proofs/v014/empirical_memory_map.py`
- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`
- `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`

Commands run:

- Required project-rule/source/proof reads.
- Source inspections with `rg`, `sed`, and `jq`.
- `python -m py_compile proofs/v014/empirical_memory_map.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/empirical_memory_map.py`
- `python -m json.tool proofs/v014/empirical_memory_map.json >/tmp/empirical_memory_map.validated.json`

Proof objects produced:

- `proofs/v014/empirical_memory_map.py`
- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`
- `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`

Verdict:

`NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`.

Unresolved risks:

- This is source-static plus prior-proof reconciliation, not a fresh GPU peak
  measurement or transfer audit.
- MYNN BouLac, non-radiation column tiling, post-physics merge, and moisture
  limiter liveness still require measurement before rewrite.
- Acoustic carry split and FP32 acoustic remain dycore/precision work and must
  not start before grid parity closes.

Next decision:

After grid parity closes, run the exact selected long-validation branch memory
preflight. If a memory-only source sprint is wanted first, choose WDM6 `slmsk`
shape-only cleanup; the only material bit-identical cleanup is moisture
transport velocity reuse for active moisture advection.
