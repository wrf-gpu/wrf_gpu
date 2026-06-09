# Sprint Contract: V0.14 Empirical Memory Map

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Produce an implementation-ready empirical/static memory map for the remaining
non-radiation memory risks on the exact current branch, without source changes.

RRTMG column/band/optics tiling is already fixed and must be treated as prior
evidence, not re-litigated. The purpose is to decide which remaining memory
fixes, if any, deserve v0.14 source sprints before long validation.

## Inputs

- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/exact_branch_memory_preflight.json`
- `proofs/v013/rrtmg_column_tile_vram_suite.json`
- `proofs/v013/gpoint_chunk_rrtmg.json`
- `proofs/v013/optics_taumol_chunk.json`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/dynamics/flux_advection.py`
- relevant non-radiation physics modules only as needed

## Write Scope

Repository write scope:

- `proofs/v014/empirical_memory_map.py`
- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`
- `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`

No production `src/` edits. No WRF source edits. No GPU run. No TOST. No
Switzerland run. Do not start FP32 implementation.

## Required Work

1. Reconcile the current memory roadmap with exact-branch evidence and source
   shapes for the remaining pending items.
2. Inspect, at minimum:
   - moisture advection duplicate transport velocity construction;
   - WDM6 `slmsk` broadcast;
   - non-radiation column physics tiling candidates;
   - post-physics non-dry sparse/donated merge;
   - PBL/surface bottom-only prep and duplicated diagnostics;
   - moisture limiter/sequentialization workspace;
   - acoustic carry split / evolving-only carry as a dycore-adjacent memory
     item, but do not propose source changes before grid parity closes.
3. For each candidate, record:
   - likely concurrently materialized arrays;
   - shape formula and fp64 byte estimate for the 641x321x50 target geometry
     where applicable;
   - whether evidence is measured, HLO/source-static, or inferred;
   - correctness risk and required proof gate;
   - whether it should block v0.14 long validation.
4. Produce a short ranked recommendation:
   - `FIX_NOW_BIT_IDENTICAL`;
   - `MEASURE_FIRST`;
   - `DEFER_SEMANTIC_OR_DYCORE`;
   - `DO_NOT_DO_BEFORE_GRID_PARITY`;
   - `NOT_WORTH_STANDALONE`.
5. Keep the top-level report context-sparing. Put detailed tables in JSON.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/empirical_memory_map.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/empirical_memory_map.py
python -m json.tool proofs/v014/empirical_memory_map.json \
  >/tmp/empirical_memory_map.validated.json
```

If the analysis is fully static and no script is needed beyond report
generation, the script may encode the inspected table and validation metadata.

## Acceptance Criteria

- No production source edits.
- JSON validates and records evidence strength for every candidate.
- The report clearly separates bit-identical layout fixes from semantic/dycore
  memory changes.
- The result names the smallest safe v0.14 memory source sprint, if any.
- The result explicitly states whether any remaining memory work should block
  long validation after grid parity closes.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
