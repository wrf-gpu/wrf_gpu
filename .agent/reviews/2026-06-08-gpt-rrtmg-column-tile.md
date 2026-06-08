# v0.13 RRTMG Column-Tiling Handoff

## objective

Implement leading-column tiling for the public RRTMG SW/LW operational solves so large 1 km nests no longer materialise radiation transients across the full column batch, while preserving public solver interfaces and CPU bit identity against the whole-column reference.

## files changed

- `src/gpuwrf/physics/rrtmg_sw.py`
- `src/gpuwrf/physics/rrtmg_lw.py`
- `proofs/v013/rrtmg_column_tile.py`
- `proofs/v013/rrtmg_column_tile.json`
- `.agent/reviews/2026-06-08-gpt-rrtmg-column-tile.md`

## commands run

- `PYTHONPATH=src JAX_PLATFORMS=cpu python -m py_compile src/gpuwrf/physics/rrtmg_sw.py src/gpuwrf/physics/rrtmg_lw.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_m5_rrtmg_column_shapes.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python proofs/v013/rrtmg_column_tile.py --mode inertness`
- `PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_m5_rrtmg_tier1.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python - <<'PY' ...` single-column and `(2,3,nz)` shape smoke
- `PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q tests/test_m5_rrtmg_column_shapes.py tests/test_m5_rrtmg_tier1.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu python -m py_compile src/gpuwrf/physics/rrtmg_sw.py src/gpuwrf/physics/rrtmg_lw.py proofs/v013/rrtmg_column_tile.py`

## proof objects produced

- `proofs/v013/rrtmg_column_tile.py`
- `proofs/v013/rrtmg_column_tile.json`

## bit-identity summary

CPU inertness proof used `leading_shape=(2,3)` and compared the public solver with column tiling disabled against:

- default column tiling enabled (`tile_cols=16384`, one effective tile on the small fixture)
- forced padded tiling (`tile_cols=4`, two scan tiles, two padded columns)
- SW topography plus clear-sky on the forced padded path

Verdict from `proofs/v013/rrtmg_column_tile.json`:

- SW all-sky: `max_abs=0.0`, `max_rel=0.0`
- SW with clear-sky: `max_abs=0.0`, `max_rel=0.0`
- LW all-sky: `max_abs=0.0`, `max_rel=0.0`
- LW with clear-sky: `max_abs=0.0`, `max_rel=0.0`
- all required cases bit-identical: `true`

## GPU proof command for the manager to run

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
  python proofs/v013/rrtmg_column_tile.py --mode vram-suite --ncol 65536 --nrep 3 --tile-cols 16384
```

## unresolved risks

- Manager GPU peak-VRAM proof ran after merge in `proofs/v013/rrtmg_column_tile_vram_suite.json`.
- GPU result: LW untiled OOMed on a 32.11 GiB allocation; LW tiled peak was 5374.84 MiB.
- GPU result: SW untiled peak was 10033.1 MiB; SW tiled peak was 1619.54 MiB.
- Runtime impact of `lax.dynamic_update_slice` result carries should still be measured in full forecast context; correctness and residency shape are proven.
- `tests/test_m5_rrtmg_tier1.py` rewrites `artifacts/m5/tier1_rrtmg_sw_parity.json` on this CPU path; the generated artifact churn was restored and is not part of this patch.

## next decision needed, if any

Manager should run the GPU VRAM suite and decide whether `16384` remains the production tile size or should be tuned after peak-memory and runtime evidence.
