"""Generic leading-axis column tiling for batched column-physics kernels.

v0.15 kernel-final (Task 3, VRAM ceiling): the un-tiled microphysics adapter
was identified as the largest remaining un-capped column-physics working set
(`proofs/perf/v015/km_bench/vram_ceiling_findings.json`).  This module factors
the scan-over-fixed-size-column-tiles pattern ALREADY proven exact for RRTMG
(`rrtmg_lw._longwave_column_tiled_impl`, `proofs/v013/rrtmg_column_tile_vram_suite.json`)
and MYNN (`mynn_pbl._tiled_mynn_step`) into one reusable helper, so the
production Thompson step (and any future batched column scheme) can cap its
per-step intermediate working set to ONE tile instead of the full grid.

Identity contract
-----------------
No column-physics op couples columns: each column's result is produced by the
same per-element kernel arithmetic whichever tile contains it, so the tiled
output is value-identical per column (GPU bit-identical; gated empirically per
scheme by an exact-output check, e.g. `proofs/perf/v015/vram_mp_tiling.json`).
Tiling is a pure execution-shape change — no math, no clamps, no reordering
within a column.

Cost contract
-------------
Inputs are padded to a tile multiple by repeating the final column (a valid
physical column; its outputs are discarded), the kernel runs under `lax.scan`
over tiles, and the stacked outputs reshape back to the original leading
width.  Peak live intermediates inside the kernel shrink to one tile; the
full-width inputs/outputs are unchanged (they exist either way).
"""

from __future__ import annotations

import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "env_bool",
    "env_int",
    "pad_columns_leaf",
    "slice_columns_leaf",
    "tiled_column_apply",
]

_FALSEY = {"0", "false", "off", "no", ""}


def env_bool(name: str, default: bool) -> bool:
    """Boolean env knob with the project's standard falsey set."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSEY


def env_int(name: str, default: int) -> int:
    """Integer env knob; malformed values fall back to the default."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def pad_columns_leaf(arr: Any, ncol: int, padded_ncol: int) -> Any:
    """Pads a flattened leading column axis by repeating the last real column.

    Leaves whose leading dim is not ``ncol`` (scalars, per-level constants)
    pass through unchanged — identical contract to MYNN's `_pad_columns_leaf`.
    """

    arr = jnp.asarray(arr)
    if arr.ndim == 0 or arr.shape[0] != ncol or padded_ncol == ncol:
        return arr
    tail = jnp.broadcast_to(arr[-1:], (padded_ncol - ncol,) + arr.shape[1:])
    return jnp.concatenate((arr, tail), axis=0)


def slice_columns_leaf(arr: Any, start: jax.Array, tile_cols: int, padded_ncol: int) -> Any:
    """Slices one fixed-size column tile from a padded leading axis."""

    arr = jnp.asarray(arr)
    if arr.ndim == 0 or arr.shape[0] != padded_ncol:
        return arr
    starts = [jnp.zeros((), dtype=start.dtype)] * arr.ndim
    starts[0] = start
    sizes = list(arr.shape)
    sizes[0] = tile_cols
    return lax.dynamic_slice(arr, starts, sizes)


def tiled_column_apply(fn: Callable[[Any], Any], in_tree: Any, *, ncol: int, tile_cols: int) -> Any:
    """Applies a batched column kernel over fixed-size leading-column tiles.

    ``in_tree`` is any pytree whose per-column leaves have leading dim ``ncol``
    (other leaves are broadcast unchanged into every tile).  ``fn`` maps the
    tile-shaped tree to an output pytree ALL of whose array leaves have the
    tile width as leading dim (the contract every batched column scheme
    satisfies).  Outputs are scan-stacked and reshaped back to ``ncol``-leading
    arrays, discarding the pad columns.

    The caller decides activation (typically ``ncol > tile_cols`` with the
    scheme's env knobs) so the untiled graph stays byte-for-byte untouched
    whenever tiling is off or unnecessary.
    """

    if tile_cols <= 0:
        raise ValueError(f"tile_cols must be positive, got {tile_cols}")
    n_tiles = -(-ncol // tile_cols)
    padded_ncol = n_tiles * tile_cols
    padded = jax.tree_util.tree_map(
        lambda a: pad_columns_leaf(a, ncol, padded_ncol), in_tree
    )

    def body(carry, tile_index):
        start = tile_index * tile_cols
        tile = jax.tree_util.tree_map(
            lambda a: slice_columns_leaf(a, start, tile_cols, padded_ncol), padded
        )
        return carry, fn(tile)

    _, stacked = lax.scan(body, None, jnp.arange(n_tiles, dtype=jnp.int32))

    def _unstack(a):
        a = jnp.asarray(a)
        return a.reshape((padded_ncol,) + a.shape[2:])[:ncol]

    return jax.tree_util.tree_map(_unstack, stacked)
