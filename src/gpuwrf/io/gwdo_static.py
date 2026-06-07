"""Read WRF static sub-grid orography fields needed by orographic GWD (gwd_opt=1).

The orographic gravity-wave-drag scheme (``physics/gwd_gwdo.py``,
``coupling.physics_couplers.gwdo_adapter``) consumes the WPS/geo_em sub-grid
orography statistics that ride in ``wrfinput`` (and the history files):

    VAR        -> std dev of subgrid orography (m)        [GWDOStatics.var]
    CON        -> orographic convexity                    [GWDOStatics.oc1]
    OA1..OA4   -> directional asymmetry                   [oa1..oa4]
    OL1..OL4   -> directional effective length            [ol1..ol4]

These are the exact fields carried by ``init/metgrid_schema.py`` (Registry
``gwd_used_1``). The loader builds the per-run :class:`GWDOStatics` bundle the
operational dispatch attaches to ``OperationalNamelist.gwdo_statics``; it
mirrors :func:`gpuwrf.io.radiation_static.load_radiation_static` (best-effort,
wrfout-then-wrfinput fallback, real grid-rotation when ``metrics`` is given).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.coupling.physics_couplers import (
    GWDOStatics,
    build_gwdo_statics_from_wrf_fields,
)
from gpuwrf.io.gen2_accessor import Gen2Run

# The ten geo_em sub-grid orography fields (Registry gwd_used_1).
GWDO_STATIC_FIELDS = ("VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4")


def _grid_dx_m(grid) -> float:
    """Grid spacing (m) from either a dycore ``GridSpec`` or a ``Gen2GridSpec``."""

    projection = getattr(grid, "projection", None)
    if projection is not None and getattr(projection, "dx_m", None) is not None:
        return float(projection.dx_m)
    return float(grid.dx_m)


def _load_static_xy(run: Gen2Run, domain: str, name: str):
    """Load a static mass-grid field from the first wrfout, falling back to wrfinput.

    Returns ``(array, source_path)`` or ``(None, None)`` if the field is absent
    from both files (e.g. a WPS geo_em that did not compute the GWD statics).
    """

    try:
        return run.load(domain, name, time=0, lazy=False), str(run.history_files(domain)[0])
    except Exception:  # noqa: BLE001 -- fall back to wrfinput
        try:
            return run.load_wrfinput(domain, name, lazy=False), str(run.wrfinput_file(domain))
        except Exception:  # noqa: BLE001 -- field genuinely missing
            return None, None


def load_gwdo_statics(
    run_dir: str | Path | Gen2Run,
    domain: str,
    *,
    grid: GridSpec,
    metrics: DycoreMetrics | None = None,
) -> tuple[GWDOStatics | None, dict[str, Any]]:
    """Build the per-run GWDO sub-grid orography statics for one Gen2 domain.

    Returns ``(statics, metadata)``. ``statics`` is ``None`` (and the dispatch a
    no-op) when ``VAR`` is absent or identically zero -- a geo_em without the
    sub-grid orography fields produces no orographic gravity-wave drag in WRF
    either, so failing closed matches the reference behaviour rather than
    fabricating drag from zero statistics.
    """

    # Accept a Gen2Run (or any duck-typed accessor exposing load/load_wrfinput);
    # otherwise treat the argument as a run-directory path and open it.
    if isinstance(run_dir, (str, Path)):
        run = Gen2Run(run_dir)
    else:
        run = run_dir
    fields: dict[str, Any] = {}
    sources: dict[str, str | None] = {}
    for name in GWDO_STATIC_FIELDS:
        arr, src = _load_static_xy(run, domain, name)
        fields[name] = arr
        sources[name] = src

    var = fields["VAR"]
    metadata: dict[str, Any] = {
        "domain": domain,
        "fields_present": {name: fields[name] is not None for name in GWDO_STATIC_FIELDS},
        "sources": sources,
        "dx_m": _grid_dx_m(grid),
        "uses_dycore_map_rotation": bool(metrics is not None),
    }

    if var is None:
        metadata["status"] = "absent"
        metadata["reason"] = "VAR not in wrfout/wrfinput; GWD has no sub-grid orography to act on"
        return None, metadata

    var_arr = jnp.asarray(var)
    var_max = float(jnp.max(jnp.abs(var_arr)))
    metadata["var_abs_max"] = var_max
    if var_max <= 0.0:
        metadata["status"] = "zero_var"
        metadata["reason"] = "VAR identically zero; orographic GWD base stress is zero everywhere"
        return None, metadata

    sina = metrics.sina if metrics is not None else None
    cosa = metrics.cosa if metrics is not None else None
    statics = build_gwdo_statics_from_wrf_fields(
        var_arr,
        fields["CON"],
        fields["OA1"],
        fields["OA2"],
        fields["OA3"],
        fields["OA4"],
        fields["OL1"],
        fields["OL2"],
        fields["OL3"],
        fields["OL4"],
        dx_m=_grid_dx_m(grid),
        sina=sina,
        cosa=cosa,
    )
    metadata["status"] = "built"
    metadata["terrain_shape"] = [int(grid.ny), int(grid.nx)]
    return statics, metadata


__all__ = ["GWDO_STATIC_FIELDS", "load_gwdo_statics"]
