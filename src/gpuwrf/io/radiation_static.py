"""Read WRF static fields needed by RRTMG topographic radiation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.coupling.physics_couplers import (
    RRTMGRadiationStatic,
    build_radiation_static_from_wrf_fields,
)
from gpuwrf.io.gen2_accessor import Gen2Run


def _load_static_xy(run: Gen2Run, domain: str, name: str):
    """Load a static mass-grid field from the first wrfout, falling back to wrfinput."""

    try:
        return run.load(domain, name, time=0, lazy=False), str(run.history_files(domain)[0])
    except Exception:
        return run.load_wrfinput(domain, name, lazy=False), str(run.wrfinput_file(domain))


def load_radiation_static(
    run_dir: str | Path | Gen2Run,
    domain: str,
    *,
    grid: GridSpec,
    metrics: DycoreMetrics | None = None,
) -> tuple[RRTMGRadiationStatic, dict[str, Any]]:
    """Build resident RRTMG topo-radiation static fields for one Gen2 domain."""

    run = run_dir if isinstance(run_dir, Gen2Run) else Gen2Run(run_dir)
    xlat, xlat_source = _load_static_xy(run, domain, "XLAT")
    xlong, xlong_source = _load_static_xy(run, domain, "XLONG")
    msftx = metrics.msftx if metrics is not None else None
    msfty = metrics.msfty if metrics is not None else None
    sina = metrics.sina if metrics is not None else None
    cosa = metrics.cosa if metrics is not None else None
    static = build_radiation_static_from_wrf_fields(
        jnp.asarray(xlat),
        jnp.asarray(xlong),
        jnp.asarray(grid.terrain_height),
        dx_m=float(grid.projection.dx_m),
        dy_m=float(grid.projection.dy_m),
        msftx=msftx,
        msfty=msfty,
        sina=sina,
        cosa=cosa,
    )
    metadata = {
        "domain": domain,
        "xlat_source": xlat_source,
        "xlong_source": xlong_source,
        "terrain_source": grid.terrain.source_path,
        "terrain_shape": [int(grid.ny), int(grid.nx)],
        "dx_m": float(grid.projection.dx_m),
        "dy_m": float(grid.projection.dy_m),
        "slope_aspect_ref": "WRF dyn_em/start_em.F slope/slp_azi initialization",
        "uses_real_xlat_xlong": True,
        "uses_dycore_map_rotation": bool(metrics is not None),
    }
    return static, metadata


__all__ = ["load_radiation_static"]
