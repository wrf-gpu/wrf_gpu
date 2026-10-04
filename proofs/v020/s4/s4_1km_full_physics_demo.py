#!/usr/bin/env python3
"""Full-physics 1 km fp32 capability demo harness for v0.20 S4.

This is a proof-only driver.  It constructs a large, horizontally homogeneous
1 km GridSpec and a physically bounded hydrostatic initial state, then advances
the production operational runtime with Thompson + MYNN/surface + RRTMG enabled.
The model code is not patched; RRTMG column tiling is controlled only by the
runtime environment, especially:

    GPUWRF_RRTMG_LW_COLUMN_TILE_COLS
    GPUWRF_RRTMG_SW_COLUMN_TILE_COLS

Each invocation runs one precision regime in a fresh process so JAX peak-memory
counters describe that regime only.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    run_forecast_operational,
    run_forecast_operational_segmented,
)


configure_jax_x64()

R_DRY_AIR = 287.0
CP_DRY_AIR = 1004.0
CV_DRY_AIR = CP_DRY_AIR - R_DRY_AIR
P0_PA = 100000.0
THETA0_K = 300.0
P_SURFACE_PA = 100000.0
P_TOP_PA = 10000.0
DOMAIN_HEIGHT_M = 18000.0


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _build_grid(nz: int, ny: int, nx: int, *, dx_m: float) -> GridSpec:
    eta_levels = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.5, float(dx_m), float(dx_m), nx, ny)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    terrain = TerrainProvenance(
        source_path="proofs/v020/s4/s4_1km_full_physics_demo.py:synthetic-1km",
        sha256="synthetic-homogeneous-canary-1km",
        shape=(ny, nx),
        units="m",
        projection_transform="flat-canary-representative",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, float(DOMAIN_HEIGHT_M), eta_levels)
    bc = BCMetadata(
        source="periodic-proof-no-boundary",
        fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999,
        interpolation="none",
        restart_compatible=False,
    )

    base = DycoreMetrics.flat(
        ny=ny,
        nx=nx,
        nz=nz,
        eta_levels=eta_levels,
        top_pressure_pa=float(P_TOP_PA),
        provenance="synthetic-1km-flat",
    )
    eta_mass = 0.5 * (eta_levels[:-1] + eta_levels[1:])
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    metrics = DycoreMetrics(
        msftx=base.msftx,
        msfty=base.msfty,
        msfux=base.msfux,
        msfuy=base.msfuy,
        msfvx=base.msfvx,
        msfvy=base.msfvy,
        c1h=one_h,
        c2h=zero_h,
        c3h=eta_mass,
        c4h=zero_h,
        c1f=one_f,
        c2f=zero_f,
        c3f=eta_levels,
        c4f=zero_f,
        dn=base.dn,
        dnw=base.dnw,
        rdn=base.rdn,
        rdnw=base.rdnw,
        cf1=base.cf1,
        cf2=base.cf2,
        cf3=base.cf3,
        fnm=base.fnm,
        fnp=base.fnp,
        dzdx=base.dzdx,
        dzdy=base.dzdy,
        dzdx_u=base.dzdx_u,
        dzdy_v=base.dzdy_v,
        f=base.f,
        e=base.e,
        sina=base.sina,
        cosa=base.cosa,
        p_top=base.p_top,
        provenance="synthetic-1km-pure-sigma",
    )
    return GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta_levels,
        terrain_height=terrain_height,
        metrics=metrics,
        halo_width=2,
        staggering="c-grid",
    )


def _zeros_state(grid: GridSpec, *, mp_physics: int = 8) -> State:
    device = jax.local_devices()[0]
    fields = {
        field: jax.device_put(
            jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(field)),
            device,
        )
        for field, shape in _state_field_shapes(grid, mp_physics=mp_physics).items()
    }
    return State(**fields)


def _zeros_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    device = jax.local_devices()[0]

    def z(shape: tuple[int, ...], field: str) -> jax.Array:
        return jax.device_put(jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(field)), device)

    return Tendencies(
        u=z((nz, ny, nx + 1), "u"),
        v=z((nz, ny + 1, nx), "v"),
        w=z((nz + 1, ny, nx), "w"),
        theta=z((nz, ny, nx), "theta"),
        qv=z((nz, ny, nx), "qv"),
        p=z((nz, ny, nx), "p"),
        ph=z((nz + 1, ny, nx), "ph"),
        mu=z((ny, nx), "mu"),
    )


def _make_state(grid: GridSpec, *, force_fp64: bool) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    state = _zeros_state(grid)

    eta = np.asarray(grid.eta_levels, dtype=np.float64)
    eta_mass = 0.5 * (eta[:-1] + eta[1:])
    mu_val = float(P_SURFACE_PA - P_TOP_PA)
    mu_total = np.full((ny, nx), mu_val, dtype=np.float64)

    p_mass_col = eta_mass * mu_val + float(P_TOP_PA)
    p_total = np.broadcast_to(p_mass_col[:, None, None], (nz, ny, nx)).copy()
    theta_col = THETA0_K + 20.0 * (1.0 - eta_mass)
    theta = np.broadcast_to(theta_col[:, None, None], (nz, ny, nx)).copy()

    dnw = eta[1:] - eta[:-1]
    alpha = (R_DRY_AIR / P0_PA) * theta_col * (p_mass_col / P0_PA) ** (-CV_DRY_AIR / CP_DRY_AIR)
    ph_col = np.zeros(nz + 1, dtype=np.float64)
    for k in range(nz):
        ph_col[k + 1] = ph_col[k] - dnw[k] * mu_val * alpha[k]
    ph_total = np.broadcast_to(ph_col[:, None, None], (nz + 1, ny, nx)).copy()

    qv_col = 0.012 * np.exp(-4.0 * (1.0 - eta_mass)) + 1.0e-5
    qv = np.broadcast_to(qv_col[:, None, None], (nz, ny, nx)).copy()
    qke = np.full((nz, ny, nx), 0.05, dtype=np.float64)
    ni = np.full((nz, ny, nx), 1.0e5, dtype=np.float64)
    nr = np.full((nz, ny, nx), 1.0e4, dtype=np.float64)

    surface = (ny, nx)
    state = state.replace(
        u=jnp.full((nz, ny, nx + 1), 5.0, dtype=jnp.float64),
        v=jnp.full((nz, ny + 1, nx), 1.0, dtype=jnp.float64),
        theta=jnp.asarray(theta),
        qv=jnp.asarray(qv),
        mu_total=jnp.asarray(mu_total),
        p_total=jnp.asarray(p_total),
        ph_total=jnp.asarray(ph_total),
        Ni=jnp.asarray(ni),
        Nr=jnp.asarray(nr),
        qke=jnp.asarray(qke),
        ustar=jnp.full(surface, 0.25, dtype=jnp.float64),
        rhosfc=jnp.full(surface, 1.15, dtype=jnp.float64),
        t_skin=jnp.full(surface, 290.0, dtype=jnp.float64),
        soil_moisture=jnp.full(surface, 0.25, dtype=jnp.float64),
        xland=jnp.ones(surface, dtype=jnp.float64),
        lakemask=jnp.zeros(surface, dtype=jnp.float64),
        mavail=jnp.full(surface, 0.5, dtype=jnp.float64),
        roughness_m=jnp.full(surface, 0.03, dtype=jnp.float64),
        lu_index=jnp.full(surface, 16, dtype=jnp.int32),
    )
    return _enforce_operational_precision(state, force_fp64=force_fp64)


def _state_dtype_fraction_fp32(state: State) -> float:
    fp32_bytes = 0
    total_bytes = 0
    for field in STATE_FIELD_ORDER:
        value = getattr(state, field, None)
        if value is None:
            continue
        nbytes = int(value.size) * int(value.dtype.itemsize)
        total_bytes += nbytes
        if value.dtype == jnp.float32:
            fp32_bytes += nbytes
    return float(fp32_bytes / total_bytes) if total_bytes else 0.0


def _peak_vram_mib() -> float | None:
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    return None if peak is None else float(peak) / (1024.0 * 1024.0)


def _stats(state: State) -> dict[str, Any]:
    nonfinite_counts = {}
    for name in state.active_field_names():
        value = getattr(state, name)
        if value is None:
            continue
        try:
            count = jnp.sum(~jnp.isfinite(value))
        except TypeError:
            continue
        nonfinite_counts[name] = count
    leaves = {
        "theta_min_k": jnp.nanmin(state.theta),
        "theta_max_k": jnp.nanmax(state.theta),
        "qv_min": jnp.nanmin(state.qv),
        "qv_max": jnp.nanmax(state.qv),
        "w_max_abs_ms": jnp.nanmax(jnp.abs(state.w)),
        "mu_min_pa": jnp.nanmin(state.mu_total),
        "mu_max_pa": jnp.nanmax(state.mu_total),
        "p_min_pa": jnp.nanmin(state.p_total),
        "p_max_pa": jnp.nanmax(state.p_total),
    }
    ready = jax.device_get(leaves)
    counts = {k: int(v) for k, v in jax.device_get(nonfinite_counts).items()}
    bad = {k: v for k, v in counts.items() if v}
    out = {k: float(v) for k, v in ready.items()}
    out["all_finite"] = not bad
    out["nonfinite_fields"] = bad
    return out


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    regime = args.regime
    force_fp64 = regime == "fp64"
    hours = float(args.steps) * float(args.dt_s) / 3600.0
    record: dict[str, Any] = {
        "proof": "v020-s4-1km-full-physics-capability",
        "regime": regime,
        "force_fp64": force_fp64,
        "aggressive_fp32": not force_fp64,
        "acoustic_precision_mode": os.environ.get("GPUWRF_ACOUSTIC_PRECISION_MODE"),
        "grid": {
            "nx": int(args.nx),
            "ny": int(args.ny),
            "nz": int(args.nz),
            "cols": int(args.nx) * int(args.ny),
            "dx_m": float(args.dx_m),
            "dy_m": float(args.dx_m),
        },
        "forecast": {
            "dt_s": float(args.dt_s),
            "steps": int(args.steps),
            "hours": hours,
            "run_physics": True,
            "run_boundary": False,
            "mp_physics": 8,
            "bl_pbl_physics": 5,
            "sf_sfclay_physics": 5,
            "cu_physics": 0,
            "ra_sw_physics": 4,
            "ra_lw_physics": 4,
            "radiation_cadence_steps": int(args.radiation_cadence_steps),
            "radiation_calls_expected": sum(
                1
                for step in range(1, int(args.steps) + 1)
                if step % int(args.radiation_cadence_steps) == 0
            ),
            "runtime_entry": (
                "run_forecast_operational_segmented"
                if int(args.segment_steps) > 0
                else "run_forecast_operational"
            ),
            "segment_steps": int(args.segment_steps) if int(args.segment_steps) > 0 else None,
        },
        "env": {
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "GPUWRF_RRTMG_SW_COLUMN_TILING": os.environ.get("GPUWRF_RRTMG_SW_COLUMN_TILING"),
            "GPUWRF_RRTMG_LW_COLUMN_TILING": os.environ.get("GPUWRF_RRTMG_LW_COLUMN_TILING"),
            "GPUWRF_RRTMG_SW_COLUMN_TILE_COLS": os.environ.get("GPUWRF_RRTMG_SW_COLUMN_TILE_COLS"),
            "GPUWRF_RRTMG_LW_COLUMN_TILE_COLS": os.environ.get("GPUWRF_RRTMG_LW_COLUMN_TILE_COLS"),
            "GPUWRF_RRTMG_LW_CLOUD_OPTICS_FP32": os.environ.get("GPUWRF_RRTMG_LW_CLOUD_OPTICS_FP32"),
        },
        "ok": False,
        "oom": False,
        "error": None,
        "wall_s": None,
        "peak_vram_mib": None,
        "state_dtype_fraction_fp32": None,
        "stability": None,
    }
    start = time.perf_counter()
    try:
        grid = _build_grid(args.nz, args.ny, args.nx, dx_m=args.dx_m)
        state = _make_state(grid, force_fp64=force_fp64)
        record["state_dtype_fraction_fp32"] = _state_dtype_fraction_fp32(state)
        namelist = OperationalNamelist.from_grid(
            grid,
            tendencies=_zeros_tendencies(grid),
            metrics=grid.metrics,
            dt_s=float(args.dt_s),
            acoustic_substeps=int(args.acoustic_substeps),
            radiation_cadence_steps=int(args.radiation_cadence_steps),
            use_vertical_solver=True,
            disable_guards=False,
            force_fp64=force_fp64,
        )
        namelist = dataclasses.replace(
            namelist,
            run_physics=True,
            run_boundary=False,
            mp_physics=8,
            bl_pbl_physics=5,
            sf_sfclay_physics=5,
            sf_surface_physics=None,
            cu_physics=0,
            ra_sw_physics=4,
            ra_lw_physics=4,
            use_noahmp=False,
        )
        if int(args.segment_steps) > 0:
            result = run_forecast_operational_segmented(
                state,
                namelist,
                hours,
                segment_steps=int(args.segment_steps),
            )
        else:
            result = run_forecast_operational(state, namelist, hours)
        jax.block_until_ready(result)
        record["stability"] = _stats(result)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        record["error"] = text
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "resource_exhausted",
                "out of memory",
                "outofmemory",
                "out-of-memory",
                "ran out of memory",
                "cuda_error_out_of_memory",
            )
        ):
            record["oom"] = True
        record["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
    finally:
        record["wall_s"] = time.perf_counter() - start
        record["peak_vram_mib"] = _peak_vram_mib()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("fp64", "aggressive"), required=True)
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dx-m", type=float, default=1000.0)
    parser.add_argument("--dt-s", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--acoustic-substeps", type=int, default=10)
    parser.add_argument("--radiation-cadence-steps", type=int, default=1)
    parser.add_argument(
        "--segment-steps",
        type=int,
        default=0,
        help="Use the production segmented forecast entry with this segment length when >0.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    record = run_one(args)
    text = json.dumps(record, indent=2, sort_keys=True, default=_json_default)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    if record["ok"] or record["oom"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
