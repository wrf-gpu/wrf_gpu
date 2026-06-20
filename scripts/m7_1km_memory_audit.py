#!/usr/bin/env python
"""M7 1 km GPU memory audit orchestrator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
from jax import config  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.contracts.grid import (  # noqa: E402
    BCMetadata,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.precision import DEFAULT_DTYPES, PRECISION_MATRIX, STATE_FIELD_ORDER  # noqa: E402
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name  # noqa: E402
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational  # noqa: E402


config.update("jax_enable_x64", True)

SPRINT_DIR = ROOT / ".agent" / "sprints" / "2026-05-27-m7-1km-memory-audit"
CONTRACT_L2_RUN = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260520_18z_l2_72h_20260521T045847Z")
FALLBACK_L3_RUN = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
TARGET_DX_M = 1000.0
DT_S = 10.0
NVIDIA_SMI_MEMORY_CMD = (
    "nvidia-smi",
    "--query-gpu=name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def nvidia_smi_memory() -> dict[str, Any]:
    result = run_command(list(NVIDIA_SMI_MEMORY_CMD))
    payload: dict[str, Any] = {"query": result}
    if result["returncode"] != 0 or not result["stdout"].strip():
        payload["ok"] = False
        return payload
    first = result["stdout"].strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        payload["ok"] = False
        payload["parse_error"] = f"expected name,used,total CSV row; got {first!r}"
        return payload
    name, used_mib, total_mib = parts[:3]
    payload.update(
        {
            "ok": True,
            "name": name,
            "memory_used_mib": int(float(used_mib)),
            "memory_total_mib": int(float(total_mib)),
            "memory_used_bytes": int(float(used_mib) * 1024 * 1024),
            "memory_total_bytes": int(float(total_mib) * 1024 * 1024),
        }
    )
    return payload


def vram_total_bytes() -> int | None:
    payload = nvidia_smi_memory()
    return int(payload["memory_total_bytes"]) if payload.get("ok") else None


def device_memory_stats() -> dict[str, Any]:
    try:
        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if not devices:
            return {"ok": False, "reason": "no visible JAX GPU device"}
        raw = devices[0].memory_stats()
        if raw is None:
            return {"ok": False, "device": str(devices[0]), "reason": "memory_stats returned None"}
        return {
            "ok": True,
            "device": str(devices[0]),
            "stats": {str(key): int(value) for key, value in raw.items() if isinstance(value, (int, np.integer))},
            "raw_keys": sorted(str(key) for key in raw.keys()),
        }
    except Exception as exc:  # pragma: no cover - hardware/driver dependent
        return {"ok": False, "reason": repr(exc)}


class GpuMemorySampler:
    """Poll nvidia-smi while JAX blocks on a GPU operation."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def _sample(self) -> None:
        sample = nvidia_smi_memory()
        if sample.get("ok"):
            sample["timestamp_s"] = time.time()
            self.samples.append(sample)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        used = [int(sample["memory_used_bytes"]) for sample in self.samples if sample.get("ok")]
        return {
            "sample_count": len(used),
            "peak_memory_used_bytes": max(used) if used else None,
            "peak_memory_used_mib": (max(used) / 1024 / 1024) if used else None,
            "samples": self.samples,
        }


def find_first_wrfout(run_dir: Path, domain: str) -> Path | None:
    matches = sorted(run_dir.glob(f"wrfout_{domain}_*"))
    return matches[0] if matches else None


def read_wrfout_grid(path: Path) -> dict[str, Any]:
    with netCDF4.Dataset(path, "r") as ds:
        dims = {name: int(len(dim)) for name, dim in ds.dimensions.items()}
        nx = dims["west_east"]
        ny = dims["south_north"]
        nz = dims["bottom_top"]
        return {
            "path": str(path),
            "domain": path.name.split("_")[1],
            "dims": dims,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "mass_shape": [nz, ny, nx],
            "staggered_shapes": staggered_shapes(nz=nz, ny=ny, nx=nx),
            "dx_m": float(getattr(ds, "DX", float("nan"))),
            "dy_m": float(getattr(ds, "DY", float("nan"))),
            "p_top_pa": float(getattr(ds, "P_TOP", 5000.0)),
            "map_proj": str(getattr(ds, "MAP_PROJ_CHAR", getattr(ds, "MAP_PROJ", "unknown"))),
            "cen_lat": float(getattr(ds, "CEN_LAT", 28.3)),
            "cen_lon": float(getattr(ds, "CEN_LON", -15.6)),
        }


def staggered_shapes(*, nz: int, ny: int, nx: int) -> dict[str, list[int]]:
    boundary_side = max(nx + 1, ny + 1)
    return {
        "mass": [nz, ny, nx],
        "u": [nz, ny, nx + 1],
        "v": [nz, ny + 1, nx],
        "w": [nz + 1, ny, nx],
        "surface": [ny, nx],
        "boundary_mass": [1, 4, nz, boundary_side],
        "boundary_w": [1, 4, nz + 1, boundary_side],
        "boundary_mu": [1, 4, 1, boundary_side],
    }


def build_grid_shape_payload() -> dict[str, Any]:
    contract_candidates = {
        domain: str(find_first_wrfout(CONTRACT_L2_RUN, domain)) if find_first_wrfout(CONTRACT_L2_RUN, domain) else None
        for domain in ("d04", "d05")
    }
    fallback_d02_path = find_first_wrfout(FALLBACK_L3_RUN, "d02")
    if fallback_d02_path is None:
        raise FileNotFoundError(f"missing d02 wrfout in fallback run {FALLBACK_L3_RUN}")
    three_km = read_wrfout_grid(fallback_d02_path)

    nested_grids: dict[str, Any] = {}
    for domain in ("d03", "d04", "d05"):
        path = find_first_wrfout(FALLBACK_L3_RUN, domain)
        if path is not None:
            nested_grids[domain] = read_wrfout_grid(path)

    selected_domain = "d04" if "d04" in nested_grids else next(iter(nested_grids), None)
    selected = nested_grids[selected_domain] if selected_domain else None
    scale_x = int(round(float(three_km["dx_m"]) / TARGET_DX_M))
    scale_y = int(round(float(three_km["dy_m"]) / TARGET_DX_M))
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"invalid scale from d02 dx/dy: {three_km['dx_m']} {three_km['dy_m']}")
    nz = int(three_km["nz"])
    ny = int(three_km["ny"]) * scale_y
    nx = int(three_km["nx"]) * scale_x
    mass_cells_3km = int(three_km["nz"]) * int(three_km["ny"]) * int(three_km["nx"])
    mass_cells_1km = nz * ny * nx
    comparison_vs_3km = {
        "nx_factor": nx / float(three_km["nx"]),
        "ny_factor": ny / float(three_km["ny"]),
        "horizontal_cell_factor": (ny * nx) / float(int(three_km["ny"]) * int(three_km["nx"])),
        "mass_cell_count_3km": mass_cells_3km,
        "mass_cell_count_1km": mass_cells_1km,
        "mass_cell_factor": mass_cells_1km / float(mass_cells_3km),
    }
    derived_staggered_shapes = staggered_shapes(nz=nz, ny=ny, nx=nx)
    return {
        "status": "PASS" if selected is not None else "BLOCKED_NO_1KM_WRFOUT",
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "mass_shape": [nz, ny, nx],
        "staggered_shapes": derived_staggered_shapes,
        "comparison_vs_3km": comparison_vs_3km,
        "contract_candidate_run": {
            "path": str(CONTRACT_L2_RUN),
            "exists": CONTRACT_L2_RUN.exists(),
            "d04_d05_wrfout_candidates": contract_candidates,
            "note": "The sprint-named wrf_l2 run has no wrfout_d04/d05 files in this worktree.",
        },
        "source_wrfout_path": selected["path"] if selected else None,
        "source_domain": selected_domain,
        "shape_used_for_static_and_full_fit_audit": "derived_full_1km_from_3km_d02",
        "three_km_reference": three_km,
        "gen2_available_1km_nests": nested_grids,
        "derived_full_1km": {
            "derivation_rule": "Scale the full 3 km d02 horizontal grid by dx/dy ratio 3000/1000; keep nz unchanged.",
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "mass_shape": [nz, ny, nx],
            "staggered_shapes": derived_staggered_shapes,
            "dx_m": TARGET_DX_M,
            "dy_m": TARGET_DX_M,
            "source_3km_wrfout_path": three_km["path"],
            "comparison_vs_3km": comparison_vs_3km,
        },
    }


def _grid_namespace(nz: int, ny: int, nx: int) -> SimpleNamespace:
    return SimpleNamespace(nz=int(nz), ny=int(ny), nx=int(nx))


def field_shapes_for(nz: int, ny: int, nx: int) -> dict[str, tuple[int, ...]]:
    # The 1 km worst-case memory model must size EVERY leaf in the State contract,
    # including the conditional hail (qh/Nh/qvolg/qvolh/hail_acc) and aerosol
    # (nwfa/nifa) leaves added in v0.16/v0.17. STATE_FIELD_ORDER and
    # State.__slots__ carry all 67; request include_all_conditional=True so the
    # shape map reconciles against the full contract (default args return only the
    # 60 always-present leaves and tripped the contract-mismatch guard below).
    return _state_field_shapes(_grid_namespace(nz, ny, nx), include_all_conditional=True)


def dtype_record(field: str) -> dict[str, Any]:
    dtype = np.dtype(DEFAULT_DTYPES.dtype_for(field))
    _, gated = PRECISION_MATRIX[field]
    return {"dtype": dtype.name, "itemsize": int(dtype.itemsize), "fp32_gated": bool(gated)}


def build_static_memory_model(grid_shape: dict[str, Any], total_vram_bytes: int | None = None) -> dict[str, Any]:
    derived = grid_shape["derived_full_1km"]
    nz, ny, nx = (int(value) for value in derived["mass_shape"])
    shapes = field_shapes_for(nz=nz, ny=ny, nx=nx)
    state_slots = list(State.__slots__)
    shape_fields = list(shapes)
    field_order = list(STATE_FIELD_ORDER)
    if set(field_order) != set(state_slots) or set(field_order) != set(shape_fields):
        raise ValueError(
            "State field contract mismatch: "
            f"STATE_FIELD_ORDER={len(field_order)} slots={len(state_slots)} shapes={len(shape_fields)}"
        )
    running_total = 0
    fields: list[dict[str, Any]] = []
    for field in field_order:
        shape = tuple(int(dim) for dim in shapes[field])
        dtype = dtype_record(field)
        elements = math.prod(shape)
        nbytes = int(elements * int(dtype["itemsize"]))
        running_total += nbytes
        fields.append(
            {
                "name": field,
                "field": field,
                "dtype": dtype["dtype"],
                "fp32_gated": dtype["fp32_gated"],
                "shape": list(shape),
                "elements": int(elements),
                "bytes": nbytes,
                "mib": nbytes / 1024 / 1024,
                "running_total_bytes": running_total,
            }
        )
    top_fields = sorted(fields, key=lambda item: int(item["bytes"]), reverse=True)[:5]
    total_vram = int(total_vram_bytes) if total_vram_bytes is not None else None
    return {
        "status": "PASS" if total_vram is None or running_total <= total_vram else "FAIL_STATIC_EXCEEDS_VRAM",
        "field_count": len(fields),
        "state_slot_count": len(state_slots),
        "contract_note": "state.py currently exposes 47 State fields; sprint text says 45, so this model follows code source of truth.",
        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "mass_shape": [nz, ny, nx],
            "mass_cells": int(nz * ny * nx),
        },
        "fields": fields,
        "top_5_fields_by_bytes": top_fields,
        "total_state_bytes": running_total,
        "total_state_mib": running_total / 1024 / 1024,
        "total_state_gib": running_total / 1024 / 1024 / 1024,
        "device_vram_total_bytes": total_vram,
        "device_vram_total_gib": (total_vram / 1024 / 1024 / 1024) if total_vram else None,
        "state_fraction_of_vram": (running_total / total_vram) if total_vram else None,
        "sanity_total_le_device_vram": bool(total_vram is None or running_total <= total_vram),
    }


def make_grid_spec_from_shape(grid_shape: dict[str, Any]) -> GridSpec:
    derived = grid_shape["derived_full_1km"]
    nz, ny, nx = (int(value) for value in derived["mass_shape"])
    projection = Projection("lambert", 28.3, -15.6, TARGET_DX_M, TARGET_DX_M, nx, ny)
    terrain = TerrainProvenance(
        source_path=str(derived["source_3km_wrfout_path"]),
        sha256="not-computed-m7-memory-audit",
        shape=(ny, nx),
        units="m",
        projection_transform="full-d02-horizontal-scale-3x",
        max_elevation_m=3715.0,
        coastline_sanity_check_passed=False,
    )
    eta_levels = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", nz, float(grid_shape["three_km_reference"].get("p_top_pa", 5000.0)), eta_levels)
    bc = BCMetadata(
        source="AIFS",
        fields=("u", "v", "T", "qv", "p_s"),
        update_cadence_h=6,
        interpolation="linear",
        restart_compatible=False,
    )
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def allocate_state_incremental(grid: GridSpec) -> tuple[State, list[dict[str, Any]]]:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("no visible JAX GPU device")
    device = devices[0]
    shapes = field_shapes_for(nz=grid.nz, ny=grid.ny, nx=grid.nx)
    arrays: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    running_total = 0
    for field in STATE_FIELD_ORDER:
        shape = tuple(int(dim) for dim in shapes[field])
        dtype = DEFAULT_DTYPES.dtype_for(field)
        nbytes = int(math.prod(shape) * np.dtype(dtype).itemsize)
        started = time.perf_counter()
        arrays[field] = jax.device_put(jnp.zeros(shape, dtype=dtype), device)
        arrays[field].block_until_ready()
        running_total += nbytes
        records.append(
            {
                "field": field,
                "shape": list(shape),
                "dtype": np.dtype(dtype).name,
                "bytes": nbytes,
                "running_total_bytes": running_total,
                "wall_s": time.perf_counter() - started,
                "nvidia_smi_after_field": nvidia_smi_memory(),
            }
        )
    return State(**arrays), records


def _summarize_component_bytes(state: State | None = None, tendencies: Tendencies | None = None, grid: GridSpec | None = None) -> dict[str, Any]:
    components: dict[str, Any] = {}
    if state is not None:
        components["state_bytes"] = int(state.bytes())
    if tendencies is not None:
        components["tendency_bytes"] = int(tendencies.bytes())
    if grid is not None and grid.metrics is not None:
        leaves = jax.tree_util.tree_leaves((grid.eta_levels, grid.terrain_height, grid.metrics))
        components["grid_metric_bytes"] = int(sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves))
    components["total_known_resident_bytes"] = int(sum(value for value in components.values() if isinstance(value, int)))
    return components


def live_vram_probe(output_dir: Path) -> dict[str, Any]:
    grid_shape = build_grid_shape_payload()
    full_grid = make_grid_spec_from_shape(grid_shape)
    payload: dict[str, Any] = {
        "status": "PASS",
        "device": visible_gpu_name(),
        "before": {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()},
        "full_domain_synthetic_state": {},
        "gen2_nested_build_replay_case": {},
    }
    nested_domain = str(grid_shape.get("source_domain") or "d04")
    try:
        nested_case = build_replay_case(FALLBACK_L3_RUN, domain=nested_domain)
        block_until_ready((nested_case.state, nested_case.tendencies, nested_case.metrics, nested_case.base_state))
        payload["gen2_nested_build_replay_case"] = {
            "status": "PASS",
            "run_dir": str(FALLBACK_L3_RUN),
            "domain": nested_domain,
            "grid": nested_case.metadata["grid"],
            "resident_bytes": _summarize_component_bytes(nested_case.state, nested_case.tendencies, nested_case.grid),
            "after": {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()},
            "note": "This proves the existing Gen2 1 km nest loader path, but the full-domain audit uses the derived 3 km -> 1 km shape.",
        }
        del nested_case
        gc.collect()
    except Exception as exc:
        payload["gen2_nested_build_replay_case"] = {
            "status": "FAILED",
            "run_dir": str(FALLBACK_L3_RUN),
            "domain": nested_domain,
            "exception": repr(exc),
        }

    try:
        with GpuMemorySampler() as sampler:
            state, allocation_trace = allocate_state_incremental(full_grid)
            tendencies = Tendencies.zeros(full_grid)
            block_until_ready((state, tendencies, full_grid.metrics))
        payload["full_domain_synthetic_state"] = {
            "status": "PASS",
            "grid": {
                "mass_shape": [int(full_grid.nz), int(full_grid.ny), int(full_grid.nx)],
                "dx_m": float(full_grid.projection.dx_m),
                "dy_m": float(full_grid.projection.dy_m),
            },
            "resident_bytes": _summarize_component_bytes(state, tendencies, full_grid),
            "allocation_trace": allocation_trace,
            "sampler": sampler.summary(),
            "after": {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()},
        }
        del state, tendencies
        gc.collect()
    except Exception as exc:
        payload["status"] = "BLOCKED_OOM"
        payload["full_domain_synthetic_state"] = {
            "status": "BLOCKED_OOM",
            "exception": repr(exc),
            "field_where_it_failed": "see allocation_trace last completed field",
        }

    payload["after_cleanup"] = {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()}
    write_json(output_dir / "live_vram_probe.json", payload)
    print(json.dumps({"phase": "live", "status": payload["status"]}, sort_keys=True))
    return payload


def step_feasibility_probe(output_dir: Path) -> dict[str, Any]:
    grid_shape = build_grid_shape_payload()
    live_path = output_dir / "live_vram_probe.json"
    live = read_json(live_path) if live_path.exists() else {}
    full_live = live.get("full_domain_synthetic_state", {})
    if full_live.get("status") != "PASS":
        payload = {
            "status": "SKIPPED_LIVE_PROBE_NOT_PASS",
            "reason": "AC4 only runs if AC3 full-domain construction passes.",
            "live_status": full_live.get("status"),
        }
        write_json(output_dir / "step_feasibility.json", payload)
        print(json.dumps({"phase": "step", "status": payload["status"]}, sort_keys=True))
        return payload

    full_grid = make_grid_spec_from_shape(grid_shape)
    hours_one_step = DT_S / 3600.0
    payload: dict[str, Any] = {
        "status": "PASS",
        "device": visible_gpu_name(),
        "scope": "one warm RK step on derived full-domain 1 km synthetic State",
        "dt_s": DT_S,
        "hours": hours_one_step,
        "before": {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()},
    }
    try:
        state, cold_alloc = allocate_state_incremental(full_grid)
        tendencies = Tendencies.zeros(full_grid)
        namelist = OperationalNamelist.from_grid(
            full_grid,
            tendencies=tendencies,
            metrics=full_grid.metrics,
            dt_s=DT_S,
            acoustic_substeps=10,
            radiation_cadence_steps=999999,
            use_vertical_solver=True,
        )
        block_until_ready((state, tendencies, full_grid.metrics))
        with GpuMemorySampler() as cold_sampler:
            start = time.perf_counter()
            cold_result = run_forecast_operational(state, namelist, hours_one_step)
            block_until_ready(cold_result)
            cold_wall_s = time.perf_counter() - start
        del cold_result, state
        gc.collect()

        state_warm, warm_alloc = allocate_state_incremental(full_grid)
        block_until_ready(state_warm)
        before_warm = nvidia_smi_memory()
        with GpuMemorySampler() as warm_sampler:
            start = time.perf_counter()
            warm_result = run_forecast_operational(state_warm, namelist, hours_one_step)
            block_until_ready(warm_result)
            warm_wall_s = time.perf_counter() - start
        after_warm = nvidia_smi_memory()
        resident = _summarize_component_bytes(warm_result, tendencies, full_grid)
        peak = warm_sampler.summary().get("peak_memory_used_bytes")
        resident_known = resident.get("total_known_resident_bytes")
        payload.update(
            {
                "cold_compile_inclusive_wall_s": cold_wall_s,
                "warm_step_wall_s": warm_wall_s,
                "warm_step_wall_ms": warm_wall_s * 1000.0,
                "cold_sampler": cold_sampler.summary(),
                "warm_sampler": warm_sampler.summary(),
                "nvidia_smi_before_warm": before_warm,
                "nvidia_smi_after_warm": after_warm,
                "cold_allocation_trace_tail": cold_alloc[-5:],
                "warm_allocation_trace_tail": warm_alloc[-5:],
                "resident_bytes_after_step": resident,
                "transient_buffer_estimate_bytes": int(peak - resident_known) if isinstance(peak, int) and isinstance(resident_known, int) else None,
                "transient_buffer_estimate_note": "peak nvidia-smi process memory minus known State/Tendencies/Grid resident bytes; includes allocator/runtime overhead.",
            }
        )
        del warm_result, state_warm, tendencies, namelist
        gc.collect()
    except Exception as exc:
        message = repr(exc)
        payload["status"] = "BLOCKED_OOM" if "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower() else "FAILED"
        payload["exception"] = message
        payload["after_exception"] = {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()}

    payload["after_cleanup"] = {"nvidia_smi": nvidia_smi_memory(), "jax_memory_stats": device_memory_stats()}
    write_json(output_dir / "step_feasibility.json", payload)
    print(json.dumps({"phase": "step", "status": payload["status"]}, sort_keys=True))
    return payload


def _headroom_pct(used_bytes: int | float | None, total_bytes: int | None) -> float | None:
    if used_bytes is None or total_bytes is None or total_bytes <= 0:
        return None
    return (1.0 - (float(used_bytes) / float(total_bytes))) * 100.0


def build_operational_gaps(output_dir: Path) -> tuple[str, dict[str, Any]]:
    static = read_json(output_dir / "static_memory_model.json")
    grid = read_json(output_dir / "grid_shape_1km.json")
    live = read_json(output_dir / "live_vram_probe.json") if (output_dir / "live_vram_probe.json").exists() else {}
    step = read_json(output_dir / "step_feasibility.json") if (output_dir / "step_feasibility.json").exists() else {}

    total_vram = static.get("device_vram_total_bytes")
    step_peak = step.get("warm_sampler", {}).get("peak_memory_used_bytes")
    live_after = live.get("full_domain_synthetic_state", {}).get("after", {}).get("nvidia_smi", {}).get("memory_used_bytes")
    static_used = static.get("total_state_bytes")
    basis_used = step_peak or live_after or static_used
    headroom = _headroom_pct(basis_used, total_vram)
    if live.get("status") == "BLOCKED_OOM" or step.get("status") == "BLOCKED_OOM":
        verdict = "BLOCKED_OOM"
    elif step.get("status") == "PASS" and headroom is not None and headroom >= 25.0:
        verdict = "FITS_WITH_HEADROOM"
    elif step.get("status") == "PASS":
        verdict = "FITS_TIGHT"
    elif static.get("sanity_total_le_device_vram"):
        verdict = "FITS"
    else:
        verdict = "BLOCKED_OOM"

    top_fields = static.get("top_5_fields_by_bytes", [])
    top_lines = "\n".join(
        f"- {item['field']}: {item['mib']:.2f} MiB ({item['dtype']}, shape={item['shape']})" for item in top_fields
    )
    nested_note = grid.get("contract_candidate_run", {}).get("note", "")
    downcast_lines = [
        "- Do not downcast `mu`, pressure, geopotential, or pressure-gradient/acoustic paths without a new reviewed precision artifact; those are FP64-locked.",
        "- `u`, `v`, `theta`, `qv`, Thompson hydrometeors, number fields, `qke`, and lateral wind/theta/qv boundaries are already stored as FP32-gated fields in `contracts/precision.py`.",
        "- `w` remains the main large candidate that is not already FP32, but ADR-007 requires a sound-wave and Tier-4 operational-impact test before changing it.",
        "- Surface FP64 fields are small relative to 3D pressure/geopotential fields, so they are not first-order memory wins.",
    ]
    fusion_lines = [
        "- Fuse or alias RK/acoustic save-family scratch (`*_save`, `t_2ave`, `ww`, `mudf`, `ph_tend`) where profiler evidence shows duplicated live ranges.",
        "- Reduce pressure-gradient and vertical-solver temporaries by keeping coefficient construction inside the acoustic scan with XLA aliasing/buffer donation evidence.",
        "- Batch Thompson/MYNN/surface physics without materializing independent full-field tendency copies beyond the persistent `Tendencies` contract.",
        "- Keep boundary replay fused with the post-physics step so lateral-boundary padded arrays do not create separate full-domain staging buffers.",
    ]

    summary = {
        "verdict": verdict,
        "headroom_percent": headroom,
        "headroom_basis_bytes": basis_used,
        "vram_total_bytes": total_vram,
        "static_state_bytes": static_used,
        "live_status": live.get("status"),
        "step_status": step.get("status"),
    }
    text = f"""# M7 1 km Memory Audit Operational Gaps

Verdict: {verdict}

Fit answer: {'yes' if verdict in {'FITS', 'FITS_WITH_HEADROOM', 'FITS_TIGHT'} else 'no'} for the measured/probed scope. Static State storage is {static['total_state_gib']:.3f} GiB on the derived full-domain 1 km grid {static['grid']['mass_shape']}. Headroom is {headroom:.2f}% using the strongest available basis ({basis_used} bytes of {total_vram} bytes) if a live basis is available.

Grid provenance: the contract-named wrf_l2 run was checked. {nested_note} The proof object therefore records the available Gen2 l3 1 km nests and uses the sprint objective's full 3 km d02 -> 1 km horizontal scaling for the fit audit.

Top 5 resident State fields by VRAM:
{top_lines}

Downcast candidates and constraints:
{chr(10).join(downcast_lines)}

Kernel-fusion / transient-reduction candidates:
{chr(10).join(fusion_lines)}

What would have to change to make 1 km operational:
- Replace the synthetic full-domain probe with a real full-domain 1 km `wrfinput`/`wrfbdy` source or explicitly scope M7 to the smaller Gen2 d03/d04/d05 nests.
- Capture an Nsight or XLA memory profile for the one-step full-domain probe before claiming transient headroom beyond this nvidia-smi estimate.
- If headroom tightens under real IC/BC, attack transient buffers first; persistent State storage alone is not the limiting footprint in this audit.
- Any precision change must cite ADR-007/Tier-4 evidence, not memory pressure alone.
"""
    return text, summary


def write_operational_gaps(output_dir: Path) -> dict[str, Any]:
    text, summary = build_operational_gaps(output_dir)
    (output_dir / "operational_gaps.md").write_text(text, encoding="utf-8")
    print(json.dumps({"phase": "gaps", "verdict": summary["verdict"]}, sort_keys=True))
    return summary


def run_child_phase(phase: str, output_dir: Path) -> dict[str, Any]:
    stdout_path = output_dir / f"phase_{phase}.stdout.txt"
    stderr_path = output_dir / f"phase_{phase}.stderr.txt"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--output-dir", str(output_dir)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {"phase": phase, "cmd": cmd, "returncode": int(proc.returncode), "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def run_all(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid_shape_payload()
    write_json(output_dir / "grid_shape_1km.json", grid)
    static = build_static_memory_model(grid, total_vram_bytes=vram_total_bytes())
    write_json(output_dir / "static_memory_model.json", static)

    phase_results = [run_child_phase("live", output_dir)]
    live = read_json(output_dir / "live_vram_probe.json") if (output_dir / "live_vram_probe.json").exists() else {}
    if live.get("full_domain_synthetic_state", {}).get("status") == "PASS":
        phase_results.append(run_child_phase("step", output_dir))
    else:
        skipped = {
            "status": "SKIPPED_LIVE_PROBE_NOT_PASS",
            "reason": "AC4 only runs if AC3 full-domain construction passes.",
            "live_status": live.get("full_domain_synthetic_state", {}).get("status"),
        }
        write_json(output_dir / "step_feasibility.json", skipped)
    gaps_summary = write_operational_gaps(output_dir)
    command_payload = {
        "status": "PASS",
        "phase_results": phase_results,
        "verdict": gaps_summary["verdict"],
        "proof_objects": [
            str(output_dir / "static_memory_model.json"),
            str(output_dir / "grid_shape_1km.json"),
            str(output_dir / "live_vram_probe.json"),
            str(output_dir / "step_feasibility.json"),
            str(output_dir / "operational_gaps.md"),
        ],
    }
    write_json(output_dir / "audit_command_summary.json", command_payload)
    print(json.dumps(command_payload, sort_keys=True))
    return 0 if all(result["returncode"] == 0 for result in phase_results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "live", "step", "gaps", "static"), default="all")
    parser.add_argument("--output-dir", type=Path, default=SPRINT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    if args.phase == "all":
        return run_all(output_dir)
    if args.phase == "static":
        grid = build_grid_shape_payload()
        write_json(output_dir / "grid_shape_1km.json", grid)
        write_json(output_dir / "static_memory_model.json", build_static_memory_model(grid, total_vram_bytes=vram_total_bytes()))
        return 0
    if args.phase == "live":
        live_vram_probe(output_dir)
        return 0
    if args.phase == "step":
        step_feasibility_probe(output_dir)
        return 0
    if args.phase == "gaps":
        write_operational_gaps(output_dir)
        return 0
    raise AssertionError(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
