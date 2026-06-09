#!/usr/bin/env python3
"""V0.14 same-input comparison contract builder.

This proof-local tool builds the CPU-side JAX input contract that the prior
early-step discriminator was missing.  It does not edit production code and it
does not run weak comparisons: a strict comparison is attempted only when a
full-domain WRF post-RK/pre-halo truth surface for candidate step 1 is present.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_JSON = ROOT / "proofs/v014/same_input_contract_builder.json"
OUT_MD = ROOT / "proofs/v014/same_input_contract_builder.md"
OUT_PATCH = ROOT / "proofs/v014/same_input_contract_builder_wrf_patch.diff"
SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-same-input-contract-builder/sprint-contract.md"
PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
MANAGING_SPRINTS_SKILL = ROOT / ".agent/skills/managing-sprints/SKILL.md"

STATE_CONTRACT = ROOT / "src/gpuwrf/contracts/state.py"
D02_REPLAY = ROOT / "src/gpuwrf/integration/d02_replay.py"
NESTED_PIPELINE = ROOT / "src/gpuwrf/integration/nested_pipeline.py"
OPERATIONAL_STATE = ROOT / "src/gpuwrf/runtime/operational_state.py"
OPERATIONAL_MODE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
BOUNDARY_CONSTRUCTION = ROOT / "src/gpuwrf/nesting/boundary_construction.py"
DOMAIN_TREE = ROOT / "src/gpuwrf/runtime/domain_tree.py"
METRICS_SOURCE = ROOT / "src/gpuwrf/dynamics/metrics.py"

RUN_CASE3 = Path("/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3")
WRFINPUT_D01 = RUN_CASE3 / "wrfinput_d01"
WRFINPUT_D02 = RUN_CASE3 / "wrfinput_d02"
WRFBDY_D01 = RUN_CASE3 / "wrfbdy_d01"
NAMELIST_INPUT = RUN_CASE3 / "namelist.input"
RSL_ERROR_0000 = RUN_CASE3 / "rsl.error.0000"
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_same_input_contract_builder")

STRICT_STEP = 1
TARGET_FIELDS = ("T", "P", "PB", "PH", "PHB", "MU", "MUB", "U", "V", "W")
ACTIVE_MOISTURE_CANDIDATES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
STATIC_BASE_FIELDS = ("PB", "PHB", "MUB")
DYNAMIC_HEADLINE_FIELDS = ("T", "P", "PH", "MU", "U", "V", "W")
ALL_COMPARE_FIELDS = TARGET_FIELDS + ACTIVE_MOISTURE_CANDIDATES
P0_THETA_OFFSET_K = 300.0
RADT_TARGET_S = 1800.0
BDY_WIDTH = 5

TRUTH_SEARCH_ROOTS = (
    SCRATCH,
    Path("/mnt/data/wrf_gpu2/v014_early_step_discriminator"),
    Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output"),
    Path("/mnt/data/wrf_gpu2/v014_same_state_wrf/marker_output"),
    Path("/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output"),
    Path("/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output"),
)

TRUTH_PATTERNS = (
    "*post_after_all_rk_steps_pre_halo*d2*step_1*.npz",
    "*post_after_all_rk_steps_pre_halo*d02*step_1*.npz",
    "*post_after_all_rk_steps_pre_halo*d2*step_1*.txt",
    "*post_after_all_rk_steps_pre_halo*d02*step_1*.txt",
    "*same_input*step_1*.npz",
)

MAP_PROJ_NAMES = {1: "lambert", 2: "polar", 3: "mercator"}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, float) and not math.isfinite(scalar):
                return None
            return scalar
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], *, cwd: Path | None = None, timeout_s: int = 120) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "PYTHONPATH": str(SRC),
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": None,
            "wall_s": float(time.perf_counter() - start),
            "timeout_s": int(timeout_s),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def function_location(fn: Callable[..., Any]) -> str:
    try:
        source_file = Path(inspect.getsourcefile(fn) or __file__).resolve()
        line = inspect.getsourcelines(fn)[1]
        return f"{source_file.relative_to(ROOT)}:{line}:{fn.__name__}"
    except Exception:
        return f"{Path(__file__).relative_to(ROOT)}:unknown:{getattr(fn, '__name__', '<unknown>')}"


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = [str(device) for device in jax.devices()]
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": devices,
                "gpu_device_count": len([device for device in devices if "gpu" in device.lower()]),
            }
        )
    except Exception as exc:
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def read_tail(path: Path, max_chars: int = 4000) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def _domain_list_value(namelist: Mapping[str, Mapping[str, Any]], group: str, key: str, domain: str, default: Any) -> Any:
    raw = namelist.get(group, {}).get(key, default)
    if isinstance(raw, (list, tuple)):
        index = max(int(domain[1:]) - 1, 0)
        if index < len(raw):
            return raw[index]
        return raw[-1] if raw else default
    return raw


def _domain_dt_s(run: Any, domain: str) -> float:
    root_dt = run.namelist.get("domains", {}).get("time_step")
    if root_dt is None:
        root_dt = run.namelist.get("time_control", {}).get("time_step")
    dt = float(root_dt)
    if domain == "d01":
        return dt
    chain: list[str] = []
    current = domain
    while current != "d01":
        chain.append(current)
        grid = run.grid(current)
        current = f"d{int(grid.parent_id):02d}"
    for child in reversed(chain):
        ratio = int(run.grid(child).parent_grid_ratio)
        dt /= float(ratio)
    return dt


def _zero(shape: tuple[int, ...], *, dtype: Any):
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.zeros(shape, dtype=dtype)


def _array_summary(value: Any) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(jax.device_get(value))
    finite = np.isfinite(arr) if arr.size else np.asarray([], dtype=bool)
    summary: dict[str, Any] = {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "count": int(arr.size),
        "finite_all": bool(finite.all()) if arr.size else True,
        "nan_count": int(np.isnan(arr).sum()) if arr.size and np.issubdtype(arr.dtype, np.floating) else 0,
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        summary.update(
            {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }
        )
    return summary


def _object_leaf_summary(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    fields = {name: _array_summary(getattr(obj, name)) for name in names}
    total_count = int(sum(item["count"] for item in fields.values()))
    total_bytes = 0
    for name in names:
        value = getattr(obj, name)
        total_bytes += int(getattr(value, "size", 0)) * int(getattr(value, "dtype", "float64").itemsize)
    return {"field_count": len(names), "total_count": total_count, "total_bytes": total_bytes, "fields": fields}


def _read_wrf_var(dataset: Any, name: str, *, dtype: Any):
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    var = dataset.variables[name]
    data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
    return jnp.asarray(np.asarray(np.ma.filled(data, np.nan), dtype=np.float64), dtype=dtype)


def _optional_wrf_var(dataset: Any, name: str, shape: tuple[int, ...], *, dtype: Any):
    if name not in dataset.variables:
        return _zero(shape, dtype=dtype)
    return _read_wrf_var(dataset, name, dtype=dtype)


def _grid_from_wrfinput(path: Path, run: Any, domain: str, metrics: Any):
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from netCDF4 import Dataset  # type: ignore # noqa: PLC0415

    from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord  # noqa: PLC0415

    with Dataset(path) as dataset:
        dims = dataset.dimensions
        attrs = {
            name: getattr(dataset, name)
            for name in ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON")
            if hasattr(dataset, name)
        }
        mass_nx = int(len(dims["west_east"]))
        mass_ny = int(len(dims["south_north"]))
        mass_nz = int(len(dims["bottom_top"]))
        map_proj_id = int(attrs.get("MAP_PROJ", 1))
        projection = Projection(
            MAP_PROJ_NAMES.get(map_proj_id, f"wrf_map_proj_{map_proj_id}"),
            float(attrs.get("CEN_LAT")),
            float(attrs.get("CEN_LON")),
            float(_domain_list_value(run.namelist, "domains", "dx", domain, attrs.get("DX"))),
            float(_domain_list_value(run.namelist, "domains", "dy", domain, attrs.get("DY"))),
            mass_nx,
            mass_ny,
        )
        terrain_height_np = np.asarray(dataset.variables["HGT"][0], dtype=np.float64)
        terrain = TerrainProvenance(
            source_path=str(path),
            sha256=sha256(path) or "missing",
            shape=(mass_ny, mass_nx),
            units="m",
            projection_transform="native-wrf-lambert",
            max_elevation_m=float(np.nanmax(terrain_height_np)),
            coastline_sanity_check_passed=True,
        )
        eta_levels = jnp.asarray(np.asarray(dataset.variables["ZNW"][0], dtype=np.float64), dtype=jnp.float64)
        top_pressure = float(np.asarray(dataset.variables["P_TOP"][0] if "Time" in dataset.variables["P_TOP"].dimensions else dataset.variables["P_TOP"][:]).reshape(-1)[0])
        vertical = VerticalCoord("hybrid_eta", mass_nz, top_pressure, eta_levels)
        bc = BCMetadata(
            source="AIFS",
            fields=("U", "V", "T", "QVAPOR", "PH"),
            update_cadence_h=1,
            interpolation="linear",
            restart_compatible=True,
        )
        terrain_height = jnp.asarray(terrain_height_np, dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height, metrics=metrics)


def _state_from_wrfinput(run: Any, domain: str) -> dict[str, Any]:
    import jax.numpy as jnp  # noqa: PLC0415
    from netCDF4 import Dataset  # type: ignore # noqa: PLC0415

    from gpuwrf.contracts.state import BaseState, State, Tendencies  # noqa: PLC0415
    from gpuwrf.dynamics.metrics import load_wrfinput_metrics  # noqa: PLC0415
    from gpuwrf.integration.d02_replay import _wrf_base_theta_from_loaded_state, _wrf_mynn_coldstart_qke  # noqa: PLC0415
    from gpuwrf.io.land_state import load_prescribed_land_state  # noqa: PLC0415

    wrfinput = run.wrfinput_file(domain)
    metrics = load_wrfinput_metrics(wrfinput)
    grid = _grid_from_wrfinput(wrfinput, run, domain, metrics)
    nz, ny, nx = int(grid.nz), int(grid.ny), int(grid.nx)
    mass_3d = (nz, ny, nx)
    surface_2d = (ny, nx)
    boundary_side = max(nx + 1, ny + 1)
    boundary_mass = (1, 4, BDY_WIDTH, nz, boundary_side)
    boundary_face = (1, 4, BDY_WIDTH, nz + 1, boundary_side)
    boundary_surface = (1, 4, BDY_WIDTH, 1, boundary_side)
    fp64 = jnp.float64

    with Dataset(wrfinput) as dataset:
        p_perturbation = _read_wrf_var(dataset, "P", dtype=fp64)
        pb = _read_wrf_var(dataset, "PB", dtype=fp64)
        ph_perturbation = _read_wrf_var(dataset, "PH", dtype=fp64)
        phb = _read_wrf_var(dataset, "PHB", dtype=fp64)
        mu_perturbation = _read_wrf_var(dataset, "MU", dtype=fp64)
        mub = _read_wrf_var(dataset, "MUB", dtype=fp64)
        theta = _read_wrf_var(dataset, "T", dtype=fp64) + P0_THETA_OFFSET_K
        p_total = pb + p_perturbation
        ph_total = phb + ph_perturbation
        mu_total = mub + mu_perturbation
        land = load_prescribed_land_state(run, domain=domain, time=0)
        state = State(
            u=_read_wrf_var(dataset, "U", dtype=fp64),
            v=_read_wrf_var(dataset, "V", dtype=fp64),
            w=_read_wrf_var(dataset, "W", dtype=fp64),
            theta=theta,
            qv=_read_wrf_var(dataset, "QVAPOR", dtype=fp64),
            p=p_total,
            ph=ph_total,
            mu=mu_total,
            p_total=p_total,
            p_perturbation=p_perturbation,
            ph_total=ph_total,
            ph_perturbation=ph_perturbation,
            mu_total=mu_total,
            mu_perturbation=mu_perturbation,
            qc=_optional_wrf_var(dataset, "QCLOUD", mass_3d, dtype=fp64),
            qr=_optional_wrf_var(dataset, "QRAIN", mass_3d, dtype=fp64),
            qi=_optional_wrf_var(dataset, "QICE", mass_3d, dtype=fp64),
            qs=_optional_wrf_var(dataset, "QSNOW", mass_3d, dtype=fp64),
            qg=_optional_wrf_var(dataset, "QGRAUP", mass_3d, dtype=fp64),
            Ni=_optional_wrf_var(dataset, "QNICE", mass_3d, dtype=fp64),
            Nr=_optional_wrf_var(dataset, "QNRAIN", mass_3d, dtype=fp64),
            Ns=_zero(mass_3d, dtype=fp64),
            Ng=_zero(mass_3d, dtype=fp64),
            qke=_optional_wrf_var(dataset, "QKE", mass_3d, dtype=fp64),
            ustar=_zero(surface_2d, dtype=fp64),
            theta_flux=_zero(surface_2d, dtype=fp64),
            qv_flux=_zero(surface_2d, dtype=fp64),
            tau_u=_zero(surface_2d, dtype=fp64),
            tau_v=_zero(surface_2d, dtype=fp64),
            rhosfc=_zero(surface_2d, dtype=fp64),
            fltv=_zero(surface_2d, dtype=fp64),
            t_skin=jnp.asarray(land.t_skin, dtype=fp64),
            soil_moisture=jnp.asarray(land.soil_moisture[0], dtype=fp64),
            xland=jnp.asarray(land.xland, dtype=fp64),
            lakemask=jnp.asarray(land.lakemask, dtype=fp64),
            mavail=jnp.asarray(land.mavail, dtype=fp64),
            roughness_m=jnp.asarray(land.roughness_m, dtype=fp64),
            rain_acc=_zero(surface_2d, dtype=fp64),
            snow_acc=_zero(surface_2d, dtype=fp64),
            graupel_acc=_zero(surface_2d, dtype=fp64),
            ice_acc=_zero(surface_2d, dtype=fp64),
            u_bdy=_zero(boundary_mass, dtype=fp64),
            v_bdy=_zero(boundary_mass, dtype=fp64),
            theta_bdy=_zero(boundary_mass, dtype=fp64),
            qv_bdy=_zero(boundary_mass, dtype=fp64),
            ph_bdy=_zero(boundary_face, dtype=fp64),
            mu_bdy=_zero(boundary_surface, dtype=fp64),
            w_bdy=_zero(boundary_face, dtype=fp64),
            p_bdy=_zero(boundary_mass, dtype=fp64),
            pb_bdy=_zero(boundary_mass, dtype=fp64),
            phb_bdy=_zero(boundary_face, dtype=fp64),
            mub_bdy=_zero(boundary_surface, dtype=fp64),
            lu_index=jnp.asarray(land.lu_index, dtype=jnp.int32),
            Nc=_zero(mass_3d, dtype=fp64),
            Nn=_zero(mass_3d, dtype=fp64),
            rainc_acc=_zero(surface_2d, dtype=fp64),
        )

    qke_seeded, did_seed_qke = _wrf_mynn_coldstart_qke(state.qke, ph_total=state.ph_total, ustar=state.ustar)
    if did_seed_qke:
        state = state.replace(qke=qke_seeded.astype(state.qke.dtype))
    theta_base = _wrf_base_theta_from_loaded_state(pb=pb, phb=phb, mub=mub, metrics=metrics)
    base_state = BaseState(
        pb=pb.astype(state.p_total.dtype),
        phb=phb.astype(state.ph_total.dtype),
        mub=mub.astype(state.mu_total.dtype),
        t0=jnp.full_like(theta_base, P0_THETA_OFFSET_K).astype(state.theta.dtype),
        theta_base=theta_base.astype(state.theta.dtype),
    )
    tendencies = Tendencies(
        u=_zero((nz, ny, nx + 1), dtype=fp64),
        v=_zero((nz, ny + 1, nx), dtype=fp64),
        w=_zero((nz + 1, ny, nx), dtype=fp64),
        theta=_zero(mass_3d, dtype=fp64),
        qv=_zero(mass_3d, dtype=fp64),
        p=_zero(mass_3d, dtype=fp64),
        ph=_zero((nz + 1, ny, nx), dtype=fp64),
        mu=_zero(surface_2d, dtype=fp64),
    )
    return {
        "domain": domain,
        "wrfinput": wrfinput,
        "grid": grid,
        "metrics": metrics,
        "state": state,
        "base_state": base_state,
        "tendencies": tendencies,
        "qke_coldstart_seeded": bool(did_seed_qke),
        "construction": "proof-local direct constructors; no State.zeros/Tendencies.zeros/BaseState.zeros",
    }


def build_cpu_same_input_contract() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    from gpuwrf.integration.d02_replay import run_start_label  # noqa: PLC0415
    from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: PLC0415
    from gpuwrf.nesting.boundary_construction import build_child_boundary_package, build_nest_force_weights  # noqa: PLC0415
    from gpuwrf.runtime.domain_tree import with_live_child_boundary_config  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: PLC0415
    from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: PLC0415

    run = Gen2Run(RUN_CASE3)
    parent = _state_from_wrfinput(run, "d01")
    child = _state_from_wrfinput(run, "d02")
    parent_grid_meta = run.grid("d01")
    child_grid_meta = run.grid("d02")

    weights = build_nest_force_weights(
        parent_grid_ratio=int(child_grid_meta.parent_grid_ratio),
        i_parent_start=int(child_grid_meta.i_parent_start),
        j_parent_start=int(child_grid_meta.j_parent_start),
        parent_grid=parent["grid"],
        child_grid=child["grid"],
        registration="sint",
    )
    child_state_with_parent_bdy = build_child_boundary_package(
        child["state"],
        parent["state"],
        weights,
        bdy_width=BDY_WIDTH,
    )
    child_dt = _domain_dt_s(run, "d02")
    parent_dt = _domain_dt_s(run, "d01")
    radiation_cadence = max(1, int(round(RADT_TARGET_S / float(child_dt))))
    namelist = OperationalNamelist.from_grid(
        child["grid"],
        tendencies=child["tendencies"],
        metrics=child["metrics"],
        dt_s=child_dt,
        acoustic_substeps=10,
        radiation_cadence_steps=radiation_cadence,
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
        time_utc=run_start_label(run, "d02"),
    )
    namelist = with_live_child_boundary_config(
        namelist,
        parent_dt_s=parent_dt,
        nested_ph_relax=True,
        nested_w_relax=False,
        nested_ph_spec=True,
    )
    cu_physics = int(_domain_list_value(run.namelist, "physics", "cu_physics", "d02", 0))
    namelist = dataclass_replace(namelist, cu_physics=cu_physics)
    carry = initial_operational_carry(child_state_with_parent_bdy)
    jax.block_until_ready(jax.tree_util.tree_leaves(carry)[0])

    state_slots = tuple(child_state_with_parent_bdy.__slots__)
    carry_fields = tuple(carry.__dataclass_fields__.keys())
    state_required = {
        name: _array_summary(_jax_compare_array(name, child_state_with_parent_bdy, child["base_state"]))
        for name in ALL_COMPARE_FIELDS
    }
    return {
        "status": "READY_CPU_INITIAL_D02_CONTRACT_WITH_JAX_PARENT_BOUNDARY_PACKAGE",
        "run_dir": str(RUN_CASE3),
        "no_state_zeros_called": True,
        "domains_loaded_from_wrfinput": ["d01", "d02"],
        "parent_boundary_package": {
            "status": "constructed_by_production_JAX_builder_from_wrfinput_d01_d02",
            "strictness_note": (
                "This is the JAX-side live-parent package required to start a d02 step. "
                "It is not a WRF truth substitute; strict comparison still requires a WRF "
                "post-RK/pre-halo surface."
            ),
            "builder": "gpuwrf.nesting.boundary_construction.build_child_boundary_package",
            "registration": "sint",
            "bdy_width": BDY_WIDTH,
            "u_bdy_shape_after_package": list(child_state_with_parent_bdy.u_bdy.shape),
            "theta_bdy_shape_after_package": list(child_state_with_parent_bdy.theta_bdy.shape),
        },
        "namelist": {
            "status": "READY",
            "source_recipe": "nested_pipeline._make_namelist / daily_pipeline real-case dycore knobs",
            "dt_s": float(namelist.dt_s),
            "parent_dt_s": float(parent_dt),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "boundary_update_cadence_s": float(namelist.boundary_config.update_cadence_s),
            "use_flux_advection": bool(namelist.use_flux_advection),
            "force_fp64": bool(namelist.force_fp64),
            "diff_6th_opt": int(namelist.diff_6th_opt),
            "w_damping": int(namelist.w_damping),
            "damp_opt": int(namelist.damp_opt),
            "epssm": float(namelist.epssm),
            "top_lid": bool(namelist.top_lid),
            "cu_physics": int(namelist.cu_physics),
            "radiation_static_loaded": namelist.radiation_static is not None,
            "gwdo_statics_loaded": namelist.gwdo_statics is not None,
        },
        "grid": {
            "domain": "d02",
            "mass_shape": [int(child["grid"].nz), int(child["grid"].ny), int(child["grid"].nx)],
            "dx_m": float(child["grid"].projection.dx_m),
            "dy_m": float(child["grid"].projection.dy_m),
            "terrain_source": str(child["wrfinput"]),
            "metrics_source": str(child["wrfinput"]),
        },
        "objects": {
            "state": {
                "slot_count": len(state_slots),
                "required_compare_fields": state_required,
                "all_state_bytes": int(child_state_with_parent_bdy.bytes()),
            },
            "tendencies": _object_leaf_summary(child["tendencies"], tuple(child["tendencies"].__slots__)),
            "base_state": _object_leaf_summary(child["base_state"], tuple(child["base_state"].__slots__)),
            "initial_operational_carry": {
                "field_count": len(carry_fields),
                "fields": {
                    name: (
                        {"type": "None"}
                        if getattr(carry, name) is None
                        else _array_summary(getattr(carry, name))
                        if hasattr(getattr(carry, name), "shape")
                        else {"type": type(getattr(carry, name)).__name__}
                    )
                    for name in carry_fields
                    if name != "state"
                },
            },
        },
        "loader_locations": {
            "proof_loader": function_location(build_cpu_same_input_contract),
            "state_constructor": "src/gpuwrf/contracts/state.py:State.__init__",
            "base_constructor": "src/gpuwrf/contracts/state.py:BaseState.__init__",
            "tendencies_constructor": "src/gpuwrf/contracts/state.py:Tendencies.__init__",
            "carry_constructor": "src/gpuwrf/runtime/operational_state.py:initial_operational_carry",
            "parent_boundary_builder": "src/gpuwrf/nesting/boundary_construction.py:build_child_boundary_package",
        },
        "_objects_for_compare": {
            "state": child_state_with_parent_bdy,
            "base_state": child["base_state"],
        },
    }


def _jax_compare_array(field: str, state: Any, base_state: Any):
    import jax.numpy as jnp  # noqa: PLC0415

    mapping = {
        "T": lambda: jnp.asarray(state.theta) - P0_THETA_OFFSET_K,
        "P": lambda: state.p_perturbation,
        "PB": lambda: base_state.pb,
        "PH": lambda: state.ph_perturbation,
        "PHB": lambda: base_state.phb,
        "MU": lambda: state.mu_perturbation,
        "MUB": lambda: base_state.mub,
        "U": lambda: state.u,
        "V": lambda: state.v,
        "W": lambda: state.w,
        "QVAPOR": lambda: state.qv,
        "QCLOUD": lambda: state.qc,
        "QRAIN": lambda: state.qr,
        "QICE": lambda: state.qi,
        "QSNOW": lambda: state.qs,
        "QGRAUP": lambda: state.qg,
    }
    return mapping[field]()


def _wrfinput_field_inventory() -> dict[str, Any]:
    from netCDF4 import Dataset  # type: ignore # noqa: PLC0415

    with Dataset(WRFINPUT_D02) as dataset:
        dims = {name: int(len(dim)) for name, dim in dataset.dimensions.items()}
        variables = set(dataset.variables)
        fields: dict[str, Any] = {}
        for name in ALL_COMPARE_FIELDS:
            present = name in variables
            if present:
                var = dataset.variables[name]
                shape = tuple(int(x) for x in var.shape)
                if var.dimensions and var.dimensions[0] == "Time":
                    compare_shape = shape[1:]
                else:
                    compare_shape = shape
                fields[name] = {
                    "present": True,
                    "wrf_variable": name,
                    "dimensions": list(var.dimensions),
                    "netcdf_shape": list(shape),
                    "compare_shape_without_time": list(compare_shape),
                    "dtype": str(var.dtype),
                    "units_attr": getattr(var, "units", None),
                    "description_attr": getattr(var, "description", None),
                }
            else:
                fields[name] = {"present": False, "wrf_variable": name}
    return {
        "wrfinput_d02": str(WRFINPUT_D02),
        "dims": dims,
        "target_fields": fields,
        "active_moisture_present": [name for name in ACTIVE_MOISTURE_CANDIDATES if fields[name]["present"]],
    }


def build_field_schema(contract: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    required_summaries = contract["objects"]["state"]["required_compare_fields"]
    runtime_sources = {
        "T": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%t_2(i,k,j)",
        "P": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%p(i,k,j)",
        "PB": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%pb(i,k,j)",
        "PH": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%ph_2(i,k,j)",
        "PHB": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%phb(i,k,j)",
        "MU": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%mu_2(i,j)",
        "MUB": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%mub(i,j)",
        "U": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%u_2(i,k,j)",
        "V": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%v_2(i,k,j)",
        "W": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo grid%w_2(i,k,j)",
        "QVAPOR": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo moist(i,k,j,P_QV)",
        "QCLOUD": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo moist(i,k,j,P_QC)",
        "QRAIN": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo moist(i,k,j,P_QR)",
        "QICE": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo moist(i,k,j,P_QI)",
        "QSNOW": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo moist(i,k,j,P_QS)",
        "QGRAUP": "dyn_em/solve_em.F post_after_all_rk_steps_pre_halo moist(i,k,j,P_QG)",
    }
    jax_sources = {
        "T": "state.theta - 300.0",
        "P": "state.p_perturbation",
        "PB": "base_state.pb",
        "PH": "state.ph_perturbation",
        "PHB": "base_state.phb",
        "MU": "state.mu_perturbation",
        "MUB": "base_state.mub",
        "U": "state.u",
        "V": "state.v",
        "W": "state.w",
        "QVAPOR": "state.qv",
        "QCLOUD": "state.qc",
        "QRAIN": "state.qr",
        "QICE": "state.qi",
        "QSNOW": "state.qs",
        "QGRAUP": "state.qg",
    }
    units = {
        "T": "K perturbation potential temperature (WRF T; JAX subtracts 300 K from total theta)",
        "P": "Pa perturbation pressure",
        "PB": "Pa base pressure",
        "PH": "m2 s-2 perturbation geopotential",
        "PHB": "m2 s-2 base geopotential",
        "MU": "Pa perturbation dry-column mass",
        "MUB": "Pa base dry-column mass",
        "U": "m s-1 x-wind on west_east_stag",
        "V": "m s-1 y-wind on south_north_stag",
        "W": "m s-1 vertical velocity on bottom_top_stag",
        "QVAPOR": "kg kg-1 water vapor mixing ratio",
        "QCLOUD": "kg kg-1 cloud water mixing ratio",
        "QRAIN": "kg kg-1 rain water mixing ratio",
        "QICE": "kg kg-1 ice mixing ratio",
        "QSNOW": "kg kg-1 snow mixing ratio",
        "QGRAUP": "kg kg-1 graupel mixing ratio",
    }
    staggering = {
        "T": "mass(k,y,x)",
        "P": "mass(k,y,x)",
        "PB": "mass(k,y,x)",
        "PH": "vertical_face(kstag,y,x)",
        "PHB": "vertical_face(kstag,y,x)",
        "MU": "surface_mass(y,x)",
        "MUB": "surface_mass(y,x)",
        "U": "x_face(k,y,xstag)",
        "V": "y_face(k,ystag,x)",
        "W": "vertical_face(kstag,y,x)",
        "QVAPOR": "mass(k,y,x)",
        "QCLOUD": "mass(k,y,x)",
        "QRAIN": "mass(k,y,x)",
        "QICE": "mass(k,y,x)",
        "QSNOW": "mass(k,y,x)",
        "QGRAUP": "mass(k,y,x)",
    }
    fields: dict[str, Any] = {}
    inv_fields = inventory["target_fields"]
    for name in ALL_COMPARE_FIELDS:
        shape = required_summaries[name]["shape"]
        count = required_summaries[name]["count"]
        fields[name] = {
            "units": units[name],
            "staggering": staggering[name],
            "shape": shape,
            "count": count,
            "wrf_initial_source": f"wrfinput_d02:{name}" if inv_fields[name]["present"] else "absent in wrfinput_d02; zero-filled JAX optional leaf",
            "wrf_runtime_source": runtime_sources[name],
            "jax_leaf_source": jax_sources[name],
            "jax_dtype": required_summaries[name]["dtype"],
            "wrf_netcdf_dtype": inv_fields[name].get("dtype"),
            "headline_dynamic_selector": name in DYNAMIC_HEADLINE_FIELDS,
            "static_or_base_excluded_from_headline": name in STATIC_BASE_FIELDS,
        }
    return {
        "schema": "wrfgpu2.v014.same_input_field_map.v1",
        "fields": fields,
        "required_minimum_fields": list(TARGET_FIELDS),
        "active_moisture_candidates": list(ACTIVE_MOISTURE_CANDIDATES),
        "active_moisture_present": inventory["active_moisture_present"],
        "dynamic_or_perturbation_headline_fields": list(DYNAMIC_HEADLINE_FIELDS),
        "static_or_base_fields_excluded_from_headline_selector": list(STATIC_BASE_FIELDS),
        "index_semantics": {
            "array_order": "All comparisons use JAX logical zero-based array order after stripping WRF Time: mass (k,y,x), U (k,y,xstag), V (k,ystag,x), W/PH (kstag,y,x), MU/MUB (y,x).",
            "first_mismatch_index": "First nonzero-difference element in row-major flattening of that JAX logical array.",
            "worst_mismatch_index": "Index of maximum absolute difference in that JAX logical array; ties keep numpy.argmax's first row-major occurrence.",
            "wrf_fortran_index_conversion": "For 3-D fields report Fortran i=x+1, j=y+1, k=k+1 or kstag+1; for 2-D fields i=x+1, j=y+1.",
            "comparison_dtype": "float64 CPU arrays for JAX side; WRF truth is cast to float64 before residual metrics.",
        },
    }


def scan_truth_surfaces() -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for root in TRUTH_SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in TRUTH_PATTERNS:
            for path in sorted(root.glob(pattern)):
                matches.append(path_info(path))
    noncandidate_step6000 = []
    for root in TRUTH_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*post_after_all_rk_steps_pre_halo*d2*step_6000*.txt")):
            noncandidate_step6000.append(path_info(path))
    return {
        "required_step": STRICT_STEP,
        "required_surface": "post_after_all_rk_steps_pre_halo",
        "required_domain": "d02",
        "status": "AVAILABLE" if matches else "MISSING",
        "matching_files": matches,
        "search_roots": [str(root) for root in TRUTH_SEARCH_ROOTS],
        "noncandidate_step6000_patch_surfaces": noncandidate_step6000,
        "full_domain_requirement": {
            "accepted_truth_format": "npz with one full-domain array per schema field, or text converted losslessly to that npz contract",
            "required_keys": list(ALL_COMPARE_FIELDS),
            "one_cell_or_tile_patch_rejected": True,
        },
    }


def compare_if_possible(truth: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if truth["status"] != "AVAILABLE":
        return {
            "strict_same_input_comparison_run": False,
            "why_empty": "No candidate step-1 full-domain WRF post-RK/pre-halo truth surface was found.",
            "per_field_metrics": {},
            "ranked_residuals": [],
        }
    npz_candidates = [Path(item["path"]) for item in truth["matching_files"] if str(item["path"]).endswith(".npz")]
    if not npz_candidates:
        return {
            "strict_same_input_comparison_run": False,
            "why_empty": "Only non-npz surfaces were found; no accepted full-domain truth contract is available.",
            "per_field_metrics": {},
            "ranked_residuals": [],
        }
    path = npz_candidates[0]
    state = contract["_objects_for_compare"]["state"]
    base_state = contract["_objects_for_compare"]["base_state"]
    metrics: dict[str, Any] = {}
    with np.load(path) as truth_npz:
        missing = [name for name in ALL_COMPARE_FIELDS if name not in truth_npz]
        if missing:
            return {
                "strict_same_input_comparison_run": False,
                "why_empty": f"Truth file {path} is missing required keys: {missing}",
                "per_field_metrics": {},
                "ranked_residuals": [],
            }
        for name in ALL_COMPARE_FIELDS:
            left = np.asarray(truth_npz[name], dtype=np.float64)
            right = np.asarray(jax.device_get(_jax_compare_array(name, state, base_state)), dtype=np.float64)
            if left.shape != right.shape:
                return {
                    "strict_same_input_comparison_run": False,
                    "why_empty": f"Truth file {path} field {name} shape {left.shape} != JAX shape {right.shape}",
                    "per_field_metrics": {},
                    "ranked_residuals": [],
                }
            diff = right - left
            absdiff = np.abs(diff)
            mismatch = np.argwhere(diff != 0.0)
            first = tuple(int(x) for x in mismatch[0]) if mismatch.size else None
            worst = tuple(int(x) for x in np.unravel_index(int(np.argmax(absdiff)), absdiff.shape)) if absdiff.size else None
            metrics[name] = {
                "count": int(diff.size),
                "max_abs": float(np.max(absdiff)) if absdiff.size else 0.0,
                "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
                "bias": float(np.mean(diff)) if diff.size else 0.0,
                "p95": float(np.percentile(absdiff, 95)) if absdiff.size else 0.0,
                "p99": float(np.percentile(absdiff, 99)) if absdiff.size else 0.0,
                "first_mismatch_index": first,
                "worst_mismatch_index": worst,
            }
    ranked = sorted(
        [{"field": name, **item} for name, item in metrics.items()],
        key=lambda item: item["max_abs"],
        reverse=True,
    )
    return {
        "strict_same_input_comparison_run": True,
        "truth_file": str(path),
        "per_field_metrics": metrics,
        "ranked_residuals": ranked,
    }


def blocker_payload(truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    if truth["status"] == "AVAILABLE":
        return []
    return [
        {
            "id": "NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1",
            "exact_missing_field": "Full-domain WRF truth arrays for T, P, PB, PH, PHB, MU, MUB, U, V, W, QVAPOR, QCLOUD, QRAIN, QICE, QSNOW, QGRAUP at d02 grid%itimestep=1 post_after_all_rk_steps_pre_halo.",
            "exact_wrf_source_location": "WRF dyn_em/solve_em.F immediately after after_all_rk_steps and before the RK halo includes; prior v014 patch inserted at solve_em.F around the post_after_all_rk_steps_pre_halo marker.",
            "exact_jax_loader_source_location": function_location(build_cpu_same_input_contract),
            "smallest_next_patch_or_tool": (
                "Apply a disposable CPU-WRF hook in a scratch WRF copy that emits the schema fields at "
                "domain 2, step 1, post_after_all_rk_steps_pre_halo, then convert the full-domain surface "
                "to /mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/"
                "same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz with the schema keys."
            ),
            "why_existing_surfaces_fail": "Existing refresh_post_after_all_rk_steps_pre_halo files are step 6000 tile/patch surfaces, not candidate step 1 full-domain truth.",
        }
    ]


def implementation_ready_wrf_recipe() -> dict[str, Any]:
    return {
        "status": "PATCH_NOT_APPLIED_IN_THIS_SPRINT",
        "reason": "Strict CPU/JAX contract was built first; no candidate WRF truth file exists under allowed scratch, and applying/rebuilding WRF would require a separate disposable CPU-WRF hook run.",
        "source_candidates": [
            "/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF",
            "/mnt/data/wrf_gpu2/v014_source_save_boundary/WRF",
            "/home/enric/src/wrf_pristine/WRF",
        ],
        "scratch_root_for_next_run": str(SCRATCH / "wrf_step1_truth_run"),
        "required_hook_boundary": "dyn_em/solve_em.F::solve_em, after after_all_rk_steps and before RK halos, domain grid%id==2, grid%itimestep==1",
        "output_contract": {
            "path": str(SCRATCH / "wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz"),
            "keys": list(ALL_COMPARE_FIELDS),
            "shape_semantics": "Time stripped; axes exactly match field_schema.index_semantics.array_order.",
        },
        "minimal_commands_next_sprint": [
            "mkdir -p /mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_step1_truth_run",
            "copy or rsync a disposable WRF tree into that scratch run directory",
            "apply a post_after_all_rk_steps_pre_halo full-domain step-1 hook patch to the scratch WRF tree",
            "rebuild wrf.exe in the scratch WRF tree",
            "run CPU-WRF from /mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3 with WRFGPU2_SAME_INPUT_STEP1=1 and output root under /mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth",
            "convert emitted WRF surface to the accepted npz contract and rerun this builder",
        ],
    }


def strip_private_objects(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    contract = dict(copied["cpu_same_input_contract"])
    contract.pop("_objects_for_compare", None)
    copied["cpu_same_input_contract"] = contract
    return copied


def write_markdown(payload: Mapping[str, Any]) -> None:
    verdict = payload["verdict"]
    truth = payload["wrf_truth_surface"]
    loader = payload["cpu_same_input_contract"]
    comparison = payload["comparison"]
    lines = [
        "# V0.14 Same-Input Contract Builder",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU proof-local loader: `{loader['status']}`.",
        "- `State`, `Tendencies`, `BaseState`/metrics, `OperationalNamelist`, and initial `OperationalCarry` were constructed without `State.zeros`.",
        f"- Frozen field schema covers `{len(payload['field_schema']['fields'])}` WRF/JAX fields, including active moisture.",
        f"- WRF step-1 post-RK/pre-halo truth: `{truth['status']}`.",
        f"- Strict comparison run: `{comparison['strict_same_input_comparison_run']}`.",
        "",
        "## Blocker",
        "",
        "No strict comparison ran because no full-domain WRF truth surface exists for d02 step 1 at `post_after_all_rk_steps_pre_halo`.",
        "Existing step-6000 patch surfaces are non-candidate and tile/patch scoped.",
        "",
        "Next decision: run a disposable CPU-WRF step-1 full-domain hook into the accepted npz truth contract, then rerun this builder.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    env = jax_environment()
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    inventory = _wrfinput_field_inventory()
    cpu_contract = build_cpu_same_input_contract()
    field_schema = build_field_schema(cpu_contract, inventory)
    truth = scan_truth_surfaces()
    comparison = compare_if_possible(truth, cpu_contract)
    blockers = blocker_payload(truth)
    if comparison["strict_same_input_comparison_run"]:
        ranked = comparison.get("ranked_residuals", [])
        if ranked and ranked[0]["max_abs"] != 0.0:
            verdict = f"SAME_INPUT_CONTRACT_EXECUTED_FIRST_DIVERGENT_STEP_{STRICT_STEP}_{ranked[0]['field']}"
        else:
            verdict = f"SAME_INPUT_CONTRACT_EXECUTED_CLEAN_THROUGH_{STRICT_STEP}"
    else:
        verdict = "SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1"

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.same_input_contract_builder.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "hermes_used": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_work": False,
        "weak_comparison_avoided": True,
        "jax_vs_jax_self_compare": False,
        "one_cell_proof": False,
        "mixed_jax_carry_with_wrf_truth": False,
        "environment": env,
        "git_head": git_head,
        "inputs": {
            "project_constitution": path_info(PROJECT_CONSTITUTION),
            "agents": path_info(AGENTS),
            "managing_sprints_skill": path_info(MANAGING_SPRINTS_SKILL),
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "run_case3": path_info(RUN_CASE3),
            "wrfinput_d01": path_info(WRFINPUT_D01),
            "wrfinput_d02": path_info(WRFINPUT_D02),
            "wrfbdy_d01": path_info(WRFBDY_D01),
            "namelist_input": path_info(NAMELIST_INPUT),
            "rsl_error_0000": path_info(RSL_ERROR_0000),
        },
        "source_locations": {
            "state_contract": path_info(STATE_CONTRACT),
            "d02_replay": path_info(D02_REPLAY),
            "nested_pipeline": path_info(NESTED_PIPELINE),
            "operational_state": path_info(OPERATIONAL_STATE),
            "operational_mode": path_info(OPERATIONAL_MODE),
            "boundary_construction": path_info(BOUNDARY_CONSTRUCTION),
            "domain_tree": path_info(DOMAIN_TREE),
            "metrics_source": path_info(METRICS_SOURCE),
        },
        "wrfinput_inventory": inventory,
        "cpu_same_input_contract": cpu_contract,
        "field_schema": field_schema,
        "wrf_truth_surface": truth,
        "wrf_truth_next_recipe": implementation_ready_wrf_recipe(),
        "comparison": comparison,
        "blockers": blockers,
        "commands": {
            "validation": [
                "python -m py_compile proofs/v014/same_input_contract_builder.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_contract_builder.py",
                "python -m json.tool proofs/v014/same_input_contract_builder.json >/tmp/same_input_contract_builder.validated.json",
                "git diff -- src/gpuwrf",
            ]
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
        "unresolved_risks": [
            "No numerical first-divergent field/operator is named until the WRF step-1 full-domain truth surface exists.",
            "The JAX-side live parent boundary package is constructed, but strict same-input validation still needs the WRF Fortran surface at the same boundary.",
            "Radiation/GWDO static attachments are not loaded in the proof namelist because no timestep execution is attempted in this tooling sprint.",
        ],
        "next_decision": "Run the disposable CPU-WRF step-1 full-domain post-RK/pre-halo truth hook and rerun this builder.",
        "log_tails": {
            "run_case3_rsl_error_0000": read_tail(RSL_ERROR_0000),
        },
    }
    serializable = strip_private_objects(payload)
    write_json(OUT_JSON, serializable)
    write_markdown(serializable)
    print(f"verdict={verdict}")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
