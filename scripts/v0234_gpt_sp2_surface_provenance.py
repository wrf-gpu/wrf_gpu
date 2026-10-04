#!/usr/bin/env python3
"""Seal the source-authorized v0234 SP2 surface correction.

This CPU-only proof replays the first MYNN surface call from the authenticated
pre-PBL seam atmosphere and the authenticated step-0 lower-boundary carry.  It
independently parses pristine ``LANDUSE.TBL``, applies WRF's date/hemisphere
season rule, and compares summer and winter replays with the sealed pristine
WRF UST/WSPD dump.  It also proves the distinct MYNN mean-solver surface-density
formula against the dumped operands.

No production PBL adapter, GPU, WRF, or MPI execution occurs.  The emitted NPZ
contains only the four source-authorized State handles needed to remeasure the
single production PBL adapter on the fixed head: ustar, tau_u, tau_v, rhosfc.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping

from netCDF4 import Dataset
import numpy as np


REPO = Path(__file__).resolve().parent.parent
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SEAM = STAGED / "seam-capture"
INPUTS = STAGED / "inputs"
WRF_ROOT = STAGED / "wrf-dumps"
WRF_SOURCE = STAGED / "wrf-source/pristine"
WRFINPUT = INPUTS / "wrfinput_d03"
CARRY = INPUTS / "last-healthy-d03-step-0.pkl"
LANDUSE = WRF_SOURCE / "run/LANDUSE.TBL"
PHYSICS_INIT = WRF_SOURCE / "phys/module_physics_init.F"
MYNN_SOURCE = WRF_SOURCE / "phys/MYNN-EDMF/module_bl_mynnedmf.F90"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"
OUTER_READER = REPO / "scripts/v0234_gpt_operand_attribution.py"

EXPECTED = {
    "seam_manifest": "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873",
    "seam_canonical": "f6d14eee33d1f43364238c9a42849cb0993aaa71fa9b660c397d0c3ee7096d6a",
    "carry": "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "landuse": "7f661318ff5f06aed6b4b5508e4d077dc794cff03d5fb4447fe59c88d0ed2ece",
    "physics_init": "f48cf9793f782a72c3969e986eef24746252f030cd33cfba71cc82b81d128d90",
    "mynn_source": "6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72",
}
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
SOURCE_FILES = (
    "src/gpuwrf/io/land_state.py",
    "src/gpuwrf/physics/noah_mp.py",
    "src/gpuwrf/physics/noahmp_coupler.py",
    "src/gpuwrf/coupling/noahmp_surface_hook.py",
    "src/gpuwrf/physics/surface_layer.py",
    "src/gpuwrf/coupling/physics_couplers.py",
    "scripts/v0234_gpt_adversarial_audit.py",
    "scripts/v0234_gpt_operand_attribution.py",
    "scripts/v0234_gpt_sp2_surface_provenance.py",
)


class SurfaceProofFailure(RuntimeError):
    """Fail-closed evidence, source, or replay failure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def canonical_without_self(value: Mapping[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SurfaceProofFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False,
        text=not binary, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        stderr = result.stderr.decode() if binary else result.stderr
        raise SurfaceProofFailure(f"GIT:{' '.join(args)}:{stderr.strip()}")
    return result.stdout if binary else result.stdout.strip()


def tracked_record(relative: str) -> dict[str, Any]:
    path = REPO / relative
    if path.is_symlink() or not path.is_file():
        raise SurfaceProofFailure(f"SOURCE_NOT_REGULAR:{relative}")
    disk = path.read_bytes()
    head = git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise SurfaceProofFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": sha256_bytes(disk),
        "git_blob": git("rev-parse", f"HEAD:{relative}"),
        "size": len(disk),
    }


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    environment = {key: os.environ.get(key) for key in THREAD_ENV}
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if affinity != ALLOWED_CPUS:
        raise SurfaceProofFailure(f"CPUSET:{sorted(affinity)}")
    if environment != THREAD_ENV:
        raise SurfaceProofFailure(f"THREAD_ENV:{environment}")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SurfaceProofFailure("JAX_PLATFORM")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SurfaceProofFailure("CUDA_VISIBILITY")
    if os.environ.get("JAX_ENABLE_X64", "").lower() not in {"1", "true"}:
        raise SurfaceProofFailure("JAX_X64")
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise SurfaceProofFailure(f"PRIORITY:{nice}:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists():
        raise SurfaceProofFailure("PREEMPT_CPU")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": environment,
        "jax_platforms": "cpu",
        "cuda_visible_devices": "",
        "jax_enable_x64": os.environ["JAX_ENABLE_X64"],
        "nice": nice,
        "ionice_class": int(ioprio) >> 13,
        "preempt_cpu_absent": True,
    }


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if (
        lhs.shape != rhs.shape
        or lhs.dtype.hasobject
        or rhs.dtype.hasobject
        or not np.isfinite(lhs).all()
        or not np.isfinite(rhs).all()
    ):
        raise SurfaceProofFailure(f"METRIC_INPUT:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(rhs.astype(np.float64) ** 2)))
    return {
        "rms": rms,
        "max_abs": float(np.max(np.abs(delta))),
        "exact_fraction": float(np.mean(lhs == rhs)),
        "reference_rms": reference_rms,
        "relative_rms": rms / reference_rms if reference_rms else None,
    }


def split_metrics(left: np.ndarray, right: np.ndarray, land: np.ndarray) -> dict[str, Any]:
    return {
        "all": metrics(left, right),
        "land": metrics(np.asarray(left)[land], np.asarray(right)[land]),
        "water": metrics(np.asarray(left)[~land], np.asarray(right)[~land]),
    }


def validate_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SurfaceProofFailure("TRACKED_WORKTREE_NOT_CLEAN")
    sources = {relative: tracked_record(relative) for relative in SOURCE_FILES}
    for label, path in (
        ("carry", CARRY),
        ("wrfinput", WRFINPUT),
        ("landuse", LANDUSE),
        ("physics_init", PHYSICS_INIT),
        ("mynn_source", MYNN_SOURCE),
    ):
        if path.is_symlink() or not path.is_file():
            raise SurfaceProofFailure(f"INPUT_NOT_REGULAR:{path}")
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise SurfaceProofFailure(f"INPUT_HASH:{label}:{actual}")
    tree: dict[str, str] = {}
    for path in sorted(WRF_ROOT.rglob("*")):
        if path.is_symlink():
            raise SurfaceProofFailure(f"WRF_TREE_SYMLINK:{path}")
        if path.is_file():
            tree[path.relative_to(WRF_ROOT).as_posix()] = sha256_file(path)
    tree_sha = sha256_bytes(json.dumps(
        tree, sort_keys=True, separators=(",", ":"),
    ).encode())
    if len(tree) != 462 or tree_sha != EXPECTED["wrf_tree"]:
        raise SurfaceProofFailure(f"WRF_TREE:{len(tree)}:{tree_sha}")
    manifest_path = SEAM / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED["seam_manifest"]:
        raise SurfaceProofFailure("SEAM_MANIFEST_HASH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        canonical_without_self(manifest) == EXPECTED["seam_canonical"]
        and manifest.get("canonical_payload_sha256") == EXPECTED["seam_canonical"]
        and manifest.get("status") == "CAPTURE_COMPLETE_HOLD_5_OF_28"
        and manifest.get("seam", {}).get("trace_invocation_count") == 1
        and manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
        and manifest.get("provenance", {}).get("synthetic_reconstruction") is False
    ):
        raise SurfaceProofFailure("SEAM_AUTHORITY")
    authority = {
        "seam_manifest_sha256": EXPECTED["seam_manifest"],
        "seam_manifest_canonical_sha256": EXPECTED["seam_canonical"],
        "carry_sha256": EXPECTED["carry"],
        "wrfinput_d03_sha256": EXPECTED["wrfinput"],
        "wrf_dump_tree": {"file_count": len(tree), "sha256": tree_sha},
        "pristine_wrf_sources": {
            "LANDUSE.TBL": EXPECTED["landuse"],
            "module_physics_init.F": EXPECTED["physics_init"],
            "module_bl_mynnedmf.F90": EXPECTED["mynn_source"],
        },
    }
    return manifest, {"sources": sources, "authority": authority}


def load_seam_state(manifest: Mapping[str, Any], State: Any, jax: Any) -> Any:
    values: list[Any] = []
    for expected_index, item in enumerate(manifest.get("state_leaves", [])):
        if item.get("slot_index") != expected_index:
            raise SurfaceProofFailure(f"SEAM_SLOT_ORDER:{expected_index}")
        if item.get("kind") == "none":
            values.append(None)
            continue
        path = SEAM / item["file"]
        if path.is_symlink() or sha256_file(path) != item["file_sha256"]:
            raise SurfaceProofFailure(f"SEAM_LEAF_FILE:{item['slot_name']}")
        value = np.load(path, allow_pickle=False)
        if not (
            list(value.shape) == item["shape"]
            and value.dtype.str == item["dtype"]
            and array_sha(value) == item["logical_c_bitpayload_sha256"]
        ):
            raise SurfaceProofFailure(f"SEAM_LEAF_RECORD:{item['slot_name']}")
        values.append(jax.device_put(value))
    if len(values) != len(State.__slots__):
        raise SurfaceProofFailure(f"SEAM_STATE_SCHEMA:{len(values)}")
    return State.tree_unflatten(None, tuple(values))


def parse_modified_igbp(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("MODIFIED_IGBP_MODIS_NOAH")
    except ValueError as exc:
        raise SurfaceProofFailure("LANDUSE_SECTION") from exc
    header = [part.strip() for part in lines[start + 1].split(",", 2)]
    if header[:2] != ["61", "2"]:
        raise SurfaceProofFailure(f"LANDUSE_HEADER:{header}")
    cursor = start + 2
    result: dict[str, dict[int, dict[str, float]]] = {}
    for season in ("SUMMER", "WINTER"):
        if lines[cursor].strip() != season:
            raise SurfaceProofFailure(f"LANDUSE_SEASON_ORDER:{cursor}:{lines[cursor]}")
        cursor += 1
        rows: dict[int, dict[str, float]] = {}
        for category in range(1, 62):
            numeric = lines[cursor].split("'", 1)[0].rstrip(" ,")
            fields = [part.strip() for part in numeric.split(",")]
            if len(fields) != 8 or int(fields[0]) != category:
                raise SurfaceProofFailure(f"LANDUSE_ROW:{season}:{category}:{fields}")
            rows[category] = {
                "slmo": float(fields[2]),
                "sfz0_m": float(fields[4]) / 100.0,
            }
            cursor += 1
        result[season.lower()] = rows
    return result


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SurfaceProofFailure(f"IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_correction(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise SurfaceProofFailure(f"CORRECTION_NOT_FRESH:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        name: np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        for name, value in arrays.items()
    }
    for name, value in clean.items():
        if value.shape != (93, 111) or not np.isfinite(value).all():
            raise SurfaceProofFailure(f"CORRECTION_ARRAY:{name}:{value.shape}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez(temporary, **clean)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "logical_c_bitpayload_sha256": array_sha(value),
            }
            for name, value in clean.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--correction-output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    correction_output = args.correction_output.resolve()
    try:
        resources = resource_gate()
        manifest, static = validate_static_inputs()

        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.io.land_state import _landuse_season_for_run
        from gpuwrf.physics.noah_mp import (
            mavail_from_prescribed_fields,
            roughness_from_prescribed_fields,
        )
        from gpuwrf.physics.noahmp_coupler import (
            _mynn_pbl_surface_density,
            assemble_noahmp_forcing,
        )
        from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

        if jax.default_backend() != "cpu" or any(
            device.platform != "cpu" for device in jax.devices()
        ):
            raise SurfaceProofFailure("NON_CPU_BACKEND")
        if not bool(jax.config.jax_enable_x64):
            raise SurfaceProofFailure("JAX_X64_DISABLED")

        seam_state = load_seam_state(manifest, State, jax)
        with CARRY.open("rb") as stream:
            carry = pickle.load(stream)
        if type(carry).__name__ != "OperationalCarry" or not hasattr(carry, "state"):
            raise SurfaceProofFailure(f"CARRY_SCHEMA:{type(carry)}")
        initial = carry.state

        run = Gen2Run(INPUTS)
        grid = run.grid("d03").as_grid_spec()
        from dataclasses import replace as dataclass_replace
        grid = dataclass_replace(grid, metrics=load_wrfinput_metrics(WRFINPUT))
        season, julday, cen_lat = _landuse_season_for_run(run, "d03")
        if (season, julday) != (2, 60) or not cen_lat > 0.0:
            raise SurfaceProofFailure(f"WRF_SEASON:{season}:{julday}:{cen_lat}")

        with Dataset(WRFINPUT) as dataset:
            landmask = np.asarray(dataset["LANDMASK"][0], dtype=np.float64)
            wrf_lu = np.asarray(dataset["LU_INDEX"][0], dtype=np.int32)
            wrf_xland = np.asarray(dataset["XLAND"][0], dtype=np.float64)
        if (
            metrics(np.asarray(initial.xland), wrf_xland)["max_abs"] != 0.0
            or metrics(np.asarray(initial.lu_index), wrf_lu)["max_abs"] != 0.0
        ):
            raise SurfaceProofFailure("CARRY_WRFINPUT_LAND_AUTHORITY")

        parsed = parse_modified_igbp(LANDUSE)
        cats = np.arange(1, 62, dtype=np.float64).reshape(1, 61)
        ones = np.ones_like(cats)
        soil = np.ones((1, 1, 61), dtype=np.float64)
        production_tables: dict[str, dict[str, np.ndarray]] = {}
        for number, name in ((1, "summer"), (2, "winter")):
            production_tables[name] = {
                "sfz0_m": np.asarray(roughness_from_prescribed_fields(
                    ones, ones, lu_index=cats, season=number,
                )).reshape(-1),
                "slmo": np.asarray(mavail_from_prescribed_fields(
                    ones, ones, soil, lu_index=cats, season=number,
                )).reshape(-1),
            }
            for field in ("sfz0_m", "slmo"):
                authority = np.asarray([
                    parsed[name][category][field] for category in range(1, 62)
                ])
                if metrics(production_tables[name][field], authority)["max_abs"] > 5.0e-15:
                    raise SurfaceProofFailure(f"PRODUCTION_TABLE:{name}:{field}")

        surface_fields = (
            "t_skin", "soil_moisture", "xland", "lakemask", "mavail",
            "roughness_m", "ustar",
        )
        old_state = seam_state.replace(**{
            name: getattr(initial, name) for name in surface_fields
        })
        old_z0_native = np.asarray(initial.roughness_m)
        old_mavail_native = np.asarray(initial.mavail)
        summer_z0 = np.asarray(roughness_from_prescribed_fields(
            initial.xland, landmask, lu_index=wrf_lu, season=1,
        ))
        summer_mavail = np.asarray(mavail_from_prescribed_fields(
            initial.xland, landmask, initial.soil_moisture,
            lu_index=wrf_lu, season=1,
        ))
        summer_closure = {
            "roughness_m": metrics(
                summer_z0.astype(old_z0_native.dtype), old_z0_native,
            ),
            "mavail": metrics(
                summer_mavail.astype(np.float32).astype(old_mavail_native.dtype),
                old_mavail_native,
            ),
        }
        if any(value["max_abs"] != 0.0 for value in summer_closure.values()):
            raise SurfaceProofFailure(f"OLD_SUMMER_CLOSURE:{summer_closure}")

        winter_z0 = np.asarray(roughness_from_prescribed_fields(
            initial.xland, landmask, lu_index=wrf_lu, season=2,
        ))
        winter_mavail = np.asarray(mavail_from_prescribed_fields(
            initial.xland, landmask, initial.soil_moisture,
            lu_index=wrf_lu, season=2,
        ))
        independent_z0 = np.asarray([
            parsed["winter"][int(category)]["sfz0_m"]
            for category in wrf_lu.reshape(-1)
        ]).reshape(wrf_lu.shape)
        independent_mavail = np.asarray([
            parsed["winter"][int(category)]["slmo"]
            for category in wrf_lu.reshape(-1)
        ]).reshape(wrf_lu.shape)
        winter_authority_closure = {
            "roughness_m": metrics(winter_z0, independent_z0),
            "mavail": metrics(winter_mavail, independent_mavail),
        }
        if any(value["max_abs"] > 5.0e-15 for value in winter_authority_closure.values()):
            raise SurfaceProofFailure(f"WINTER_AUTHORITY:{winter_authority_closure}")

        new_state = old_state.replace(
            roughness_m=jnp.asarray(winter_z0),
            mavail=jnp.asarray(winter_mavail),
        )
        old_view = _build_column_view(old_state, grid)
        new_view = _build_column_view(new_state, grid)
        old_diag = surface_layer_with_diagnostics(old_view, first_timestep=True)
        new_diag = surface_layer_with_diagnostics(new_view, first_timestep=True)
        old_ust = np.asarray(old_diag.fluxes.ustar)
        new_ust = np.asarray(new_diag.fluxes.ustar)
        seam_ust = np.asarray(seam_state.ustar)
        replay_closure = {
            "ustar": metrics(old_ust, seam_ust),
            "tau_u": metrics(np.asarray(old_diag.fluxes.tau_u), np.asarray(seam_state.tau_u)),
            "tau_v": metrics(np.asarray(old_diag.fluxes.tau_v), np.asarray(seam_state.tau_v)),
        }
        if any(value["rms"] > 5.0e-13 for value in replay_closure.values()):
            raise SurfaceProofFailure(f"AUTHENTIC_REPLAY:{replay_closure}")

        audit = import_file("v0234_sp2_surface_dump_reader", AUDIT_READER)
        outer = import_file("v0234_sp2_surface_outer_reader", OUTER_READER)
        outer.WRF_ROOT = WRF_ROOT
        reader = audit.IndependentWrfDump(WRF_ROOT)
        wrf_ust = outer.load_outer2(reader, "ust")
        wrf_wspd = outer.load_outer2(reader, "wspd")
        lower = reader.columns("bc_lower_operands")
        if lower.shape != (32, 93, 111):
            raise SurfaceProofFailure(f"LOWER_OPERANDS:{lower.shape}")
        land = wrf_xland < 1.5
        ust_before = split_metrics(old_ust, wrf_ust, land)
        ust_after = split_metrics(new_ust, wrf_ust, land)
        water_invariance = metrics(new_ust[~land], old_ust[~land])
        if not (
            ust_after["all"]["rms"] < ust_before["all"]["rms"]
            and ust_after["land"]["rms"] < ust_before["land"]["rms"]
            and water_invariance["max_abs"] == 0.0
        ):
            raise SurfaceProofFailure(
                f"UST_IMPROVEMENT:{ust_before['all']['rms']}:"
                f"{ust_after['all']['rms']}:{water_invariance['max_abs']}"
            )

        forcing = assemble_noahmp_forcing(new_view, None, None, None, 6.0)
        pbl_density = np.asarray(_mynn_pbl_surface_density(forcing))
        independent_port_density = (
            np.asarray(new_view.psfc, dtype=np.float64)
            / (287.0 * (
                np.asarray(new_view.t_air, dtype=np.float64)[..., 0]
                + 0.608 * np.asarray(new_view.qv, dtype=np.float64)[..., 0]
            ))
        )
        if metrics(pbl_density, independent_port_density)["max_abs"] > 5.0e-15:
            raise SurfaceProofFailure("PORT_DENSITY_FORMULA")
        wrf_formula_density = (
            lower[0]
            / (np.float32(287.0) * (lower[1] + np.float32(0.608) * lower[2]))
        ).astype(np.float32)
        wrf_density_formula_closure = metrics(wrf_formula_density, lower[10])
        density_before = metrics(np.asarray(seam_state.rhosfc), lower[10])
        density_after = metrics(pbl_density, lower[10])
        if not (
            wrf_density_formula_closure["rms"] <= 2.0e-7
            and density_after["rms"] < density_before["rms"]
        ):
            raise SurfaceProofFailure(
                f"DENSITY_GATE:{wrf_density_formula_closure['rms']}:"
                f"{density_before['rms']}:{density_after['rms']}"
            )

        port_wind = np.maximum(np.sqrt(
            np.asarray(new_view.u)[..., 0] ** 2
            + np.asarray(new_view.v)[..., 0] ** 2
        ), 0.2)
        wind_closure = metrics(port_wind, wrf_wspd)
        wrf_drag = lower[10] * lower[6] * lower[6] / lower[7]
        drag_old = (
            np.asarray(seam_state.rhosfc) * seam_ust * seam_ust / port_wind
        )
        drag_ust_only = (
            np.asarray(seam_state.rhosfc) * new_ust * new_ust / port_wind
        )
        drag_density_only = pbl_density * seam_ust * seam_ust / port_wind
        drag_corrected = pbl_density * new_ust * new_ust / port_wind
        drag_metrics = {
            "old": metrics(drag_old, wrf_drag),
            "winter_ust_only": metrics(drag_ust_only, wrf_drag),
            "mynn_density_only": metrics(drag_density_only, wrf_drag),
            "combined": metrics(drag_corrected, wrf_drag),
        }
        if not drag_metrics["combined"]["rms"] < drag_metrics["old"]["rms"]:
            raise SurfaceProofFailure(f"DRAG_IMPROVEMENT:{drag_metrics}")

        active_categories = []
        for category in sorted(int(value) for value in np.unique(wrf_lu[land])):
            mask = land & (wrf_lu == category)
            active_categories.append({
                "category": category,
                "cell_count": int(np.sum(mask)),
                "summer_sfz0_m": parsed["summer"][category]["sfz0_m"],
                "winter_sfz0_m": parsed["winter"][category]["sfz0_m"],
                "summer_slmo": parsed["summer"][category]["slmo"],
                "winter_slmo": parsed["winter"][category]["slmo"],
            })

        correction = save_correction(correction_output, {
            "ustar": new_ust,
            "tau_u": np.asarray(new_diag.fluxes.tau_u),
            "tau_v": np.asarray(new_diag.fluxes.tau_v),
            "rhosfc": pbl_density,
        })
        proof = {
            "schema": "wrfgpu2-v0234-gpt-sp2-surface-provenance-v1",
            "verdict": "WRF_SEASONAL_SURFACE_AND_MYNN_DENSITY_FIX_PROVEN",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "backend": "jax-cpu-and-numpy-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_pbl_adapter_invocations": 0,
            "git": {
                "head": git("rev-parse", "HEAD"),
                "tree": git("rev-parse", "HEAD^{tree}"),
                "branch": git("branch", "--show-current"),
                "tracked_worktree_clean": True,
            },
            "resource_gate": resources,
            **static,
            "wrf_source_rules": {
                "season": {
                    "location": "phys/module_physics_init.F:1833-1835",
                    "rule": "ISN=2 for JULDAY<105 or JULDAY>288; invert in Southern Hemisphere",
                    "fixture": {"julday": julday, "cen_lat": cen_lat, "isn": season},
                },
                "surface_fields": {
                    "location": "phys/module_physics_init.F:1968,1972",
                    "rule": "Z0=SFZ0(IS,ISN)/100; MAVAIL=SLMO(IS,ISN)",
                },
                "mynn_density": {
                    "location": "phys/MYNN-EDMF/module_bl_mynnedmf.F90:3960",
                    "rule": "rhosfc=psfc/(R_d*(tk(kts)+p608*qv(kts)))",
                },
            },
            "landuse_table_authority": {
                "section": "MODIFIED_IGBP_MODIS_NOAH",
                "categories": 61,
                "seasons": 2,
                "production_summer_matches_independent_parse": True,
                "production_winter_matches_independent_parse": True,
                "initial_carry_is_exact_summer_table": summer_closure,
                "corrected_fields_match_winter_table": winter_authority_closure,
                "active_land_categories": active_categories,
                "land_cells": int(np.sum(land)),
                "water_cells": int(np.sum(~land)),
            },
            "authenticated_surface_replay": {
                "common_atmosphere": "sealed post-QKE pre-PBL State",
                "common_lower_boundary": "authenticated step-0 operational carry",
                "first_timestep": True,
                "old_replay_vs_seam": replay_closure,
                "old_summer_vs_pristine_wrf_ust": ust_before,
                "corrected_winter_vs_pristine_wrf_ust": ust_after,
                "water_old_vs_corrected": water_invariance,
                "ust_all_rms_improvement_factor": (
                    ust_before["all"]["rms"] / ust_after["all"]["rms"]
                ),
                "wind_vs_pristine_wrf": wind_closure,
            },
            "mynn_density_proof": {
                "wrf_formula_recomputed_vs_dumped_rhosfc": wrf_density_formula_closure,
                "old_surface_driver_density_vs_pristine_wrf": density_before,
                "corrected_mynn_density_vs_pristine_wrf": density_after,
                "port_inputs_vs_dumped_wrf": {
                    "psfc": metrics(np.asarray(new_view.psfc), lower[0]),
                    "tk": metrics(np.asarray(new_view.t_air)[..., 0], lower[1]),
                    "qv": metrics(np.asarray(new_view.qv)[..., 0], lower[2]),
                },
            },
            "surface_drag_discriminator": drag_metrics,
            "correction_artifact": correction,
            "correction_contract": {
                "changed_state_handles": ["ustar", "tau_u", "tau_v", "rhosfc"],
                "unchanged_seam_handles": ["theta_flux", "qv_flux", "fltv"],
                "synthetic_reconstruction": False,
                "empirical_tuning": False,
                "source_authorized": True,
            },
            "gates": {
                "old_surface_replay_rms_le_5e-13": True,
                "winter_ust_strictly_improves_all_and_land": True,
                "water_ust_bitwise_unchanged": True,
                "wrf_density_formula_rms_le_2e-7": True,
                "combined_surface_drag_strictly_improves": True,
            },
            "passed": True,
        }
        atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "correction": correction,
            "ust_rms_before": ust_before["all"]["rms"],
            "ust_rms_after": ust_after["all"]["rms"],
            "density_rms_before": density_before["rms"],
            "density_rms_after": density_after["rms"],
            "drag_rms_before": drag_metrics["old"]["rms"],
            "drag_rms_after": drag_metrics["combined"]["rms"],
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
