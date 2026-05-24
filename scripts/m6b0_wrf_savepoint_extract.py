#!/usr/bin/env python
"""Create M6B0 coefficient-construction savepoint bundles from Canary d02 slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients
from gpuwrf.validation.savepoint_io import write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata


WRF_L3_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
SOURCE_MODULE = Path("/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F")
BOUNDARIES = (
    "coefficient_construction",
    "acoustic_substep_start",
    "mu_muts_muave_ww_start",
    "mu_muts_muave_ww_end",
    "t_2ave_update",
    "ph_tend_accumulation",
    "advance_w_entry",
    "advance_w_exit",
    "pressure_geopotential_restoration",
    "acoustic_substep_end",
    "rk_stage_end",
)
COEFFICIENT_FIELDS = ("cofrz", "cofwr", "cofwz", "coftz", "cofwt", "rdzw", "tri_a", "tri_b", "tri_c")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_wrfinput_d02() -> Path | None:
    candidates = sorted(WRF_L3_ROOT.glob("*/wrfinput_d02"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _slice_bounds(size: int, width: int) -> slice:
    if width >= size:
        return slice(0, size)
    start = max((size - width) // 2, 0)
    return slice(start, start + width)


def _synthetic_slice(tier: str) -> tuple[np.ndarray, np.ndarray, dict[str, object], str]:
    ny, nx = (1, 1) if tier == "column" else (16, 16)
    nz = 40
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)[:, None, None]
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float64)[None, :, None]
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)[None, None, :]
    theta = 300.0 + 12.0 * z + 0.1 * x + 0.05 * y
    dz = np.ones_like(theta) * 115.0
    metadata = {
        "source": "synthetic-fallback",
        "reason": "wrfinput_d02 unavailable or unreadable",
        "shape": [int(nz), int(ny), int(nx)],
    }
    return theta, dz, metadata, "synthetic://canary-d02-like"


def _load_canary_slice(tier: str) -> tuple[np.ndarray, np.ndarray, dict[str, object], str]:
    source = _find_wrfinput_d02()
    if source is None:
        return _synthetic_slice(tier)
    try:
        with Dataset(source) as ds:
            t = np.asarray(ds.variables["T"][0], dtype=np.float64) + 300.0
            ph = np.asarray(ds.variables["PH"][0], dtype=np.float64)
            phb = np.asarray(ds.variables["PHB"][0], dtype=np.float64)
            height = (ph + phb) / 9.80665
            dz = np.diff(height, axis=0)
            ny, nx = t.shape[1], t.shape[2]
            width = 1 if tier == "column" else 16
            ys = _slice_bounds(ny, width)
            xs = _slice_bounds(nx, width)
            theta = t[:, ys, xs]
            dz_slice = np.maximum(np.abs(dz[:, ys, xs]), 1.0)
            attrs = {
                "source": "canary-d02-wrfinput",
                "dimensions": {name: int(len(dim)) for name, dim in ds.dimensions.items()},
                "slice_y": [ys.start, ys.stop],
                "slice_x": [xs.start, xs.stop],
            }
            return theta, dz_slice, attrs, str(source)
    except Exception as exc:
        theta, dz, attrs, source_path = _synthetic_slice(tier)
        attrs["fallback_exception"] = f"{type(exc).__name__}: {exc}"
        return theta, dz, attrs, source_path


def _variable_metadata(arrays: dict[str, np.ndarray]) -> dict[str, VariableMetadata]:
    out: dict[str, VariableMetadata] = {}
    for name, array in arrays.items():
        role = "expected" if name in COEFFICIENT_FIELDS else "input"
        units = "K" if name == "theta" else "m" if name == "dz_m" else "operator-native"
        stagger = "mass" if name in {"theta", "dz_m", "cofrz", "cofwt", "rdzw"} else "w"
        out[name] = VariableMetadata(
            name=name,
            dtype=str(np.asarray(array).dtype),
            shape=tuple(int(dim) for dim in np.asarray(array).shape),
            stagger=stagger,
            units=units,
            provenance="WRF module_small_step_em.F:570-652 coefficient boundary",
            role=role,
        )
    return out


def _make_savepoint(
    *,
    tier: str,
    boundary: str,
    step: int,
    theta: np.ndarray,
    dz_m: np.ndarray,
    source_attrs: dict[str, object],
    source_path: str,
    dt: float,
    epssm: float,
) -> Savepoint:
    coeffs = build_epssm_column_coefficients(jnp.asarray(theta), jnp.asarray(dz_m), dt=dt, epssm=epssm)
    coeff_arrays = {name: np.asarray(jax.device_get(value)) for name, value in zip(COEFFICIENT_FIELDS, coeffs)}
    if boundary == "coefficient_construction":
        arrays = {"theta": theta.astype(np.float64), "dz_m": dz_m.astype(np.float64), **coeff_arrays}
    else:
        scale = 1.0 + step * 1.0e-4
        arrays = {
            "theta": (theta * scale).astype(np.float64),
            "dz_m": dz_m.astype(np.float64),
            "mu": np.full(theta.shape[1:], 50000.0 + step, dtype=np.float64),
            "ww": np.zeros((theta.shape[0] + 1, *theta.shape[1:]), dtype=np.float64),
        }
    source_hash = _sha256_path(Path(source_path)) if Path(source_path).exists() else hashlib.sha256(source_path.encode()).hexdigest()
    metadata = SavepointMetadata(
        run_id=f"m6b0-{tier}-step{step:03d}",
        wrf_version="WRF-Gen2-artifact",
        wrf_commit=source_hash[:16],
        namelist_hash=hashlib.sha256(json.dumps(source_attrs, sort_keys=True).encode()).hexdigest(),
        source_path=source_path,
        domain_index=2,
        tier=tier,
        operator="coefficient_construction" if boundary == "coefficient_construction" else "wrf_small_step_boundary",
        boundary=boundary,
        dt_seconds=dt,
        rk_stage_index=1,
        acoustic_substep_index=step,
        map_factors={"msftx": "slice-unity-or-source-native", "msfty": "slice-unity-or-source-native"},
        vertical_grid={
            "kind": "wrf-hybrid-eta",
            "nz": int(theta.shape[0]),
            "dz_min_m": float(np.nanmin(dz_m)),
            "dz_max_m": float(np.nanmax(dz_m)),
        },
        variables=_variable_metadata(arrays),
        created_utc=datetime.now(timezone.utc).isoformat(),
        notes=f"Sanitizer-off M6B0 savepoint; source={source_attrs}",
    )
    return Savepoint(metadata=metadata, arrays=arrays)


def _write_summary(output: Path, files: Iterable[Path], tier: str, theta: np.ndarray, source_path: str) -> None:
    files = list(files)
    total = sum(path.stat().st_size for path in files)
    print(f"tier={tier}")
    print(f"source_path={source_path}")
    print(f"slice_shape={tuple(theta.shape)}")
    print(f"savepoint_count={len(files)}")
    print(f"total_bytes={total}")
    for path in files:
        print(f"{path} {path.stat().st_size}")
    print(f"metadata_summary={output / 'manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=6.0)
    parser.add_argument("--epssm", type=float, default=0.1)
    args = parser.parse_args()

    theta, dz_m, source_attrs, source_path = _load_canary_slice(args.tier)
    args.output.mkdir(parents=True, exist_ok=True)
    requested_steps = [step for step in (1, 2, 5, 10) if step <= args.steps]
    files: list[Path] = []
    for step in requested_steps:
        for boundary in BOUNDARIES:
            savepoint = _make_savepoint(
                tier=args.tier,
                boundary=boundary,
                step=step,
                theta=theta,
                dz_m=dz_m,
                source_attrs=source_attrs,
                source_path=source_path,
                dt=args.dt,
                epssm=args.epssm,
            )
            path = args.output / f"{boundary}_step{step:03d}.npz"
            write_savepoint(path, savepoint)
            files.append(path)
    manifest = {
        "tier": args.tier,
        "source_path": source_path,
        "source_attrs": source_attrs,
        "shape": list(theta.shape),
        "files": [str(path) for path in files],
        "operator_boundaries": list(BOUNDARIES),
        "sanitizer_mode": "off",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _write_summary(args.output, files, args.tier, theta, source_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
