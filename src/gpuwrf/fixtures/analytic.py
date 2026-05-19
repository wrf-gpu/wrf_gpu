"""Deterministic analytic M1 fixture generators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import zipfile

import numpy as np


STENCIL_ID = "analytic-stencil-3d-advdiff-v1"
COLUMN_ID = "analytic-column-thermo-v1"
CREATED_UTC = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class VariableMeta:
    name: str
    units: str
    staggering: str
    dtype: str
    tolerance_abs: float
    tolerance_rel: float
    tolerance_rationale: str


@dataclass(frozen=True)
class FixturePayload:
    fixture_id: str
    scenario: str
    arrays: dict[str, np.ndarray]
    variables: list[VariableMeta]


def _ddx4(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return (
        -np.roll(field, -2, axis=axis)
        + 8.0 * np.roll(field, -1, axis=axis)
        - 8.0 * np.roll(field, 1, axis=axis)
        + np.roll(field, 2, axis=axis)
    ) / (12.0 * spacing)


def _ddx2(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)


def _lap4(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return (
        -np.roll(field, -2, axis=axis)
        + 16.0 * np.roll(field, -1, axis=axis)
        - 30.0 * field
        + 16.0 * np.roll(field, 1, axis=axis)
        - np.roll(field, 2, axis=axis)
    ) / (12.0 * spacing * spacing)


def _lap2(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return (np.roll(field, -1, axis=axis) - 2.0 * field + np.roll(field, 1, axis=axis)) / (spacing * spacing)


def generate_stencil_fixture(seed: int = 0) -> FixturePayload:
    """Generate a small 3D advection-diffusion reference update."""

    rng = np.random.default_rng(seed)
    nz, ny, nx = 8, 16, 32
    dx, dy, dz, dt = 900.0, 900.0, 120.0, 3.0
    x = np.arange(nx, dtype=np.float64)[None, None, :]
    y = np.arange(ny, dtype=np.float64)[None, :, None]
    z = np.arange(nz, dtype=np.float64)[:, None, None]
    phase = rng.uniform(-0.25, 0.25, size=3)

    phi_initial = (
        290.0
        + 4.0 * np.sin(2.0 * np.pi * (x / nx + phase[0]))
        + 2.0 * np.cos(2.0 * np.pi * (y / ny + phase[1]))
        + 0.25 * z
        + 0.5 * np.sin(2.0 * np.pi * (x / nx + y / ny + z / nz + phase[2]))
    ).astype(np.float64)

    xu = np.arange(nx + 1, dtype=np.float64)[None, None, :]
    yv = np.arange(ny + 1, dtype=np.float64)[None, :, None]
    zw = np.arange(nz + 1, dtype=np.float64)[:, None, None]
    u_face = (7.0 + 1.1 * np.sin(2.0 * np.pi * xu / nx) + 0.2 * np.cos(2.0 * np.pi * y / ny)).astype(np.float32)
    u_face = np.broadcast_to(u_face, (nz, ny, nx + 1)).copy()
    v_face = (-2.0 + 0.8 * np.cos(2.0 * np.pi * yv / ny) + 0.15 * np.sin(2.0 * np.pi * x / nx)).astype(np.float32)
    v_face = np.broadcast_to(v_face, (nz, ny + 1, nx)).copy()
    w_face = (0.08 * np.sin(2.0 * np.pi * zw / nz) + 0.02 * np.cos(2.0 * np.pi * x / nx)).astype(np.float32)
    w_face = np.broadcast_to(w_face, (nz + 1, ny, nx)).copy()

    u_mass = 0.5 * (u_face[:, :, :-1].astype(np.float64) + u_face[:, :, 1:].astype(np.float64))
    v_mass = 0.5 * (v_face[:, :-1, :].astype(np.float64) + v_face[:, 1:, :].astype(np.float64))
    w_mass = 0.5 * (w_face[:-1, :, :].astype(np.float64) + w_face[1:, :, :].astype(np.float64))
    diffusivity = 18.0 + 2.0 * np.sin(2.0 * np.pi * z / nz)

    advection = u_mass * _ddx4(phi_initial, dx, axis=2) + v_mass * _ddx4(phi_initial, dy, axis=1) + w_mass * _ddx2(phi_initial, dz, axis=0)
    diffusion = diffusivity * (_lap4(phi_initial, dx, axis=2) + _lap4(phi_initial, dy, axis=1) + _lap2(phi_initial, dz, axis=0))
    phi_next = (phi_initial + dt * (-advection + diffusion)).astype(np.float64)

    arrays = {
        "phi_initial": phi_initial,
        "u_face": u_face,
        "v_face": v_face,
        "w_face": w_face,
        "phi_next": phi_next,
    }
    variables = [
        VariableMeta("phi_initial", "K", "mass", "float64", 1.0e-10, 0.0, "fp64 analytic field; strict absolute tolerance"),
        VariableMeta("u_face", "m s-1", "u", "float32", 1.0e-6, 1.0e-7, "stored fp32 face velocity; compare at fp32 scale"),
        VariableMeta("v_face", "m s-1", "v", "float32", 1.0e-6, 1.0e-7, "stored fp32 face velocity; compare at fp32 scale"),
        VariableMeta("w_face", "m s-1", "w", "float32", 1.0e-7, 1.0e-7, "small stored fp32 face velocity; strict absolute tolerance"),
        VariableMeta("phi_next", "K", "mass", "float64", 1.0e-10, 1.0e-12, "fp64 NumPy reference update; strict oracle tolerance"),
    ]
    return FixturePayload(
        fixture_id=STENCIL_ID,
        scenario="32x16x8 periodic staggered-grid advection-diffusion single timestep",
        arrays=arrays,
        variables=variables,
    )


def generate_column_fixture(seed: int = 0) -> FixturePayload:
    """Generate a single-column thermodynamic source update."""

    rng = np.random.default_rng(seed)
    levels = np.arange(40, dtype=np.float64)
    sigma = levels / 39.0
    pressure_initial = (100000.0 * np.exp(-2.35 * sigma)).astype(np.float64)
    temperature_initial = (
        301.0
        - 58.0 * sigma
        + 1.5 * np.sin(2.0 * np.pi * sigma + rng.uniform(-0.1, 0.1))
        + 0.35 * np.cos(5.0 * np.pi * sigma)
    ).astype(np.float64)
    qv_initial = (0.015 * np.exp(-3.0 * sigma) + 0.0015 * np.sin(3.0 * np.pi * sigma + 0.2)).astype(np.float64)
    qv_initial = np.maximum(qv_initial, 1.0e-5)

    saturation_qv = (0.006 + 0.009 * np.exp(-4.0 * sigma) + 0.001 * np.cos(2.0 * np.pi * sigma)).astype(np.float64)
    excess = np.maximum(qv_initial - saturation_qv, 0.0)
    deficit = np.maximum(0.72 * saturation_qv - qv_initial, 0.0)
    condensation = 0.32 * excess
    evaporation = np.minimum(0.04 * deficit, 0.18 * qv_initial)
    qv_next = np.maximum(qv_initial - condensation + evaporation, 1.0e-8)

    cp_d = 1004.0
    lv = 2.5e6
    latent_mass = condensation - evaporation
    temperature_next = (temperature_initial + (lv / cp_d) * latent_mass).astype(np.float64)
    pressure_next = pressure_initial.copy()
    mse_delta = (cp_d * (temperature_next - temperature_initial) + lv * (qv_next - qv_initial)).astype(np.float64)

    arrays = {
        "temperature_initial": temperature_initial,
        "qv_initial": qv_initial,
        "pressure_initial": pressure_initial,
        "saturation_qv": saturation_qv,
        "temperature_next": temperature_next,
        "qv_next": qv_next,
        "pressure_next": pressure_next,
        "mse_delta": mse_delta,
    }
    variables = [
        VariableMeta("temperature_initial", "K", "mass", "float64", 1.0e-11, 0.0, "fp64 deterministic column input"),
        VariableMeta("qv_initial", "kg kg-1", "mass", "float64", 1.0e-13, 1.0e-12, "fp64 deterministic vapor input"),
        VariableMeta("pressure_initial", "Pa", "mass", "float64", 1.0e-8, 1.0e-12, "fp64 pressure profile input"),
        VariableMeta("saturation_qv", "kg kg-1", "mass", "float64", 1.0e-13, 1.0e-12, "fp64 branch threshold input"),
        VariableMeta("temperature_next", "K", "mass", "float64", 1.0e-11, 1.0e-12, "fp64 latent-heating reference output"),
        VariableMeta("qv_next", "kg kg-1", "mass", "float64", 1.0e-13, 1.0e-12, "fp64 vapor reference output"),
        VariableMeta("pressure_next", "Pa", "mass", "float64", 1.0e-8, 1.0e-12, "pressure held fixed by analytic operator"),
        VariableMeta("mse_delta", "J kg-1", "mass", "float64", 1.0e-10, 1.0e-12, "moist static energy residual should stay near roundoff"),
    ]
    return FixturePayload(
        fixture_id=COLUMN_ID,
        scenario="40-level analytic moist thermodynamic column with branch-dependent phase source",
        arrays=arrays,
        variables=variables,
    )


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a byte-stable .npz archive with fixed zip metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 5, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(payload: FixturePayload, sample_path: Path, seed: int) -> dict[str, object]:
    rel_sample = sample_path.as_posix()
    return {
        "fixture_id": payload.fixture_id,
        "source": "analytic",
        "source_commit": f"analytic.py seed {seed}",
        "wrf_version": None,
        "scenario": payload.scenario,
        "created_utc": CREATED_UTC,
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": f"python scripts/generate_analytic_fixtures.py --seed {seed} --out fixtures/samples/",
        "external_uri": None,
        "sample_slice_path": rel_sample,
        "git_commit": "worker/gpt/m1-analytic-fixtures",
        "license_notes": "Analytic synthetic fixture generated in-repository; no external data license.",
        "variables": [
            {
                "name": meta.name,
                "units": meta.units,
                "shape": [int(dim) for dim in payload.arrays[meta.name].shape],
                "staggering": meta.staggering,
                "dtype": meta.dtype,
                "tolerance_abs": meta.tolerance_abs,
                "tolerance_rel": meta.tolerance_rel,
                "tolerance_rationale": meta.tolerance_rationale,
                "tier_overrides": None,
            }
            for meta in payload.variables
        ],
        "files": [
            {
                "path": rel_sample,
                "checksum_sha256": _sha256(sample_path),
                "bytes": sample_path.stat().st_size,
                "external": False,
            }
        ],
    }


def _format_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if "e" in text and "." not in text.split("e", 1)[0]:
            text = text.replace("e", ".0e", 1)
        return text
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"unsupported scalar {type(value).__name__}")


def _yaml_lines(value: object, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return lines
    return [f"{prefix}{_format_scalar(value)}"]


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_yaml_lines(manifest)) + "\n", encoding="utf-8")


def generate_all(seed: int, sample_dir: Path, manifest_dir: Path) -> list[Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for payload in (generate_stencil_fixture(seed), generate_column_fixture(seed)):
        sample_path = sample_dir / f"{payload.fixture_id}.npz"
        write_deterministic_npz(sample_path, payload.arrays)
        manifest_path = manifest_dir / f"{payload.fixture_id}.yaml"
        write_manifest(manifest_path, _manifest(payload, sample_path, seed))
        generated.extend([sample_path, manifest_path])
    return generated
