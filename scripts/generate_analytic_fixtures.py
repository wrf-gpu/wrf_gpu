#!/usr/bin/env python3
"""Generate the M1 analytic fixture manifests and committed sample slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.fixtures.analytic import CREATED_UTC, generate_all, generate_stencil_fixture, write_deterministic_npz, write_manifest  # noqa: E402


UPWIND5_ID = "analytic-stencil-3d-upwind5-v1"


def _derivative5_upwind(field: np.ndarray, velocity: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Matches the M4 fifth-order periodic upwind derivative in NumPy."""

    backward = (
        137.0 * field
        - 300.0 * np.roll(field, 1, axis=axis)
        + 300.0 * np.roll(field, 2, axis=axis)
        - 200.0 * np.roll(field, 3, axis=axis)
        + 75.0 * np.roll(field, 4, axis=axis)
        - 12.0 * np.roll(field, 5, axis=axis)
    ) / (60.0 * spacing)
    forward = (
        -137.0 * field
        + 300.0 * np.roll(field, -1, axis=axis)
        - 300.0 * np.roll(field, -2, axis=axis)
        + 200.0 * np.roll(field, -3, axis=axis)
        - 75.0 * np.roll(field, -4, axis=axis)
        + 12.0 * np.roll(field, -5, axis=axis)
    ) / (60.0 * spacing)
    return np.where(velocity >= 0.0, backward, forward)


def _derivative3_upwind(field: np.ndarray, velocity: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Matches the M4 third-order periodic vertical upwind derivative in NumPy."""

    backward = (
        11.0 * field
        - 18.0 * np.roll(field, 1, axis=axis)
        + 9.0 * np.roll(field, 2, axis=axis)
        - 2.0 * np.roll(field, 3, axis=axis)
    ) / (6.0 * spacing)
    forward = (
        -11.0 * field
        + 18.0 * np.roll(field, -1, axis=axis)
        - 9.0 * np.roll(field, -2, axis=axis)
        + 2.0 * np.roll(field, -3, axis=axis)
    ) / (6.0 * spacing)
    return np.where(velocity >= 0.0, backward, forward)


def _derivative3_upwind_vertical(field: np.ndarray, velocity: np.ndarray, spacing: float) -> np.ndarray:
    """Matches the M4 no-wrap vertical upwind derivative in NumPy."""

    nz = int(field.shape[0])
    if nz == 1:
        return np.zeros_like(field)
    if nz == 2:
        grad = (field[1:2, :, :] - field[0:1, :, :]) / spacing
        return np.concatenate((grad, grad), axis=0)
    if nz == 3:
        lower_first = (field[1:2, :, :] - field[0:1, :, :]) / spacing
        lower_second = (3.0 * field[2:3, :, :] - 4.0 * field[1:2, :, :] + field[0:1, :, :]) / (
            2.0 * spacing
        )
        backward = np.concatenate((lower_first, lower_first, lower_second), axis=0)
        upper_second = (-3.0 * field[0:1, :, :] + 4.0 * field[1:2, :, :] - field[2:3, :, :]) / (
            2.0 * spacing
        )
        upper_first = (field[2:3, :, :] - field[1:2, :, :]) / spacing
        forward = np.concatenate((upper_second, upper_first, upper_first), axis=0)
        return np.where(velocity >= 0.0, backward, forward)

    lower_first = (field[1:2, :, :] - field[0:1, :, :]) / spacing
    lower_second = (3.0 * field[2:3, :, :] - 4.0 * field[1:2, :, :] + field[0:1, :, :]) / (2.0 * spacing)
    backward_core = (
        11.0 * field[3:, :, :]
        - 18.0 * field[2:-1, :, :]
        + 9.0 * field[1:-2, :, :]
        - 2.0 * field[:-3, :, :]
    ) / (6.0 * spacing)
    backward = np.concatenate((lower_first, lower_first, lower_second, backward_core), axis=0)
    forward_core = (
        -11.0 * field[:-3, :, :]
        + 18.0 * field[1:-2, :, :]
        - 9.0 * field[2:-1, :, :]
        + 2.0 * field[3:, :, :]
    ) / (6.0 * spacing)
    upper_second = (-3.0 * field[-3:-2, :, :] + 4.0 * field[-2:-1, :, :] - field[-1:, :, :]) / (
        2.0 * spacing
    )
    upper_first = (field[-1:, :, :] - field[-2:-1, :, :]) / spacing
    forward = np.concatenate((forward_core, upper_second, upper_first, upper_first), axis=0)
    return np.where(velocity >= 0.0, backward, forward)


def _sha256(path: Path) -> str:
    """Computes the sibling fixture checksum without editing M1 generator code."""

    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upwind5_payload(seed: int) -> dict[str, np.ndarray]:
    """Builds the sibling oracle arrays for the dycore's actual advection scheme."""

    base = generate_stencil_fixture(seed).arrays
    phi = base["phi_initial"].astype(np.float64)
    u_face = base["u_face"]
    v_face = base["v_face"]
    w_face = base["w_face"]
    u_mass = 0.5 * (u_face[:, :, :-1].astype(np.float64) + u_face[:, :, 1:].astype(np.float64))
    v_mass = 0.5 * (v_face[:, :-1, :].astype(np.float64) + v_face[:, 1:, :].astype(np.float64))
    w_mass = 0.5 * (w_face[:-1, :, :].astype(np.float64) + w_face[1:, :, :].astype(np.float64))
    tendency = -(
        u_mass * _derivative5_upwind(phi, u_mass, 900.0, axis=2)
        + v_mass * _derivative5_upwind(phi, v_mass, 900.0, axis=1)
        + w_mass * _derivative3_upwind_vertical(phi, w_mass, 120.0)
    )
    return {
        "phi_initial": phi,
        "u_face": u_face,
        "v_face": v_face,
        "w_face": w_face,
        "phi_next_upwind5": (phi + 3.0 * tendency).astype(np.float64),
    }


def _upwind5_manifest(sample_path: Path, seed: int) -> dict[str, object]:
    """Creates a schema-compatible manifest for the dycore-specific sibling fixture."""

    arrays = _upwind5_payload(seed)
    rel_sample = sample_path.as_posix()
    variables = [
        ("phi_initial", "K", "mass", "float64", 1.0e-10, 0.0, "fp64 analytic input reused from M1 stencil"),
        ("u_face", "m s-1", "u", "float32", 1.0e-6, 1.0e-7, "stored fp32 face velocity reused from M1 stencil"),
        ("v_face", "m s-1", "v", "float32", 1.0e-6, 1.0e-7, "stored fp32 face velocity reused from M1 stencil"),
        ("w_face", "m s-1", "w", "float32", 1.0e-7, 1.0e-7, "stored fp32 face velocity reused from M1 stencil"),
        (
            "phi_next_upwind5",
            "K",
            "mass",
            "float64",
            1.0e-10,
            1.0e-12,
            "fp64 NumPy 5H/3V upwind no-wrap vertical reference",
        ),
    ]
    return {
        "fixture_id": UPWIND5_ID,
        "source": "analytic",
        "source_commit": f"generate_analytic_fixtures.py upwind5 seed {seed}",
        "wrf_version": None,
        "scenario": "32x16x8 periodic-horizontal rigid-vertical staggered-grid 5H/3V upwind single timestep",
        "created_utc": CREATED_UTC,
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": f"python scripts/generate_analytic_fixtures.py --seed {seed} --out fixtures/samples/",
        "external_uri": None,
        "sample_slice_path": rel_sample,
        "git_commit": "worker/gpt/m4-dycore-rk3-advection-acoustic",
        "license_notes": "Analytic synthetic fixture generated in-repository; no external data license.",
        "variables": [
            {
                "name": name,
                "units": units,
                "shape": [int(dim) for dim in arrays[name].shape],
                "staggering": staggering,
                "dtype": dtype,
                "tolerance_abs": tolerance_abs,
                "tolerance_rel": tolerance_rel,
                "tolerance_rationale": rationale,
                "tier_overrides": None,
            }
            for name, units, staggering, dtype, tolerance_abs, tolerance_rel, rationale in variables
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


def generate_upwind5_sibling(seed: int, sample_dir: Path, manifest_dir: Path) -> list[Path]:
    """Writes the M4 sibling fixture without changing the frozen M1 generator."""

    sample_path = sample_dir / f"{UPWIND5_ID}.npz"
    write_deterministic_npz(sample_path, _upwind5_payload(seed))
    manifest_path = manifest_dir / f"{UPWIND5_ID}.yaml"
    write_manifest(manifest_path, _upwind5_manifest(sample_path, seed))
    return [sample_path, manifest_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic analytic M1 fixtures.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic generator seed.")
    parser.add_argument("--out", default="fixtures/samples/", help="Output directory for sample .npz files.")
    parser.add_argument(
        "--manifest-out",
        default="fixtures/manifests/",
        help="Output directory for fixture manifest YAML files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_dir = Path(args.out)
    manifest_dir = Path(args.manifest_out)
    generated = generate_all(args.seed, sample_dir, manifest_dir)
    generated.extend(generate_upwind5_sibling(args.seed, sample_dir, manifest_dir))
    for path in generated:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
