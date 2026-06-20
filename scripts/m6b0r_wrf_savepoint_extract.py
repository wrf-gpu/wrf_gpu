#!/usr/bin/env python
"""Emit M6B0-R HDF5 savepoints from the real Canary d02 WRF state."""

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

import numpy as np
from netCDF4 import Dataset

from gpuwrf.validation.savepoint_io import write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata


SPRINT = ROOT / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission"
SOURCE_WRFOUT = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/"
    "wrfout_d02_2026-05-22_00:00:00"
)
WRF_COMMIT = "115e5756f98ee2370d62b6709baac6417d8f7338"
COEF_FIELDS = ("a", "alpha", "gamma")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _center_slice(size: int, width: int) -> slice:
    if width >= size:
        return slice(0, size)
    start = max((size - width) // 2, 0)
    return slice(start, start + width)


def _golden_slice(hgt: np.ndarray) -> tuple[slice, slice, str]:
    width_y, width_x = 40, 64
    best: tuple[float, int, int] | None = None
    for y0 in range(0, hgt.shape[0] - width_y + 1, 2):
        for x0 in range(0, hgt.shape[1] - width_x + 1, 4):
            tile = hgt[y0 : y0 + width_y, x0 : x0 + width_x]
            score = float(np.nanmean(np.abs(tile)) + np.nanstd(tile))
            if best is None or score < best[0]:
                best = (score, y0, x0)
    if best is None:
        return slice(0, min(width_y, hgt.shape[0])), slice(0, min(width_x, hgt.shape[1])), "golden-fallback"
    _, y0, x0 = best
    run_id = f"m6b0r-golden-canary-d02-20260522T000000Z-y{y0:02d}x{x0:03d}-64x40x44"
    return slice(y0, y0 + width_y), slice(x0, x0 + width_x), run_id


def _slice_for_tier(ds: Dataset, tier: str) -> tuple[slice, slice, str]:
    ny = len(ds.dimensions["south_north"])
    nx = len(ds.dimensions["west_east"])
    if tier == "column":
        return _center_slice(ny, 1), _center_slice(nx, 1), "m6b0r-column-canary-d02-20260522T000000Z"
    if tier == "patch16":
        return _center_slice(ny, 16), _center_slice(nx, 16), "m6b0r-patch16-canary-d02-20260522T000000Z"
    hgt = np.asarray(ds.variables["HGT"][0], dtype=np.float64)
    return _golden_slice(hgt)


def _resolve_top_lid(ds: Dataset) -> bool:
    """Resolves WRF ``top_lid`` namelist flag from the wrfout/namelist.

    Order of precedence: explicit wrfout attribute, sibling ``namelist.output``
    (post-run canonicalised value), sibling ``namelist.input`` (pre-run user
    value), then the WRF default ``.false.``. Matches WRF ``module_small_step_em.F``
    line 619-620 ``lid_flag=1; IF(top_lid)lid_flag=0`` semantics.
    """

    raw = getattr(ds, "TOP_LID", None)
    if raw is not None:
        if isinstance(raw, str):
            return raw.strip().upper() in {"T", ".T.", ".TRUE.", "TRUE", "1"}
        return bool(raw)
    run_dir = Path(SOURCE_WRFOUT).parent
    for namelist_name in ("namelist.output", "namelist.input"):
        namelist_path = run_dir / namelist_name
        if not namelist_path.exists():
            continue
        try:
            text = namelist_path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if "top_lid" not in stripped.lower():
                continue
            _, _, rhs = stripped.partition("=")
            rhs = rhs.split("!", 1)[0].split(",", 1)[0].strip()
            if "*" in rhs:
                _, _, rhs = rhs.partition("*")
                rhs = rhs.strip()
            token = rhs.strip(".").upper()
            if token in {"T", "TRUE"}:
                return True
            if token in {"F", "FALSE"}:
                return False
    return False


def _load_state(tier: str) -> dict[str, object]:
    with Dataset(SOURCE_WRFOUT) as ds:
        ys, xs, run_id = _slice_for_tier(ds, tier)
        theta = np.asarray(ds.variables["T"][0, :, ys, xs], dtype=np.float64) + 300.0
        ph = np.asarray(ds.variables["PH"][0, :, ys, xs], dtype=np.float64)
        phb = np.asarray(ds.variables["PHB"][0, :, ys, xs], dtype=np.float64)
        height = (ph + phb) / 9.80665
        dz_m = np.maximum(np.abs(np.diff(height, axis=0)), 1.0)
        mut = np.asarray(ds.variables["MU"][0, ys, xs] + ds.variables["MUB"][0, ys, xs], dtype=np.float64)
        hgt = np.asarray(ds.variables["HGT"][0, ys, xs], dtype=np.float64)
        mapfac = np.asarray(ds.variables["MAPFAC_M"][0, ys, xs], dtype=np.float64)
        attrs = {
            "dt": float(getattr(ds, "DT", 6.0)),
            "epssm": float(getattr(ds, "EPSSM", 0.1) or 0.1),
            "top_lid": _resolve_top_lid(ds),
            "dims": {name: int(len(dim)) for name, dim in ds.dimensions.items()},
            "slice_y": [int(ys.start), int(ys.stop)],
            "slice_x": [int(xs.start), int(xs.stop)],
            "terrain_min_m": float(np.nanmin(hgt)),
            "terrain_max_m": float(np.nanmax(hgt)),
            "mapfac_min": float(np.nanmin(mapfac)),
            "mapfac_max": float(np.nanmax(mapfac)),
        }
        c1h = np.asarray(ds.variables["C1H"][0], dtype=np.float64)
        c2h = np.asarray(ds.variables["C2H"][0], dtype=np.float64)
        c1f = np.asarray(ds.variables["C1F"][0], dtype=np.float64)
        c2f = np.asarray(ds.variables["C2F"][0], dtype=np.float64)
        rdn = np.asarray(ds.variables["RDN"][0], dtype=np.float64)
        rdnw = np.asarray(ds.variables["RDNW"][0], dtype=np.float64)
    return {
        "theta": theta,
        "dz_m": dz_m,
        "mut": mut,
        "c1h": c1h,
        "c2h": c2h,
        "c1f": c1f,
        "c2f": c2f,
        "rdn": rdn,
        "rdnw": rdnw,
        "attrs": attrs,
        "run_id": run_id,
    }


def _wrf_calc_coef_w(
    state: dict[str, object],
    *,
    dts: float,
    epssm: float,
    g: float = 9.80665,
    top_lid: bool | None = None,
) -> dict[str, np.ndarray]:
    """Python translation of WRF ``calc_coef_w`` (module_small_step_em.F:570-652).

    Fortran-to-Python index mapping (``kde = nz + 1`` in 1-based Fortran,
    so ``F(kde-1) -> P[nz-1]``, ``F(kde) -> P[nz]``):
      * Top ``a`` row (line 626): ``c1f(kde-1)`` and ``c1h(kde-1)`` ->
        ``c1f[nz-1]``, ``c1h[nz-1]``.
      * Top ``b`` row (line 646): ``c1h(kde-1)`` and ``c1f(kde)`` ->
        ``c1h[nz-1]``, ``c1f[nz]``.
      * ``lid_flag`` is ``0`` when the namelist sets ``top_lid=.true.``
        (line 619-620), otherwise ``1``.
    """

    mut = np.asarray(state["mut"], dtype=np.float64)
    c1h = np.asarray(state["c1h"], dtype=np.float64)
    c2h = np.asarray(state["c2h"], dtype=np.float64)
    c1f = np.asarray(state["c1f"], dtype=np.float64)
    c2f = np.asarray(state["c2f"], dtype=np.float64)
    rdn = np.asarray(state["rdn"], dtype=np.float64)
    rdnw = np.asarray(state["rdnw"], dtype=np.float64)
    nz = int(np.asarray(state["theta"]).shape[0])
    ny, nx = mut.shape
    cqw = np.ones((nz + 1, ny, nx), dtype=np.float64)
    c2a = np.ones((nz, ny, nx), dtype=np.float64)
    a = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    alpha = np.ones_like(a)
    gamma = np.zeros_like(a)
    cof = (0.5 * dts * g * (1.0 + epssm)) ** 2

    # WRF lines 619-620: ``lid_flag = 1; IF(top_lid) lid_flag = 0``.
    if top_lid is None:
        top_lid = bool(state["attrs"]["top_lid"])  # type: ignore[index]
    lid_flag = 0.0 if bool(top_lid) else 1.0

    a[1, :, :] = 0.0
    k_top = nz - 1
    # WRF line 626 (k = kde-1): top ``a`` denominator uses c1f(kde-1) / c1h(kde-1).
    denom_top_a = (c1h[k_top] * mut + c2h[k_top]) * (c1f[k_top] * mut + c2f[k_top])
    # WRF line 646 (k = kde):   top ``b`` denominator uses c1h(kde-1) and c1f(kde).
    denom_top_b = (c1h[k_top] * mut + c2h[k_top]) * (c1f[nz] * mut + c2f[nz])
    a[nz, :, :] = -2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1] * lid_flag / denom_top_a
    gamma[0, :, :] = 0.0

    for kk in range(2, nz):
        k = kk - 1
        denom = (c1h[k] * mut + c2h[k]) * (c1f[k] * mut + c2f[k])
        a[kk, :, :] = -cqw[kk] * cof * rdn[kk] * rdnw[kk - 1] * c2a[kk - 1] / denom

    for k in range(1, nz):
        denom1 = (c1h[k] * mut + c2h[k]) * (c1f[k] * mut + c2f[k])
        denom0 = (c1h[k - 1] * mut + c2h[k - 1]) * (c1f[k] * mut + c2f[k])
        denomp = (c1h[k] * mut + c2h[k]) * (c1f[k + 1] * mut + c2f[k + 1])
        b = 1.0 + cqw[k] * cof * rdn[k] * (rdnw[k] * c2a[k] / denom1 + rdnw[k - 1] * c2a[k - 1] / denom0)
        c = -cqw[k] * cof * rdn[k] * rdnw[k] * c2a[k] / denomp
        alpha[k, :, :] = 1.0 / (b - a[k] * gamma[k - 1])
        gamma[k, :, :] = c * alpha[k]

    b_top = 1.0 + 2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1] / denom_top_b
    alpha[nz, :, :] = 1.0 / (b_top - a[nz] * gamma[nz - 1])
    gamma[nz, :, :] = 0.0
    return {"a": a, "alpha": alpha, "gamma": gamma}


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    out: dict[str, VariableMetadata] = {}
    for name, array in arrays.items():
        units = "K" if name == "theta" else "m" if name == "dz_m" else "Pa" if name == "mut" else "dimensionless"
        stagger = "w" if name in COEF_FIELDS else "mass"
        out[name] = VariableMetadata(
            name=name,
            dtype=str(np.asarray(array).dtype),
            shape=tuple(int(dim) for dim in np.asarray(array).shape),
            stagger=stagger,
            units=units,
            provenance="WRF dyn_em/module_small_step_em.F calc_coef_w lines 570-652",
            role=roles.get(name, "input"),
        )
    return out


def _savepoint(
    *,
    tier: str,
    boundary: str,
    step: int,
    state: dict[str, object],
    arrays: dict[str, np.ndarray],
    roles: dict[str, str],
) -> Savepoint:
    attrs = dict(state["attrs"])  # type: ignore[arg-type]
    metadata = SavepointMetadata(
        run_id=f"{state['run_id']}-step{step:03d}-{boundary}",
        wrf_version="WRF-Gen2-artifact",
        wrf_commit=WRF_COMMIT,
        namelist_hash=hashlib.sha256(json.dumps(attrs, sort_keys=True).encode()).hexdigest(),
        source_path=str(SOURCE_WRFOUT),
        domain_index=2,
        tier=tier,
        operator="calc_coef_w",
        boundary=boundary,
        dt_seconds=float(attrs["dt"]),
        rk_stage_index=1,
        acoustic_substep_index=step,
        map_factors={"MAPFAC_M": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
        vertical_grid={"kind": "wrf-hybrid-eta", "nz": int(np.asarray(state["theta"]).shape[0])},
        variables=_var_meta(arrays, roles),
        created_utc=datetime.now(timezone.utc).isoformat(),
        notes=f"Sanitizer-off CPU-path M6B0-R extraction; attrs={attrs}",
    )
    return Savepoint(metadata=metadata, arrays=arrays)


def emit_tier(tier: str, steps: int, output: Path) -> dict[str, object]:
    state = _load_state(tier)
    output.mkdir(parents=True, exist_ok=True)
    requested_steps = [step for step in (1, 2, 5, 10) if step <= steps]
    files: list[Path] = []
    attrs = dict(state["attrs"])  # type: ignore[arg-type]
    for step in requested_steps:
        pre_arrays = {
            "theta": np.asarray(state["theta"], dtype=np.float64),
            "dz_m": np.asarray(state["dz_m"], dtype=np.float64),
            "mut": np.asarray(state["mut"], dtype=np.float64),
        }
        pre = _savepoint(tier=tier, boundary="calc_coef_w_pre", step=step, state=state, arrays=pre_arrays, roles={})
        pre_path = output / f"calc_coef_w_pre_step{step:03d}.h5"
        write_savepoint(pre_path, pre)
        files.append(pre_path)

        coeffs = _wrf_calc_coef_w(state, dts=float(attrs["dt"]), epssm=float(attrs["epssm"]))
        post_arrays = {**pre_arrays, **coeffs}
        roles = {name: "expected" for name in COEF_FIELDS}
        post = _savepoint(tier=tier, boundary="calc_coef_w_post", step=step, state=state, arrays=post_arrays, roles=roles)
        post_path = output / f"calc_coef_w_post_step{step:03d}.h5"
        write_savepoint(post_path, post)
        files.append(post_path)

    total_bytes = sum(path.stat().st_size for path in files)
    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "run_id": state["run_id"],
        "steps": requested_steps,
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": total_bytes,
        "attrs": attrs,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _print_summary(manifest: dict[str, object]) -> None:
    print(f"tier={manifest['tier']}")
    print(f"run_id={manifest['run_id']}")
    print(f"source_path={manifest['source_path']}")
    print(f"savepoint_count={len(manifest['files'])}")
    print(f"total_bytes={manifest['total_bytes']}")
    for path in manifest["files"]:  # type: ignore[index]
        p = Path(str(path))
        print(f"{p} {p.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or SPRINT / "savepoints" / args.tier
    manifest = emit_tier(args.tier, args.steps, output)
    _print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
