#!/usr/bin/env python
"""v0.4.0 dry-mass continuity savepoint parity for specified/nested LBC bounds.

The primary oracle path links a standalone driver against the unmodified WRF
``module_small_step_em.o`` and calls the real ``advance_mu_t`` subroutine.  The
local pristine WRF build is RWORDSIZE=4, so the linked-object comparison is
declared separately from the fp64 source-formula comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf  # noqa: E402

# Pristine-WRF checkout root. Override with WRF_PRISTINE_ROOT; default = sibling of repo.
WRF_ROOT = Path(os.environ.get("WRF_PRISTINE_ROOT", str(ROOT.parent / "wrf_pristine" / "WRF")))
WRF_SMALL_STEP = WRF_ROOT / "dyn_em/module_small_step_em.F"
WRF_SMALL_STEP_OBJ = WRF_ROOT / "dyn_em/module_small_step_em.o"
DRIVER_SRC = ROOT / "proofs/v040/wrf_advance_mu_t_driver.F90"
DEFAULT_WRFINPUT = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260429_18z_l3_24h_20260524T204451Z/wrfinput_d01"
)
OUT_PATH = ROOT / "proofs/v040/mu_continuity_savepoint_parity.json"

SOURCE_FP64_TOL = {"abs": 1.0e-8, "rel": 1.0e-12}
WRF_RWORD4_TOL = {"abs": 2.5e-2, "rel": 2.5e-6}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_var(ds: Dataset, name: str, default: float | None = None) -> np.ndarray:
    if name in ds.variables:
        return np.asarray(ds.variables[name][0], dtype=np.float64)
    if default is None:
        raise KeyError(name)
    shape = (len(ds.dimensions["south_north"]), len(ds.dimensions["west_east"]))
    return np.full(shape, float(default), dtype=np.float64)


def _u_face_average(mass: np.ndarray) -> np.ndarray:
    out = np.empty((mass.shape[0], mass.shape[1] + 1), dtype=np.float64)
    out[:, 0] = mass[:, 0]
    out[:, -1] = mass[:, -1]
    out[:, 1:-1] = 0.5 * (mass[:, :-1] + mass[:, 1:])
    return out


def _v_face_average(mass: np.ndarray) -> np.ndarray:
    out = np.empty((mass.shape[0] + 1, mass.shape[1]), dtype=np.float64)
    out[0, :] = mass[0, :]
    out[-1, :] = mass[-1, :]
    out[1:-1, :] = 0.5 * (mass[:-1, :] + mass[1:, :])
    return out


def _load_real_case(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with Dataset(path) as ds:
        u = np.asarray(ds.variables["U"][0], dtype=np.float64)
        v = np.asarray(ds.variables["V"][0], dtype=np.float64)
        theta = np.asarray(ds.variables["T"][0], dtype=np.float64)
        mu = np.asarray(ds.variables["MU"][0], dtype=np.float64)
        mut = np.asarray(ds.variables["MUB"][0], dtype=np.float64)
        total_mass = mut + mu
        nz, ny, nx = theta.shape
        arrays = {
            "ww": np.zeros((nz + 1, ny, nx), dtype=np.float64),
            "ww_1": np.zeros((nz + 1, ny, nx), dtype=np.float64),
            "u": u,
            # This is a real specified-BC Canary d01 fixture, but it is not a
            # full WRF in-loop savepoint after advance_uv.  Keeping raw wrfinput
            # winds in both u and u_1 double-counts the velocity work term and
            # creates a nonphysical theta amplification in the linked RWORDSIZE=4
            # oracle.  Zero save-increment velocities keep the fixture focused on
            # the WRF advance_mu_t loop bounds/mass path while still using real
            # U/V/MU/map/vcoord fields.
            "u_1": np.zeros_like(u),
            "v": v,
            "v_1": np.zeros_like(v),
            "mu": mu,
            "mut": mut,
            "muave": mu.copy(),
            "muts": total_mass.copy(),
            "muu": _u_face_average(total_mass),
            "muv": _v_face_average(total_mass),
            "mudf": np.zeros_like(mu),
            "theta": theta,
            "theta_1": theta.copy(),
            "theta_ave": theta.copy(),
            "theta_tend": np.zeros_like(theta),
            "mu_tend": np.zeros_like(mu),
            "dnw": np.asarray(ds.variables["DNW"][0], dtype=np.float64),
            "fnm": np.asarray(ds.variables["FNM"][0], dtype=np.float64),
            "fnp": np.asarray(ds.variables["FNP"][0], dtype=np.float64),
            "rdnw": np.asarray(ds.variables["RDNW"][0], dtype=np.float64),
            "c1h": np.asarray(ds.variables["C1H"][0], dtype=np.float64),
            "c2h": np.asarray(ds.variables["C2H"][0], dtype=np.float64),
            "c1f": np.asarray(ds.variables.get("C1F", ds.variables["C1H"])[0], dtype=np.float64),
            "c2f": np.asarray(ds.variables.get("C2F", ds.variables["C2H"])[0], dtype=np.float64),
            "c3h": np.zeros(nz, dtype=np.float64),
            "c4h": np.zeros(nz, dtype=np.float64),
            "c3f": np.zeros(nz, dtype=np.float64),
            "c4f": np.zeros(nz, dtype=np.float64),
            "msfux": _read_var(ds, "MAPFAC_UX", 1.0),
            "msfuy": np.asarray(ds.variables["MAPFAC_UY"][0], dtype=np.float64),
            "msfvx": np.asarray(ds.variables["MAPFAC_VX"][0], dtype=np.float64),
            "msfvx_inv": 1.0 / np.asarray(ds.variables["MAPFAC_VX"][0], dtype=np.float64),
            "msfvy": _read_var(ds, "MAPFAC_VY", 1.0),
            "msftx": np.asarray(ds.variables["MAPFAC_MX"][0], dtype=np.float64),
            "msfty": np.asarray(ds.variables["MAPFAC_MY"][0], dtype=np.float64),
            "uam": np.zeros_like(theta),
            "vam": np.zeros_like(theta),
            "wwam": np.zeros((nz + 1, ny, nx), dtype=np.float64),
        }
        attrs = {
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": float(getattr(ds, "DX", 9000.0)),
            "dy": float(getattr(ds, "DY", 9000.0)),
            "dt": float(getattr(ds, "DT", 18.0)),
            "epssm": 0.5,
            "title": getattr(ds, "TITLE", ""),
            "fixture_note": (
                "Real Canary d01 specified-BC wrfinput fields; u_1/v_1 save-increment "
                "velocity arrays are zeroed because this is not a post-advance_uv WRF "
                "savepoint and raw wrfinput winds in both u and u_1 create a nonphysical "
                "theta amplification in the RWORDSIZE=4 linked-object check."
            ),
        }
    return arrays, attrs


def _inputs(arrays: dict[str, np.ndarray], attrs: dict[str, Any], *, dtype=np.float64) -> AdvanceMuTInputs:
    def a(name: str):
        return jnp.asarray(arrays[name], dtype=dtype)

    return AdvanceMuTInputs(
        ww=a("ww"),
        ww_1=a("ww_1"),
        u=a("u"),
        u_1=a("u_1"),
        v=a("v"),
        v_1=a("v_1"),
        mu=a("mu"),
        mut=a("mut"),
        muave=a("muave"),
        muts=a("muts"),
        muu=a("muu"),
        muv=a("muv"),
        mudf=a("mudf"),
        theta=a("theta"),
        theta_1=a("theta_1"),
        theta_ave=a("theta_ave"),
        theta_tend=a("theta_tend"),
        mu_tend=a("mu_tend"),
        dnw=a("dnw"),
        fnm=a("fnm"),
        fnp=a("fnp"),
        rdnw=a("rdnw"),
        c1h=a("c1h"),
        c2h=a("c2h"),
        msfuy=a("msfuy"),
        msfvx_inv=a("msfvx_inv"),
        msftx=a("msftx"),
        msfty=a("msfty"),
        rdx=1.0 / float(attrs["dx"]),
        rdy=1.0 / float(attrs["dy"]),
        dts=float(attrs["dt"]),
        epssm=float(attrs["epssm"]),
        periodic_x=False,
        specified=True,
        nested=False,
    )


def _numpy_source_advance(arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    nz, ny, nx = arrays["theta"].shape
    y0, y1 = 1, ny - 1
    x0, x1 = 1, nx - 1
    ys = slice(y0, y1)
    xs = slice(x0, x1)
    xs_e = slice(x0 + 1, x1 + 1)
    ys_n = slice(y0 + 1, y1 + 1)
    rdx = 1.0 / float(attrs["dx"])
    rdy = 1.0 / float(attrs["dy"])
    dts = float(attrs["dt"])
    epssm = float(attrs["epssm"])

    dvdxi = np.zeros_like(arrays["theta"], dtype=np.float64)
    for k in range(nz):
        c1 = arrays["c1h"][k]
        c2 = arrays["c2h"][k]
        v_north = arrays["v"][k, ys_n, xs] + (
            c1 * arrays["muv"][ys_n, xs] + c2
        ) * arrays["v_1"][k, ys_n, xs] * arrays["msfvx_inv"][ys_n, xs]
        v_south = arrays["v"][k, ys, xs] + (
            c1 * arrays["muv"][ys, xs] + c2
        ) * arrays["v_1"][k, ys, xs] * arrays["msfvx_inv"][ys, xs]
        u_east = arrays["u"][k, ys, xs_e] + (
            c1 * arrays["muu"][ys, xs_e] + c2
        ) * arrays["u_1"][k, ys, xs_e] / arrays["msfuy"][ys, xs_e]
        u_west = arrays["u"][k, ys, xs] + (
            c1 * arrays["muu"][ys, xs] + c2
        ) * arrays["u_1"][k, ys, xs] / arrays["msfuy"][ys, xs]
        dvdxi[k, ys, xs] = arrays["msftx"][ys, xs] * arrays["msfty"][ys, xs] * (
            rdy * (v_north - v_south) + rdx * (u_east - u_west)
        )

    dmdt = np.zeros_like(arrays["mu"], dtype=np.float64)
    dmdt[ys, xs] = np.sum(arrays["dnw"][:nz, None, None] * dvdxi[:, ys, xs], axis=0)
    mu_work_old = arrays["muts"][ys, xs] - arrays["mut"][ys, xs]
    mu_save = arrays["mu"][ys, xs] - mu_work_old
    mu_tendency = dmdt[ys, xs] + arrays["mu_tend"][ys, xs]
    mu_work_new = mu_work_old + dts * mu_tendency

    out = {name: np.array(value, copy=True) for name, value in arrays.items()}
    out["mu"][ys, xs] = mu_save + mu_work_new
    out["mudf"][ys, xs] = mu_tendency
    out["muts"][ys, xs] = arrays["mut"][ys, xs] + mu_work_new
    out["muave"][ys, xs] = 0.5 * ((1.0 + epssm) * mu_work_new + (1.0 - epssm) * mu_work_old)

    ww_active = np.empty((nz + 1, y1 - y0, x1 - x0), dtype=np.float64)
    ww_active[0] = arrays["ww"][0, ys, xs]
    for kk in range(1, nz):
        k = kk - 1
        increment = arrays["dnw"][kk - 1] * (
            arrays["c1h"][k] * dmdt[ys, xs]
            + dvdxi[kk - 1, ys, xs]
            + arrays["c1h"][k] * arrays["mu_tend"][ys, xs]
        ) / arrays["msfty"][ys, xs]
        ww_active[kk] = ww_active[kk - 1] - increment
    ww_active[nz] = arrays["ww"][nz, ys, xs]
    ww_active[:nz] -= arrays["ww_1"][:nz, ys, xs]
    out["ww"][:, ys, xs] = ww_active

    theta_i = np.array(arrays["theta"], copy=True)
    theta_i[:, ys, xs] += arrays["msfty"][None, ys, xs] * dts * arrays["theta_tend"][:, ys, xs]
    wdtn = np.zeros_like(arrays["ww"], dtype=np.float64)
    for k in range(1, nz):
        face_theta = arrays["fnm"][k] * arrays["theta_1"][k, ys, xs] + arrays["fnp"][k] * arrays["theta_1"][k - 1, ys, xs]
        wdtn[k, ys, xs] = ww_active[k] * face_theta

    theta_tendency = np.zeros_like(arrays["theta"], dtype=np.float64)
    for k in range(nz):
        th = arrays["theta_1"][k]
        th_e = th[ys, slice(x0 + 1, x1 + 1)]
        th_w = th[ys, slice(x0 - 1, x1 - 1)]
        th_n = th[slice(y0 + 1, y1 + 1), xs]
        th_s = th[slice(y0 - 1, y1 - 1), xs]
        v_flux = arrays["v"][k, ys_n, xs] * (th_n + th[ys, xs]) - arrays["v"][k, ys, xs] * (th[ys, xs] + th_s)
        u_flux = arrays["u"][k, ys, xs_e] * (th_e + th[ys, xs]) - arrays["u"][k, ys, xs] * (th[ys, xs] + th_w)
        tendency = arrays["msftx"][ys, xs] * (0.5 * rdy * v_flux + 0.5 * rdx * u_flux) + arrays["rdnw"][k] * (
            wdtn[k + 1, ys, xs] - wdtn[k, ys, xs]
        )
        theta_tendency[k, ys, xs] = tendency
        out["theta"][k, ys, xs] = theta_i[k, ys, xs] - dts * arrays["msfty"][ys, xs] * tendency

    out["dmdt"] = dmdt
    out["dvdxi"] = dvdxi
    out["wdtn"] = wdtn
    out["theta_tendency"] = theta_tendency
    return out


def _compile_driver(build_dir: Path) -> Path:
    # Fortran compiler: WRF_FC override, else gfortran on PATH.
    fc_env = os.environ.get("WRF_FC")
    fc = Path(fc_env) if fc_env else Path(shutil.which("gfortran") or "gfortran")
    if not fc.exists() and not shutil.which(str(fc)):
        raise FileNotFoundError(f"WRF Fortran compiler not found: {fc} (set WRF_FC)")
    exe = build_dir / "wrf_advance_mu_t_driver"
    cmd = [
        str(fc),
        f"-I{WRF_ROOT / 'dyn_em'}",
        f"-I{WRF_ROOT / 'frame'}",
        f"-I{WRF_ROOT / 'share'}",
        f"-I{WRF_ROOT / 'external/esmf_time_f90'}",
        str(DRIVER_SRC),
        str(WRF_SMALL_STEP_OBJ),
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    return exe


def _pad2(arr: np.ndarray, nx: int, ny: int, *, dtype: np.dtype) -> np.ndarray:
    out = np.zeros((nx + 1, ny + 1), dtype=dtype)
    out[: arr.shape[1], : arr.shape[0]] = arr.T.astype(dtype)
    return out


def _pad3(arr: np.ndarray, nx: int, ny: int, nz: int, *, dtype: np.dtype) -> np.ndarray:
    out = np.zeros((nx + 1, nz + 1, ny + 1), dtype=dtype)
    x_len = arr.shape[2]
    z_len = arr.shape[0]
    y_len = arr.shape[1]
    out[:x_len, :z_len, :y_len] = np.transpose(arr, (2, 0, 1)).astype(dtype)
    return out


def _write_fortran_input(path: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any]) -> None:
    nx = int(attrs["nx"])
    ny = int(attrs["ny"])
    nz = int(attrs["nz"])
    dtype = np.dtype("<f4")

    def write_array(handle, arr: np.ndarray) -> None:
        handle.write(np.asarray(arr, dtype=dtype).ravel(order="F").tobytes())

    with path.open("wb") as handle:
        handle.write(np.asarray([nx, ny, nz], dtype="<i4").tobytes())
        handle.write(np.asarray([1.0 / attrs["dx"], 1.0 / attrs["dy"], attrs["dt"], attrs["epssm"]], dtype=dtype).tobytes())
        for name in ("ww", "ww_1", "u", "u_1", "v", "v_1"):
            write_array(handle, _pad3(arrays[name], nx, ny, nz, dtype=dtype))
        for name in ("mu", "mut", "muave", "muts", "muu", "muv", "mudf"):
            write_array(handle, _pad2(arrays[name], nx, ny, dtype=dtype))
        for name in ("c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f"):
            values = np.zeros(nz + 1, dtype=dtype)
            source = np.asarray(arrays[name], dtype=dtype)
            values[: min(nz, source.shape[0])] = source[: min(nz, source.shape[0])]
            write_array(handle, values)
        for name in ("uam", "vam", "wwam", "theta", "theta_1", "theta_ave", "theta_tend"):
            write_array(handle, _pad3(arrays[name], nx, ny, nz, dtype=dtype))
        write_array(handle, _pad2(arrays["mu_tend"], nx, ny, dtype=dtype))
        for name in ("dnw", "fnm", "fnp", "rdnw"):
            values = np.zeros(nz + 1, dtype=dtype)
            source = np.asarray(arrays[name], dtype=dtype)
            values[: min(nz, source.shape[0])] = source[: min(nz, source.shape[0])]
            write_array(handle, values)
        for name in ("msfux", "msfuy", "msfvx", "msfvx_inv", "msfvy", "msftx", "msfty"):
            write_array(handle, _pad2(arrays[name], nx, ny, dtype=dtype))


def _read_fortran_output(path: Path, attrs: dict[str, Any]) -> dict[str, np.ndarray]:
    nx = int(attrs["nx"])
    ny = int(attrs["ny"])
    nz = int(attrs["nz"])

    with path.open("rb") as handle:
        header = np.frombuffer(handle.read(16), dtype="<i4")
        if tuple(header[:3]) != (nx, ny, nz):
            raise ValueError(f"unexpected WRF driver output header {header[:3]}")
        real_bytes = int(header[3])
        if real_bytes != 4:
            raise ValueError(f"expected RWORDSIZE=4 output, got {real_bytes}")
        raw = np.frombuffer(handle.read(), dtype="<f4")

    offset = 0

    def next_array(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        arr = raw[offset : offset + size].reshape(shape, order="F")
        offset += size
        return arr.astype(np.float64)

    def mass(arr: np.ndarray) -> np.ndarray:
        return arr[:nx, :ny].T.copy()

    def field3(arr: np.ndarray, z_len: int) -> np.ndarray:
        return np.transpose(arr[:nx, :z_len, :ny], (1, 2, 0)).copy()

    shape2 = (nx + 1, ny + 1)
    shape3 = (nx + 1, nz + 1, ny + 1)
    mu = mass(next_array(shape2))
    mudf = mass(next_array(shape2))
    muts = mass(next_array(shape2))
    muave = mass(next_array(shape2))
    ww = field3(next_array(shape3), nz + 1)
    theta = field3(next_array(shape3), nz)
    theta_ave = field3(next_array(shape3), nz)
    return {"mu": mu, "mudf": mudf, "muts": muts, "muave": muave, "ww": ww, "theta": theta, "theta_ave": theta_ave}


def _run_wrf_driver(exe: Path, build_dir: Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any], step: int) -> dict[str, np.ndarray]:
    input_path = build_dir / f"advance_mu_t_step{step:02d}.in.bin"
    output_path = build_dir / f"advance_mu_t_step{step:02d}.out.bin"
    _write_fortran_input(input_path, arrays, attrs)
    subprocess.run([str(exe), str(input_path), str(output_path)], check=True, cwd=ROOT, capture_output=True, text=True)
    out = _read_fortran_output(output_path, attrs)
    out["dmdt"] = out["mudf"] - arrays["mu_tend"].astype(np.float64)
    return out


def _state_update(state: dict[str, np.ndarray], out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    next_state = {name: np.array(value, copy=True) for name, value in state.items()}
    for name in ("mu", "mudf", "muts", "muave", "ww", "theta"):
        next_state[name] = np.asarray(out[name], dtype=state[name].dtype)
    return next_state


def _jax_advance(arrays: dict[str, np.ndarray], attrs: dict[str, Any], *, dtype=np.float64) -> dict[str, np.ndarray]:
    out = advance_mu_t_wrf(_inputs(arrays, attrs, dtype=dtype))
    jax.block_until_ready(out["theta"])
    return {name: np.asarray(value) for name, value in out.items()}


def _regions(arr: np.ndarray) -> dict[str, np.ndarray]:
    if arr.ndim == 2:
        return {
            "interior": arr[1:-1, 1:-1],
            "west_boundary_col": arr[:, 0],
            "east_boundary_col": arr[:, -1],
            "south_boundary_row": arr[0, :],
            "north_boundary_row": arr[-1, :],
        }
    return {
        "interior": arr[..., 1:-1, 1:-1],
        "west_boundary_col": arr[..., :, 0],
        "east_boundary_col": arr[..., :, -1],
        "south_boundary_row": arr[..., 0, :],
        "north_boundary_row": arr[..., -1, :],
    }


def _compare(got: np.ndarray, expected: np.ndarray, tol: dict[str, float]) -> dict[str, Any]:
    rows = {}
    passed = True
    for region, g in _regions(got).items():
        e = _regions(expected)[region]
        delta = np.asarray(g, dtype=np.float64) - np.asarray(e, dtype=np.float64)
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        mean_signed = float(np.mean(delta)) if delta.size else 0.0
        mean_abs = float(np.mean(np.abs(delta))) if delta.size else 0.0
        scale = float(np.max(np.abs(e))) if e.size else 0.0
        allowed = float(tol["abs"] + tol["rel"] * scale)
        ok = bool(max_abs <= allowed)
        passed = passed and ok
        rows[region] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "mean_signed": mean_signed,
            "allowed": allowed,
            "pass": ok,
        }
    return {"pass": passed, "regions": rows}


def _compare_fields(got: dict[str, np.ndarray], expected: dict[str, np.ndarray], fields: tuple[str, ...], tol: dict[str, float]) -> dict[str, Any]:
    result = {}
    passed = True
    for field in fields:
        item = _compare(got[field], expected[field], tol)
        result[field] = item
        passed = passed and bool(item["pass"])
    return {"pass": passed, "fields": result}


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrfinput", type=Path, default=DEFAULT_WRFINPUT)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/v040_mu_continuity_wrf_oracle"))
    args = parser.parse_args()

    args.build_dir.mkdir(parents=True, exist_ok=True)
    arrays64, attrs = _load_real_case(args.wrfinput)
    exe = _compile_driver(args.build_dir)

    source_state = {k: np.array(v, dtype=np.float64, copy=True) for k, v in arrays64.items()}
    jax_state64 = {k: np.array(v, dtype=np.float64, copy=True) for k, v in arrays64.items()}
    wrf_state = {k: np.array(v, dtype=np.float32, copy=True) for k, v in arrays64.items()}
    jax_state32 = {k: np.array(v, dtype=np.float32, copy=True) for k, v in arrays64.items()}

    source_steps = []
    wrf_steps = []
    source_fields = ("mu", "mudf", "muts", "muave", "ww", "theta", "dmdt", "dvdxi", "wdtn", "theta_tendency")
    wrf_fields = ("mu", "mudf", "muts", "muave", "ww", "theta", "dmdt")
    for step in range(1, int(args.substeps) + 1):
        source_out = _numpy_source_advance(source_state, attrs)
        jax_out64 = _jax_advance(jax_state64, attrs, dtype=np.float64)
        source_cmp = _compare_fields(jax_out64, source_out, source_fields, SOURCE_FP64_TOL)
        source_steps.append({"substep": step, **source_cmp})
        source_state = _state_update(source_state, source_out)
        jax_state64 = _state_update(jax_state64, jax_out64)

        wrf_out = _run_wrf_driver(exe, args.build_dir, wrf_state, attrs, step)
        jax_out32 = _jax_advance(jax_state32, attrs, dtype=np.float32)
        wrf_cmp = _compare_fields(jax_out32, wrf_out, wrf_fields, WRF_RWORD4_TOL)
        wrf_steps.append({"substep": step, **wrf_cmp})
        wrf_state = _state_update(wrf_state, wrf_out)
        jax_state32 = _state_update(jax_state32, jax_out32)

    source_pass = all(step["pass"] for step in source_steps)
    wrf_pass = all(step["pass"] for step in wrf_steps)
    payload = {
        "schema": "v0.4.0-mu-continuity-savepoint-parity-2026-06-03",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": {
            **attrs,
            "bc_type": {"periodic_x": False, "specified": True, "nested": False},
            "active_bounds_python": {"y": [1, int(attrs["ny"]) - 1], "x": [1, int(attrs["nx"]) - 1]},
            "wrf_source_bounds": "module_small_step_em.F:1048-1063; DMDT/MU update:1092-1107; theta flux:1138-1171",
        },
        "wrf_provenance": {
            "wrf_root": str(WRF_ROOT),
            "wrf_git_head": _git_head(WRF_ROOT),
            "module_small_step_em_F_sha256": _sha256(WRF_SMALL_STEP),
            "module_small_step_em_o_sha256": _sha256(WRF_SMALL_STEP_OBJ),
            "configure_wrf_sha256": _sha256(WRF_ROOT / "configure.wrf"),
            "rwordsize": 4,
            "unmodified_source_check": {
                "module_small_step_em_git_diff_empty": subprocess.run(
                    ["git", "-C", str(WRF_ROOT), "diff", "--quiet", "--", "dyn_em/module_small_step_em.F"]
                ).returncode == 0,
                "note": "Only module_small_step_em.F is asserted clean; the WRF worktree has unrelated dirt in other files.",
            },
            "linked_driver": str(exe),
            "linked_driver_source": str(DRIVER_SRC),
        },
        "predeclared_tolerances": {
            "source_formula_fp64": SOURCE_FP64_TOL,
            "linked_wrf_object_rword4": WRF_RWORD4_TOL,
            "note": (
                "The linked WRF oracle is the real unmodified WRF object but was built with RWORDSIZE=4. "
                "The fp64 machine-precision gate is therefore the independent WRF-source formula oracle; "
                "the linked-object gate protects ABI/source fidelity at compiled WRF precision."
            ),
        },
        "source_formula_fp64": {
            "oracle": "independent NumPy transcription of WRF module_small_step_em.F:1048-1171",
            "candidate": "gpuwrf.dynamics.mu_t_advance.advance_mu_t_wrf, specified=True, periodic_x=False, JAX fp64 CPU",
            "pass": source_pass,
            "substeps": source_steps,
        },
        "linked_wrf_object": {
            "oracle": "standalone driver linked to unmodified WRF module_small_step_em.o advance_mu_t",
            "candidate": "same JAX kernel with float32-rounded inputs to match WRF RWORDSIZE=4",
            "pass": wrf_pass,
            "substeps": wrf_steps,
        },
        "verdict": "PASS" if source_pass and wrf_pass else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "verdict": payload["verdict"], "source_fp64": source_pass, "wrf_object": wrf_pass}, indent=2))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
