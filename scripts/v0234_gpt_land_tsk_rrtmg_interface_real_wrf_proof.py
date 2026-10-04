#!/usr/bin/env python3
"""Authenticated real-WRF A/B for explicit RRTMG-LW P3D/P8W/T8W inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np

from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column


REPO = Path(__file__).resolve().parent.parent
EXPECTED = {
    "oracle_manifest": "bdd0f0fd715bf06a78bc61fc63600aff1177685082ee3f044b222b8f365eb136",
    "wrfinput_d01": "a28fb63e890163e26c5797c67fbaedc877cc1c7967100edcd9f08cdfec37ffc6",
    "first_rk": "8c666fe88c46b04e297fe7b7289f55ec74fa133287b234a02f10e05cbbd11841",
    "phy_prep": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verify the pinned oracle manifest
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        error = result.stderr.decode() if binary else result.stderr
        raise RuntimeError(f"git {' '.join(args)}: {error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked_script() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    if disk != _git("show", f"HEAD:{relative}", binary=True):
        raise RuntimeError(f"proof script is not HEAD: {relative}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _load_oracle(root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray], dict[str, str]]:
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != EXPECTED["oracle_manifest"]:
        raise RuntimeError("authenticated radiation oracle manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields: dict[tuple[str, str], np.ndarray] = {}
    hashes: dict[str, str] = {}
    for item in manifest["fields"]:
        if item["scheme"] != "rrtmg_lw":
            continue
        path = root / item["file"]
        if _md5(path) != item["md5"]:
            raise RuntimeError(f"oracle MD5 drift: {item['file']}")
        raw = np.fromfile(path, dtype=">f8")
        if raw.nbytes != int(item["bytes"]):
            raise RuntimeError(f"oracle byte count drift: {item['file']}")
        fields[(item["tag"], item["name"])] = np.ascontiguousarray(
            raw.reshape(tuple(item["shape"]))
        )
        hashes[item["file"]] = _sha256(path)
    return manifest, fields, hashes


def _source_gate(first_rk: Path, phy_prep: Path) -> dict[str, Any]:
    if _sha256(first_rk) != EXPECTED["first_rk"]:
        raise RuntimeError("first-RK WRF source drift")
    if _sha256(phy_prep) != EXPECTED["phy_prep"]:
        raise RuntimeError("phy_prep WRF source drift")
    call_text = first_rk.read_text(encoding="utf-8", errors="strict")
    prep_text = phy_prep.read_text(encoding="utf-8", errors="strict")
    required = {
        "radiation_p8w_hydrostatic": (call_text, r"P8W\s*=\s*grid%p_hyd_w"),
        "radiation_p3d_hydrostatic": (call_text, r"P\s*=\s*grid%p_hyd"),
        "radiation_t8w_phy_prep": (call_text, r"T8W\s*=\s*t8w"),
        "radiation_t3d_phy_prep": (call_text, r"T\s*=\s*grid%t_phy"),
        "hydrostatic_top": (prep_text, r"p_hyd_w\(i,kte,j\)\s*=\s*p_top"),
        "hydrostatic_recursion": (
            prep_text,
            r"p_hyd_w\(i,k,j\)\s*=\s*p_hyd_w\(i,k\+1,j\)\s*-\s*\(1\.\+qtot\)\*\(c1\(k\)\*MUT\(i,j\)\+c2\(k\)\)\*dnw\(k\)",
        ),
        "hydrostatic_mass": (
            prep_text,
            r"p_hyd\(i,k,j\)\s*=\s*0\.5\*\(p_hyd_w\(i,k,j\)\+p_hyd_w\(i,k\+1,j\)\)",
        ),
        "t8w_interior": (
            prep_text,
            r"t8w\(i,k,j\)\s*=\s*fzm\(k\)\*t_phy\(i,k,j\)\+fzp\(k\)\*t_phy\(i,k-1,j\)",
        ),
        "t8w_top": (
            prep_text,
            r"t8w\(i,kde,j\)\s*=\s*w1\*t_phy\(i,kde-1,j\)\+w2\*t_phy\(i,kde-2,j\)",
        ),
    }
    matches = {
        name: bool(re.search(pattern, text, flags=re.IGNORECASE))
        for name, (text, pattern) in required.items()
    }
    if not all(matches.values()):
        raise RuntimeError(f"WRF source-expression gate: {matches}")
    return {
        "first_rk": {"path": str(first_rk), "sha256": EXPECTED["first_rk"]},
        "phy_prep": {"path": str(phy_prep), "sha256": EXPECTED["phy_prep"]},
        "required_expression_matches": matches,
    }


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError(f"invalid metric inputs: {left.shape}, {right.shape}")
    delta = left - right
    return {
        "shape": list(left.shape),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "bias": float(np.mean(delta)),
        "bitwise_mismatch_count": int(np.count_nonzero(left != right)),
        "candidate_min": float(np.min(left)),
        "candidate_max": float(np.max(left)),
        "reference_min": float(np.min(right)),
        "reference_max": float(np.max(right)),
    }


def _wrf_interface_inputs(
    wrfinput: Path,
    fields: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if _sha256(wrfinput) != EXPECTED["wrfinput_d01"]:
        raise RuntimeError("source-run wrfinput drift")

    p_oracle_zyx = np.moveaxis(fields[("in", "p")], 1, 0).astype(np.float64)
    t_oracle_zyx = np.moveaxis(fields[("in", "t")], 1, 0).astype(np.float64)
    dz_oracle_zyx = np.moveaxis(fields[("in", "dz8w")], 1, 0).astype(np.float64)
    pi_oracle_zyx = np.moveaxis(fields[("in", "pi3d")], 1, 0).astype(np.float64)

    with Dataset(wrfinput) as dataset:
        def first(name: str) -> np.ndarray:
            return np.asarray(dataset[name][0], dtype=np.float32)

        if (
            len(dataset.dimensions["west_east"]) != 93
            or len(dataset.dimensions["south_north"]) != 59
            or len(dataset.dimensions["bottom_top"]) != 44
        ):
            raise RuntimeError("source-run wrfinput dimensions do not match oracle")
        mut = (first("MU") + first("MUB")).astype(np.float32)
        c1h = first("C1H")
        c2h = first("C2H")
        dnw = first("DNW")
        fnm = first("FNM")
        fnp = first("FNP")
        p_top = np.asarray(first("P_TOP"), dtype=np.float32).reshape(())
        qtot = np.zeros_like(p_oracle_zyx, dtype=np.float32)
        for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
            qtot = (qtot + first(name)).astype(np.float32)
        ph_total = (first("PH") + first("PHB")).astype(np.float32)
        theta = (first("T") + np.float32(300.0)).astype(np.float32)

    nz, ny, nx = p_oracle_zyx.shape
    p8w = np.empty((nz + 1, ny, nx), dtype=np.float32)
    p8w[-1] = p_top
    for k in range(nz - 1, -1, -1):
        mass = (c1h[k] * mut + c2h[k]).astype(np.float32)
        p8w[k] = (
            p8w[k + 1]
            - (np.float32(1.0) + qtot[k]) * mass * dnw[k]
        ).astype(np.float32)
    p_hyd = (np.float32(0.5) * (p8w[:-1] + p8w[1:])).astype(np.float32)

    t3d = (theta * pi_oracle_zyx.astype(np.float32)).astype(np.float32)
    z_at_w = (ph_total / np.float32(9.81)).astype(np.float32)
    dz8w = (z_at_w[1:] - z_at_w[:-1]).astype(np.float32)
    z_mass = (np.float32(0.5) * (z_at_w[:-1] + z_at_w[1:])).astype(np.float32)
    t_source = t_oracle_zyx.astype(np.float32)
    t8w = np.empty((nz + 1, ny, nx), dtype=np.float32)
    t8w[1:-1] = (
        fnm[1:, None, None] * t_source[1:]
        + fnp[1:, None, None] * t_source[:-1]
    ).astype(np.float32)

    w1 = ((z_at_w[0] - z_mass[1]) / (z_mass[0] - z_mass[1])).astype(np.float32)
    w2 = (np.float32(1.0) - w1).astype(np.float32)
    t8w[0] = (w1 * t_source[0] + w2 * t_source[1]).astype(np.float32)
    w1 = ((z_at_w[-1] - z_mass[-2]) / (z_mass[-1] - z_mass[-2])).astype(np.float32)
    w2 = (np.float32(1.0) - w1).astype(np.float32)
    t8w[-1] = (w1 * t_source[-1] + w2 * t_source[-2]).astype(np.float32)

    checks = {
        "p_hyd_mass_vs_authenticated_rrtmg_p3d": _metrics(
            p_hyd.astype(np.float64), p_oracle_zyx
        ),
        "t_phy_vs_authenticated_rrtmg_t3d": _metrics(
            t3d.astype(np.float64), t_oracle_zyx
        ),
        "dz8w_vs_authenticated_rrtmg_dz8w": _metrics(
            dz8w.astype(np.float64), dz_oracle_zyx
        ),
    }
    if any(record["bitwise_mismatch_count"] != 0 for record in checks.values()):
        raise RuntimeError(f"source-run input does not replay authenticated oracle: {checks}")
    return p8w, t8w, {
        "wrfinput": {"path": str(wrfinput), "sha256": EXPECTED["wrfinput_d01"]},
        "authenticated_input_replay": checks,
        "p8w_shape": list(p8w.shape),
        "t8w_shape": list(t8w.shape),
        "p8w_float32_payload_sha256": _array_sha(p8w),
        "t8w_float32_payload_sha256": _array_sha(t8w),
        "p8w_top_bitwise_exact_5000_pa": bool(np.all(p8w[-1] == np.float32(5000.0))),
        "p8w_strictly_decreases_bottom_to_top": bool(np.all(np.diff(p8w, axis=0) < 0.0)),
    }


def run(oracle_dir: Path, wrfinput: Path, wrf_root: Path, output: Path) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean")
    script = _tracked_script()
    manifest, fields, oracle_hashes = _load_oracle(oracle_dir)
    source = _source_gate(
        wrf_root / "dyn_em/module_first_rk_step_part1.F",
        wrf_root / "dyn_em/module_big_step_utilities_em.F",
    )
    p8w_zyx, t8w_zyx, interface_source = _wrf_interface_inputs(wrfinput, fields)

    def col3(name: str) -> jax.Array:
        array = fields[("in", name)]
        nj, nk, ni = array.shape
        return jnp.asarray(np.moveaxis(array, 1, 2).reshape(nj * ni, nk), dtype=jnp.float64)

    def surface(tag: str, name: str) -> jax.Array:
        return jnp.asarray(fields[(tag, name)].reshape(-1), dtype=jnp.float64)

    T = col3("t")
    p = col3("p")
    qv = col3("qv")
    qc = col3("qc")
    qi = col3("qi")
    qs = col3("qs")
    dz = col3("dz8w")
    rho = col3("rho")
    cloud_fraction = col3("cldfra")
    zero = jnp.zeros_like(qv)
    common = dict(
        T=T,
        p=p,
        qv=qv,
        qc=qc,
        qi=qi,
        qs=qs,
        qg=zero,
        cloud_fraction=cloud_fraction,
        surface_temperature=surface("in", "tsk"),
        surface_emissivity=surface("in", "emiss"),
        dz=dz,
        rho=rho,
        top_pressure_pa=5000.0,
    )
    pre_interface = RRTMGLWColumnState(**common)
    p8w_columns = np.moveaxis(p8w_zyx, 0, -1).reshape(-1, p8w_zyx.shape[0])
    t8w_columns = np.moveaxis(t8w_zyx, 0, -1).reshape(-1, t8w_zyx.shape[0])
    wrf_interface = RRTMGLWColumnState(
        **common,
        pressure_interfaces=jnp.asarray(p8w_columns, dtype=jnp.float64),
        temperature_interfaces=jnp.asarray(t8w_columns, dtype=jnp.float64),
    )

    before = solve_rrtmg_lw_column(pre_interface, debug=False)
    after = solve_rrtmg_lw_column(wrf_interface, debug=False)
    jax.block_until_ready((before.surface_down, after.surface_down))

    glw_reference = np.asarray(surface("out", "glw"))
    pi3d = np.asarray(col3("pi3d"))
    rthraten = fields[("out", "rthratenlw")]
    nj, nk, ni = rthraten.shape
    heating_reference = np.moveaxis(rthraten, 1, 2).reshape(nj * ni, nk) * pi3d
    before_glw = _metrics(np.asarray(before.surface_down), glw_reference)
    after_glw = _metrics(np.asarray(after.surface_down), glw_reference)
    before_heating = _metrics(np.asarray(before.heating_rate), heating_reference)
    after_heating = _metrics(np.asarray(after.heating_rate), heating_reference)
    improved = after_glw["rms"] < before_glw["rms"]
    public_compatible = before.flux_down.shape == after.flux_down.shape == (5487, 46)

    record = {
        "schema": "v0234-rrtmg-lw-wrf-p8w-t8w-real-wrf-ab-v1",
        "status": "PASS" if improved and public_compatible else "FAIL",
        "is_self_compare": False,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "status_porcelain": _git("status", "--porcelain", "--untracked-files=no"),
        },
        "execution": {
            "platform": jax.default_backend(),
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_rrtmg_invocations_per_arm": 1,
        },
        "authority": {
            "script": script,
            "oracle_dir": str(oracle_dir),
            "oracle_manifest_sha256": EXPECTED["oracle_manifest"],
            "oracle_file_sha256": oracle_hashes,
            "source_run": manifest.get("source_run"),
            "physics_options": manifest.get("physics_options"),
            "wrf_source": source,
            "interface_source": interface_source,
        },
        "real_wrf_glw": {
            "accepted_top_buffer_midpoint_interfaces": before_glw,
            "wrf_p8w_t8w_interfaces": after_glw,
            "rms_ratio": after_glw["rms"] / before_glw["rms"],
            "rms_improvement_fraction": 1.0 - after_glw["rms"] / before_glw["rms"],
            "strictly_improves": improved,
        },
        "real_wrf_lw_heating": {
            "accepted_top_buffer_midpoint_interfaces": before_heating,
            "wrf_p8w_t8w_interfaces": after_heating,
            "rms_ratio": after_heating["rms"] / before_heating["rms"],
        },
        "public_layout": {
            "before_flux_shape": list(before.flux_down.shape),
            "after_flux_shape": list(after.flux_down.shape),
            "backward_compatible": public_compatible,
        },
        "gates": {
            "source_expressions_present": True,
            "authenticated_p3d_t3d_dz8w_bitwise_replay": True,
            "real_wrf_glw_strictly_improves": improved,
            "public_layout_unchanged": public_compatible,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--wrfinput", required=True, type=Path)
    parser.add_argument("--wrf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = run(args.oracle_dir, args.wrfinput, args.wrf_root, args.output)
    print(json.dumps(record["real_wrf_glw"], indent=2, sort_keys=True))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
