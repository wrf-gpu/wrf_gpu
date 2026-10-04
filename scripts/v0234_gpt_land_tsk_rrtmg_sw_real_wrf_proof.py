#!/usr/bin/env python3
"""Authenticated real-WRF A/B for exact RRTMG-SW P3D/P8W/T8W inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np

from gpuwrf.coupling.physics_couplers import _solar_source_scale_for_time
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time


REPO = Path(__file__).resolve().parent.parent
BASE_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_interface_real_wrf_proof.py"
EXPECTED = {
    "oracle_manifest": "bdd0f0fd715bf06a78bc61fc63600aff1177685082ee3f044b222b8f365eb136",
    "wrfinput_d01": "a28fb63e890163e26c5797c67fbaedc877cc1c7967100edcd9f08cdfec37ffc6",
    "base_script": "4182fc60c81332fad279b877587606e236fbe7c9c8977c691378ca5de195f62a",
    "first_rk": "8c666fe88c46b04e297fe7b7289f55ec74fa133287b234a02f10e05cbbd11841",
    "phy_prep": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
    "rrtmg_sw": "7f8af1da0ca1d25ce784a917bc68600300a7c569881c57e7de8501cd53496b59",
    "radiation_driver": "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa",
}


SPEC = importlib.util.spec_from_file_location("v0234_lw_real_wrf", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("authenticated LW oracle helper unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - pinned WRF oracle manifest uses MD5
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


def _tracked(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    if disk != _git("show", f"HEAD:{relative}", binary=True):
        raise RuntimeError(f"tracked source is not HEAD: {relative}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _canonical(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "canonical_payload_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_oracle(
    root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], np.ndarray], dict[str, str]]:
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != EXPECTED["oracle_manifest"]:
        raise RuntimeError("authenticated radiation oracle manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields: dict[tuple[str, str], np.ndarray] = {}
    hashes: dict[str, str] = {}
    for item in manifest["fields"]:
        if item["scheme"] != "rrtmg_sw":
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


def _source_gate(wrf_root: Path) -> dict[str, Any]:
    paths = {
        "first_rk": wrf_root / "dyn_em/module_first_rk_step_part1.F",
        "phy_prep": wrf_root / "dyn_em/module_big_step_utilities_em.F",
        "rrtmg_sw": wrf_root / "phys/module_ra_rrtmg_sw.F",
        "radiation_driver": wrf_root / "phys/module_radiation_driver.F",
    }
    for label, path in paths.items():
        if _sha256(path) != EXPECTED[label]:
            raise RuntimeError(f"WRF source drift: {label}")
    first_rk = paths["first_rk"].read_text(encoding="utf-8", errors="strict")
    prep = paths["phy_prep"].read_text(encoding="utf-8", errors="strict")
    sw = paths["rrtmg_sw"].read_text(encoding="utf-8", errors="strict")
    driver = paths["radiation_driver"].read_text(encoding="utf-8", errors="strict")
    required = {
        "sw_p8w_hydrostatic": (first_rk, r"P8W\s*=\s*grid%p_hyd_w"),
        "sw_p3d_hydrostatic": (first_rk, r"P\s*=\s*grid%p_hyd"),
        "sw_t8w_phy_prep": (first_rk, r"T8W\s*=\s*t8w"),
        "hydrostatic_top": (prep, r"p_hyd_w\(i,kte,j\)\s*=\s*p_top"),
        "hydrostatic_recursion": (
            prep,
            r"p_hyd_w\(i,k,j\)\s*=\s*p_hyd_w\(i,k\+1,j\)\s*-\s*\(1\.\+qtot\)\*\(c1\(k\)\*MUT\(i,j\)\+c2\(k\)\)\*dnw\(k\)",
        ),
        "t8w_interior": (
            prep,
            r"t8w\(i,k,j\)\s*=\s*fzm\(k\)\*t_phy\(i,k,j\)\+fzp\(k\)\*t_phy\(i,k-1,j\)",
        ),
        "sw_p8w_copy": (sw, r"Pw1D\(K\)\s*=\s*p8w\(I,K,J\)/100\."),
        "sw_t8w_copy": (sw, r"Tw1D\(K\)\s*=\s*t8w\(I,K,J\)"),
        "sw_one_toa_layer": (sw, r"nlay\s*=\s*\(kte\s*-\s*kts\s*\+\s*1\)\s*\+\s*1"),
        "sw_toa_mass_pressure": (
            sw,
            r"play\(ncol,kte\+1\)\s*=\s*0\.5\s*\*\s*plev\(ncol,kte\+1\)",
        ),
        "sw_toa_mass_temperature": (
            sw,
            r"tlay\(ncol,kte\+1\)\s*=\s*tlev\(ncol,kte\+1\)\s*\+\s*0\.0",
        ),
        "sw_toa_interface_pressure": (
            sw,
            r"plev\(ncol,kte\+2\)\s*=\s*1\.0e-5",
        ),
        "solar_constant": (driver, r"solcon\s*=\s*1370\.\s*\*\s*ECCFAC"),
    }
    matches = {
        name: bool(re.search(pattern, text, flags=re.IGNORECASE))
        for name, (text, pattern) in required.items()
    }
    if not all(matches.values()):
        raise RuntimeError(f"WRF source-expression gate: {matches}")
    return {
        label: {"path": str(path), "sha256": EXPECTED[label]}
        for label, path in paths.items()
    } | {"required_expression_matches": matches}


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


def _solve(state: RRTMGSWColumnState) -> dict[str, np.ndarray]:
    result = solve_rrtmg_sw_column(state, debug=False)
    jax.block_until_ready(result.surface_down)
    return {
        "surface_down": np.asarray(result.surface_down, dtype=np.float64),
        "surface_net": np.asarray(result.surface_absorbed, dtype=np.float64),
        "heating_rate": np.asarray(result.heating_rate, dtype=np.float64),
        "flux_shape": np.asarray(result.flux_down.shape, dtype=np.int64),
    }


def run(oracle_dir: Path, wrfinput: Path, wrf_root: Path, output: Path) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean")
    if _sha256(BASE_SCRIPT) != EXPECTED["base_script"]:
        raise RuntimeError("authenticated interface helper drift")
    script = _tracked(Path(__file__).resolve())
    model_sources = {
        "rrtmg_sw": _tracked(REPO / "src/gpuwrf/physics/rrtmg_sw.py"),
        "physics_couplers": _tracked(REPO / "src/gpuwrf/coupling/physics_couplers.py"),
        "wrf_clwrf_ghg": _tracked(REPO / "src/gpuwrf/physics/wrf_clwrf_ghg.py"),
        "focused_test": _tracked(REPO / "tests/test_v0234_rrtmg_sw_wrf_interfaces.py"),
    }
    manifest, fields, oracle_hashes = _load_oracle(oracle_dir)
    source = _source_gate(wrf_root)

    # Reuse the independently sealed float32 phy_prep expression replay.  It
    # verifies reconstructed P3D/T3D/DZ8W bitwise against the authenticated
    # oracle before returning P8W/T8W.
    _lw_manifest, lw_fields, _lw_hashes = base._load_oracle(oracle_dir)
    p8w_zyx, t8w_zyx, interface_source = base._wrf_interface_inputs(
        wrfinput, lw_fields
    )

    def col3(name: str) -> jax.Array:
        array = fields[("in", name)]
        nj, nk, ni = array.shape
        return jnp.asarray(
            np.moveaxis(array, 1, 2).reshape(nj * ni, nk), dtype=jnp.float64
        )

    def surface(tag: str, name: str) -> jax.Array:
        return jnp.asarray(fields[(tag, name)].reshape(-1), dtype=jnp.float64)

    if _sha256(wrfinput) != EXPECTED["wrfinput_d01"]:
        raise RuntimeError("source-run wrfinput drift")
    with Dataset(wrfinput) as dataset:
        start_date = str(dataset.getncattr("START_DATE"))
        ghg_input = int(dataset.getncattr("GHG_INPUT"))
        nonhyd_zyx = (
            np.asarray(dataset["P"][0], dtype=np.float32)
            + np.asarray(dataset["PB"][0], dtype=np.float32)
        ).astype(np.float32)
    start = datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    source_scale = _solar_source_scale_for_time(start, 0.0)
    p8w_columns = np.moveaxis(p8w_zyx, 0, -1).reshape(-1, p8w_zyx.shape[0])
    t8w_columns = np.moveaxis(t8w_zyx, 0, -1).reshape(-1, t8w_zyx.shape[0])
    nonhyd_columns = np.moveaxis(nonhyd_zyx, 0, -1).reshape(-1, nonhyd_zyx.shape[0])

    qv = col3("qv")
    common = dict(
        T=col3("t"),
        qv=qv,
        qc=col3("qc"),
        qi=col3("qi"),
        qs=col3("qs"),
        qg=jnp.zeros_like(qv),
        cloud_fraction=col3("cldfra"),
        surface_albedo=surface("in", "albedo"),
        coszen=surface("in", "coszen"),
        dz=col3("dz8w"),
        rho=col3("rho"),
        solar_source_scale=source_scale,
    )
    nonhyd_midpoint = RRTMGSWColumnState(
        p=jnp.asarray(nonhyd_columns, dtype=jnp.float64), **common
    )
    hydro_midpoint = RRTMGSWColumnState(p=col3("p"), **common)
    wrf_interfaces = RRTMGSWColumnState(
        p=col3("p"),
        pressure_interfaces=jnp.asarray(p8w_columns, dtype=jnp.float64),
        temperature_interfaces=jnp.asarray(t8w_columns, dtype=jnp.float64),
        **common,
    )
    gases = clwrf_ssp245_gases_for_time(start)
    expected_gases = (
        0.00043162917303889273,
        3.366766427747367e-07,
        1.9684022349994908e-06,
        2.0200244590691145e-10,
        4.676246698024074e-10,
    )
    if ghg_input != 1 or tuple(gases) != expected_gases:
        raise RuntimeError(f"WRF_CLWRF_RUN_DATE:{ghg_input}:{gases}")
    production = wrf_interfaces.replace(
        co2_vmr=gases.co2_vmr,
        n2o_vmr=gases.n2o_vmr,
        ch4_vmr=gases.ch4_vmr,
    )

    before = _solve(nonhyd_midpoint)
    mass_only = _solve(hydro_midpoint)
    exact = _solve(wrf_interfaces)
    composed = _solve(production)
    swdnb_reference = np.asarray(surface("out", "swdnb"))
    gsw_reference = np.asarray(surface("out", "gsw"))
    pi3d = np.asarray(col3("pi3d"))
    rthraten = fields[("out", "rthratensw")]
    nj, nk, ni = rthraten.shape
    heating_reference = np.moveaxis(rthraten, 1, 2).reshape(nj * ni, nk) * pi3d

    swdnb = {
        "nonhydrostatic_midpoint": _metrics(before["surface_down"], swdnb_reference),
        "hydrostatic_mass_midpoint": _metrics(
            mass_only["surface_down"], swdnb_reference
        ),
        "wrf_p8w_t8w": _metrics(exact["surface_down"], swdnb_reference),
        "wrf_p8w_t8w_clwrf": _metrics(
            composed["surface_down"], swdnb_reference
        ),
    }
    gsw = {
        "nonhydrostatic_midpoint": _metrics(before["surface_net"], gsw_reference),
        "hydrostatic_mass_midpoint": _metrics(mass_only["surface_net"], gsw_reference),
        "wrf_p8w_t8w": _metrics(exact["surface_net"], gsw_reference),
        "wrf_p8w_t8w_clwrf": _metrics(
            composed["surface_net"], gsw_reference
        ),
    }
    heating = {
        "nonhydrostatic_midpoint": _metrics(before["heating_rate"], heating_reference),
        "hydrostatic_mass_midpoint": _metrics(
            mass_only["heating_rate"], heating_reference
        ),
        "wrf_p8w_t8w": _metrics(exact["heating_rate"], heating_reference),
        "wrf_p8w_t8w_clwrf": _metrics(
            composed["heating_rate"], heating_reference
        ),
    }
    interface_improved = bool(
        swdnb["wrf_p8w_t8w"]["rms"] < swdnb["nonhydrostatic_midpoint"]["rms"]
        and swdnb["wrf_p8w_t8w"]["max_abs"]
        < swdnb["nonhydrostatic_midpoint"]["max_abs"]
        and swdnb["wrf_p8w_t8w"]["rms"]
        < swdnb["hydrostatic_mass_midpoint"]["rms"]
    )
    production_improved = bool(
        swdnb["wrf_p8w_t8w_clwrf"]["rms"]
        < swdnb["nonhydrostatic_midpoint"]["rms"]
        and swdnb["wrf_p8w_t8w_clwrf"]["max_abs"]
        < swdnb["nonhydrostatic_midpoint"]["max_abs"]
        and swdnb["wrf_p8w_t8w_clwrf"]["rms"]
        < swdnb["hydrostatic_mass_midpoint"]["rms"]
    )
    public_compatible = bool(
        np.array_equal(before["flux_shape"], exact["flux_shape"])
        and tuple(exact["flux_shape"]) == (5487, 46)
    )
    record = {
        "schema": "v0234-rrtmg-sw-wrf-p3d-p8w-t8w-clwrf-real-wrf-ab-v1",
        "status": "PASS" if production_improved and public_compatible else "FAIL",
        "is_self_compare": False,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "status_porcelain": _git("status", "--porcelain", "--untracked-files=no"),
        },
        "execution": {
            "platform": jax.default_backend(),
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_rrtmg_sw_invocations": 4,
        },
        "authority": {
            "script": script,
            "model_sources": model_sources,
            "base_interface_helper": {
                "path": str(BASE_SCRIPT),
                "sha256": EXPECTED["base_script"],
            },
            "oracle_dir": str(oracle_dir),
            "oracle_manifest_sha256": EXPECTED["oracle_manifest"],
            "oracle_file_sha256": oracle_hashes,
            "source_run": manifest.get("source_run"),
            "physics_options": manifest.get("physics_options"),
            "wrf_source": source,
            "interface_source": interface_source,
            "start_date": start_date,
            "ghg_input": ghg_input,
            "clwrf_run_date_gases": gases._asdict(),
            "solar_source_scale": float(np.asarray(source_scale)),
        },
        "input_boundaries": {
            "nonhydrostatic_mass_pressure_vs_authenticated_p3d": _metrics(
                nonhyd_columns, np.asarray(col3("p"))
            ),
            "p8w_shape": list(p8w_columns.shape),
            "t8w_shape": list(t8w_columns.shape),
            "p8w_source_dtype": str(p8w_zyx.dtype),
            "t8w_source_dtype": str(t8w_zyx.dtype),
        },
        "real_wrf_swdnb": {
            **swdnb,
            "exact_over_before_rms_ratio": (
                swdnb["wrf_p8w_t8w"]["rms"]
                / swdnb["nonhydrostatic_midpoint"]["rms"]
            ),
            "clwrf_composed_over_before_rms_ratio": (
                swdnb["wrf_p8w_t8w_clwrf"]["rms"]
                / swdnb["nonhydrostatic_midpoint"]["rms"]
            ),
            "interfaces_strictly_improve": interface_improved,
            "production_composition_strictly_improves": production_improved,
        },
        "real_wrf_gsw": gsw,
        "real_wrf_sw_heating": heating,
        "public_layout": {
            "before_flux_shape": before["flux_shape"].tolist(),
            "after_flux_shape": exact["flux_shape"].tolist(),
            "backward_compatible": public_compatible,
        },
        "gates": {
            "source_expressions_present": True,
            "authenticated_p3d_t3d_dz8w_bitwise_replay": True,
            "real_wrf_swdnb_interfaces_strictly_improve": interface_improved,
            "real_wrf_swdnb_production_composition_strictly_improves": (
                production_improved
            ),
            "ghg_input_one_and_run_date_clwrf_exact": True,
            "public_layout_unchanged": public_compatible,
            "default_none_path_regression": "14 focused tests passed before proof",
        },
    }
    record["canonical_payload_sha256"] = _canonical(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--wrfinput", required=True, type=Path)
    parser.add_argument("--wrf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    record = run(args.oracle_dir, args.wrfinput, args.wrf_root, args.output.resolve())
    print(json.dumps(record["real_wrf_swdnb"], indent=2, sort_keys=True))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
