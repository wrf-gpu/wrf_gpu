#!/usr/bin/env python3
"""Authenticate WRF O3RAD and run paired real-WRF LW/SW A/B gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np

from gpuwrf.coupling.physics_couplers import _solar_source_scale_for_time
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.wrf_cam_ozone import (
    OZONE_ASSET,
    OZONE_ASSET_SHA256,
    _load_wrf_cam_ozone_asset,
    wrf_cam_ozone_profile,
)


REPO = Path(__file__).resolve().parent.parent
BASE_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_interface_real_wrf_proof.py"
ASSET_MANIFEST = REPO / "data/fixtures/wrf-cam-ozone-v1.json"
EXPECTED = {
    "oracle_manifest": "bdd0f0fd715bf06a78bc61fc63600aff1177685082ee3f044b222b8f365eb136",
    "wrfinput_d01": "a28fb63e890163e26c5797c67fbaedc877cc1c7967100edcd9f08cdfec37ffc6",
    "base_script": "4182fc60c81332fad279b877587606e236fbe7c9c8977c691378ca5de195f62a",
    "cam_support": "005291e1c1da8d58045fbdedbfe0eb7309a7c6f8b5e5670816a95ea6028df6aa",
    "radiation_driver": "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa",
    "rrtmg_lw": "c7a5238612aa8a4213c8d3af6708ec6a5248e6701e19758a80e563905d306de3",
    "rrtmg_sw": "7f8af1da0ca1d25ce784a917bc68600300a7c569881c57e7de8501cd53496b59",
    "registry": "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a",
    "ozone_lat": "761597f3454b99e3dbf15d2621b1ede88269948e96e47f4695b79fabd3cc6348",
    "ozone_plev": "df26a938273b22d2d04ec52b7297827a316cac0061fab020519241cdc0e61d48",
    "ozone": "2a13ee25809672e0419e40062c1fc753580adbf604bfc67a9166ef77c815dbc0",
    "omitted_lw": "576f3ed4243c39171d565317407067a05f15312fde0fe3046531e121495d316a",
    "omitted_sw": "04a78efa4c46370b2daa96721a6b6298b620e68dd098e123388615056b1bf8cb",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"proof helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("v0234_o3_real_base", BASE_SCRIPT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
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
        raise RuntimeError(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    if disk != _git("show", f"HEAD:{relative}", binary=True):
        raise RuntimeError(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _canonical(record: dict[str, Any]) -> str:
    body = {key: value for key, value in record.items() if key != "canonical_payload_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_oracle(
    root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], np.ndarray], dict[str, str]]:
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != EXPECTED["oracle_manifest"]:
        raise RuntimeError("authenticated radiation oracle manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields: dict[tuple[str, str, str], np.ndarray] = {}
    hashes: dict[str, str] = {}
    for item in manifest["fields"]:
        if item["scheme"] not in {"rrtmg_lw", "rrtmg_sw"}:
            continue
        path = root / item["file"]
        if base._md5(path) != item["md5"]:
            raise RuntimeError(f"oracle MD5 drift: {item['file']}")
        raw = np.fromfile(path, dtype=">f8")
        if raw.nbytes != int(item["bytes"]):
            raise RuntimeError(f"oracle byte count drift: {item['file']}")
        fields[(item["scheme"], item["tag"], item["name"])] = np.ascontiguousarray(
            raw.reshape(tuple(item["shape"]))
        )
        hashes[item["file"]] = _sha256(path)
    return manifest, fields, hashes


def _source_gate(wrf_root: Path) -> dict[str, Any]:
    paths = {
        "cam_support": wrf_root / "phys/module_ra_cam_support.F",
        "radiation_driver": wrf_root / "phys/module_radiation_driver.F",
        "rrtmg_lw": wrf_root / "phys/module_ra_rrtmg_lw.F",
        "rrtmg_sw": wrf_root / "phys/module_ra_rrtmg_sw.F",
        "registry": wrf_root / "Registry/Registry.EM_COMMON",
    }
    for label, path in paths.items():
        if _sha256(path) != EXPECTED[label]:
            raise RuntimeError(f"WRF source drift: {label}")
    texts = {
        name: path.read_text(encoding="utf-8", errors="strict")
        for name, path in paths.items()
    }
    required = {
        "default_o3input_two": (
            "registry",
            r"rconfig\s+integer\s+o3input\s+namelist,physics\s+1\s+2\s+-",
        ),
        "read_pressure_table": (
            "cam_support",
            r"FILE='ozone_plev\.formatted'.*?plev\(k\)\s*=\s*plev\(k\)\*100\.",
        ),
        "read_latitude_table": ("cam_support", r"FILE='ozone_lat\.formatted'"),
        "read_monthly_ozone": (
            "cam_support",
            r"FILE='ozone\.formatted'.*?do\s+m=2,num_months.*?do\s+j=1,latsiz.*?do\s+k=1,levsiz",
        ),
        "latitude_interpolation": (
            "cam_support",
            r"ozmixm\(i,k,j,m\)=lin_interpol2\(lat_ozone\(:\),ozmixin\(1,k,:,m\),interp_pt\)",
        ),
        "fixed_365_day_cycle": (
            "radiation_driver",
            r"daysperyear\s*=\s*365\..*?intJULIAN\s*=\s*JULIAN\s*\+\s*1\.0.*?IJUL=MOD\(IJUL,365\)",
        ),
        "monthly_dates": (
            "radiation_driver",
            r"date_oz/16,\s*45,\s*75,\s*105,\s*136,\s*166,\s*197,\s*228,\s*258,\s*289,\s*319,\s*350/",
        ),
        "time_interpolation": (
            "radiation_driver",
            r"ozmixt\(i,k,j\)\s*=\s*ozmixm\(i,k,j,nm\+1\)\*fact1\s*\+\s*ozmixm\(i,k,j,np\+1\)\*fact2",
        ),
        "pressure_interpolation": (
            "radiation_driver",
            r"o3vmr\(i,kout,j\)\s*=\s*\(ozmixt\(i,kupper\(i\),j\)\*dpl\s*\+.*?ozmixt\(i,kupper\(i\)\+1,j\)\*dpu\)/\(dpl\s*\+\s*dpu\)",
        ),
        "lw_o3rad_copy": (
            "rrtmg_lw",
            r"IF\s*\(o3input\.eq\.2\).*?O31D\(K\)=O33D\(I,K,J\)",
        ),
        "lw_shifted_buffer": (
            "rrtmg_lw",
            r"o3vmr\(ncol,k\)\s*=\s*o31d\(kte\)\s*-\s*o3mmr\(kte\)\*amdo\s*\+\s*o3mmr\(k\)\*amdo.*?if\(o3vmr\(ncol,k\)\s*\.le\.\s*0\.\)o3vmr\(ncol,k\)\s*=\s*o3mmr\(k\)\*amdo",
        ),
        "sw_o3rad_copy": (
            "rrtmg_sw",
            r"IF\s*\(o3input\.eq\.2\).*?O31D\(K\)=O33D\(I,K,J\)",
        ),
        "sw_shifted_top": (
            "rrtmg_sw",
            r"o3vmr\(ncol,k\)\s*=\s*o31d\(kte\)\s*-\s*o3mmr\(kte\)\*amdo\s*\+\s*o3mmr\(k\)\*amdo.*?if\(o3vmr\(ncol,k\)\s*\.le\.\s*0\.\)o3vmr\(ncol,k\)\s*=\s*o3mmr\(k\)\*amdo",
        ),
    }
    matches = {
        name: bool(re.search(pattern, texts[source], flags=re.IGNORECASE | re.DOTALL))
        for name, (source, pattern) in required.items()
    }
    if not all(matches.values()):
        raise RuntimeError(f"WRF ozone source-expression gate: {matches}")

    run = wrf_root / "run"
    table_paths = {
        "ozone_lat": run / "ozone_lat.formatted",
        "ozone_plev": run / "ozone_plev.formatted",
        "ozone": run / "ozone.formatted",
    }
    for label, path in table_paths.items():
        if _sha256(path) != EXPECTED[label]:
            raise RuntimeError(f"WRF ozone table drift: {label}")
    latitude, pressure, ozone = _load_wrf_cam_ozone_asset()
    raw_latitude = np.asarray(
        [float(token) for token in table_paths["ozone_lat"].read_text().split()],
        dtype=np.float32,
    )
    raw_pressure = np.asarray(
        [float(token) for token in table_paths["ozone_plev"].read_text().split()],
        dtype=np.float32,
    )
    raw_ozone = np.asarray(
        [float(token) for token in table_paths["ozone"].read_text().split()],
        dtype=np.float32,
    ).reshape(12, 64, 59)
    table_exact = {
        "latitude": bool(np.array_equal(latitude, raw_latitude)),
        "pressure_x100": bool(
            np.array_equal(pressure, (raw_pressure * np.float32(100.0)).astype(np.float32))
        ),
        "ozone_read_order": bool(np.array_equal(ozone, raw_ozone)),
    }
    if not all(table_exact.values()) or _sha256(OZONE_ASSET) != OZONE_ASSET_SHA256:
        raise RuntimeError(f"production ozone asset extraction drift: {table_exact}")
    return {
        "files": {
            label: {"path": str(path), "sha256": EXPECTED[label]}
            for label, path in paths.items()
        },
        "tables": {
            label: {"path": str(path), "sha256": EXPECTED[label]}
            for label, path in table_paths.items()
        },
        "required_expression_matches": matches,
        "asset_sha256": OZONE_ASSET_SHA256,
        "asset_manifest_sha256": _sha256(ASSET_MANIFEST),
        "asset_exact_source_extraction": table_exact,
    }


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or not np.isfinite(candidate).all():
        raise RuntimeError(f"invalid metric inputs: {candidate.shape}:{reference.shape}")
    delta = candidate - reference
    return {
        "shape": list(candidate.shape),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "bias": float(np.mean(delta)),
        "bitwise_mismatch_count": int(np.count_nonzero(candidate != reference)),
        "candidate_min": float(np.min(candidate)),
        "candidate_max": float(np.max(candidate)),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
    }


def _result_digest(result) -> str:
    digest = hashlib.sha256()
    for value in result:
        if value is None:
            digest.update(b"NONE")
        else:
            array = np.asarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _omitted_leaf_regression() -> dict[str, Any]:
    temperature = jnp.asarray(
        [[[291.0, 267.0, 238.0], [292.0, 268.0, 239.0]]], dtype=jnp.float64
    )
    pressure = jnp.asarray(
        [[[90000.0, 51000.0, 17000.0], [91000.0, 52000.0, 18000.0]]],
        dtype=jnp.float64,
    )
    zero = jnp.zeros_like(temperature)
    qv = jnp.full_like(temperature, 1.0e-3)
    dz = jnp.full_like(temperature, 500.0)
    rho = jnp.full_like(temperature, 0.8)
    lw = RRTMGLWColumnState(
        temperature,
        pressure,
        qv,
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([[292.0, 293.0]], dtype=jnp.float64),
        jnp.asarray([[0.97, 0.98]], dtype=jnp.float64),
        dz,
        rho,
        5000.0,
    )
    sw = RRTMGSWColumnState(
        temperature,
        pressure,
        qv,
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([[0.12, 0.18]], dtype=jnp.float64),
        jnp.asarray([[0.35, 0.42]], dtype=jnp.float64),
        dz,
        rho,
    )
    lw_result = solve_rrtmg_lw_column(lw, debug=False)
    sw_result = solve_rrtmg_sw_column(sw, debug=False)
    jax.block_until_ready((lw_result.surface_down, sw_result.surface_down))
    observed = {"lw": _result_digest(lw_result), "sw": _result_digest(sw_result)}
    expected = {"lw": EXPECTED["omitted_lw"], "sw": EXPECTED["omitted_sw"]}
    if observed != expected:
        raise RuntimeError(f"omitted O3 leaf changed historical solver output: {observed}")
    return {
        "baseline_commit": "9bfe1e30",
        "expected_aggregate_sha256": expected,
        "observed_aggregate_sha256": observed,
        "byte_identical": True,
    }


def run(oracle_dir: Path, wrfinput: Path, wrf_root: Path, output: Path) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX_EXECUTION_MODE")
    if _sha256(BASE_SCRIPT) != EXPECTED["base_script"]:
        raise RuntimeError("authenticated oracle helper drift")
    if _sha256(wrfinput) != EXPECTED["wrfinput_d01"]:
        raise RuntimeError("source-run wrfinput drift")
    tracked = {
        "script": _tracked(Path(__file__).resolve()),
        "extractor": _tracked(REPO / "scripts/extract_wrf_cam_ozone.py"),
        "cam_ozone": _tracked(REPO / "src/gpuwrf/physics/wrf_cam_ozone.py"),
        "rrtmg_lw": _tracked(REPO / "src/gpuwrf/physics/rrtmg_lw.py"),
        "rrtmg_sw": _tracked(REPO / "src/gpuwrf/physics/rrtmg_sw.py"),
        "coupler": _tracked(REPO / "src/gpuwrf/coupling/physics_couplers.py"),
        "focused_test": _tracked(REPO / "tests/test_v0234_rrtmg_o3rad.py"),
        "asset": _tracked(OZONE_ASSET),
        "asset_manifest": _tracked(ASSET_MANIFEST),
    }
    source = _source_gate(wrf_root)
    manifest, fields, oracle_hashes = _load_oracle(oracle_dir)

    with Dataset(wrfinput) as dataset:
        start_date = str(dataset.getncattr("START_DATE"))
        julyr = int(dataset.getncattr("JULYR"))
        julday = int(dataset.getncattr("JULDAY"))
        wrfinput_xlat = np.asarray(dataset["XLAT"][0], dtype=np.float32)
    start = datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    minute = float(start.hour * 60 + start.minute) + float(start.second) / 60.0
    if not (julyr == 2026 and julday == 118 and minute == 1080.0):
        raise RuntimeError(f"source-run clock drift: {start_date}:{julyr}:{julday}")

    def col3(scheme: str, name: str) -> jax.Array:
        array = fields[(scheme, "in", name)]
        nj, nk, ni = array.shape
        return jnp.asarray(
            np.moveaxis(array, 1, 2).reshape(nj * ni, nk), dtype=jnp.float64
        )

    def surface(scheme: str, tag: str, name: str) -> jax.Array:
        return jnp.asarray(fields[(scheme, tag, name)].reshape(-1), dtype=jnp.float64)

    lw_xlat = np.asarray(fields[("rrtmg_lw", "in", "xlat")], dtype=np.float32)
    sw_xlat = np.asarray(fields[("rrtmg_sw", "in", "xlat")], dtype=np.float32)
    if not (
        np.array_equal(lw_xlat, sw_xlat)
        and np.array_equal(lw_xlat, wrfinput_xlat)
    ):
        raise RuntimeError("authenticated WRF latitude sources disagree")
    hydrostatic_p = col3("rrtmg_lw", "p")
    ozone = wrf_cam_ozone_profile(
        jnp.asarray(lw_xlat.reshape(-1), dtype=jnp.float32),
        hydrostatic_p,
        julian_day_1based=julday,
        utc_minute=minute,
    )
    jax.block_until_ready(ozone)
    ozone_np = np.asarray(ozone)
    if ozone_np.shape != (5487, 44) or not np.all(ozone_np > 0.0):
        raise RuntimeError(f"invalid production O3RAD: {ozone_np.shape}")

    lw_qv = col3("rrtmg_lw", "qv")
    lw_common = dict(
        T=col3("rrtmg_lw", "t"),
        p=hydrostatic_p,
        qv=lw_qv,
        qc=col3("rrtmg_lw", "qc"),
        qi=col3("rrtmg_lw", "qi"),
        qs=col3("rrtmg_lw", "qs"),
        qg=jnp.zeros_like(lw_qv),
        cloud_fraction=col3("rrtmg_lw", "cldfra"),
        surface_temperature=surface("rrtmg_lw", "in", "tsk"),
        surface_emissivity=surface("rrtmg_lw", "in", "emiss"),
        dz=col3("rrtmg_lw", "dz8w"),
        rho=col3("rrtmg_lw", "rho"),
        top_pressure_pa=5000.0,
    )
    lw_annual_state = RRTMGLWColumnState(**lw_common)
    lw_o3_state = RRTMGLWColumnState(**lw_common, ozone_vmr=ozone)
    lw_annual = solve_rrtmg_lw_column(lw_annual_state, debug=False)
    lw_o3 = solve_rrtmg_lw_column(lw_o3_state, debug=False)
    jax.block_until_ready((lw_annual.surface_down, lw_o3.surface_down))

    sw_qv = col3("rrtmg_sw", "qv")
    sw_common = dict(
        T=col3("rrtmg_sw", "t"),
        p=col3("rrtmg_sw", "p"),
        qv=sw_qv,
        qc=col3("rrtmg_sw", "qc"),
        qi=col3("rrtmg_sw", "qi"),
        qs=col3("rrtmg_sw", "qs"),
        qg=jnp.zeros_like(sw_qv),
        cloud_fraction=col3("rrtmg_sw", "cldfra"),
        surface_albedo=surface("rrtmg_sw", "in", "albedo"),
        coszen=surface("rrtmg_sw", "in", "coszen"),
        dz=col3("rrtmg_sw", "dz8w"),
        rho=col3("rrtmg_sw", "rho"),
        solar_source_scale=_solar_source_scale_for_time(start, 0.0),
    )
    sw_annual_state = RRTMGSWColumnState(**sw_common)
    sw_o3_state = RRTMGSWColumnState(**sw_common, ozone_vmr=ozone)
    sw_annual = solve_rrtmg_sw_column(sw_annual_state, debug=False)
    sw_o3 = solve_rrtmg_sw_column(sw_o3_state, debug=False)
    jax.block_until_ready((sw_annual.surface_down, sw_o3.surface_down))

    glw_reference = np.asarray(surface("rrtmg_lw", "out", "glw"))
    swdnb_reference = np.asarray(surface("rrtmg_sw", "out", "swdnb"))
    glw = {
        "accepted_static_top_annual_ozone": _metrics(
            np.asarray(lw_annual.surface_down), glw_reference
        ),
        "candidate_static_top_o3rad": _metrics(
            np.asarray(lw_o3.surface_down), glw_reference
        ),
        "candidate_minus_accepted": _metrics(
            np.asarray(lw_o3.surface_down), np.asarray(lw_annual.surface_down)
        ),
    }
    swdnb = {
        "accepted_midpoint_annual_ozone": _metrics(
            np.asarray(sw_annual.surface_down), swdnb_reference
        ),
        "candidate_midpoint_o3rad": _metrics(
            np.asarray(sw_o3.surface_down), swdnb_reference
        ),
        "candidate_minus_accepted": _metrics(
            np.asarray(sw_o3.surface_down), np.asarray(sw_annual.surface_down)
        ),
    }
    lw_improves = bool(
        glw["candidate_static_top_o3rad"]["rms"]
        < glw["accepted_static_top_annual_ozone"]["rms"]
    )
    sw_improves = bool(
        swdnb["candidate_midpoint_o3rad"]["rms"]
        < swdnb["accepted_midpoint_annual_ozone"]["rms"]
    )
    omitted = _omitted_leaf_regression()
    record = {
        "schema": "v0234-wrf-o3input2-lw-sw-real-wrf-ab-v1",
        "status": "PASS" if lw_improves and sw_improves else "FAIL",
        "verdict": (
            "WRF_O3RAD_STRICTLY_IMPROVES_AUTHENTIC_GLW_AND_SWDNB"
            if lw_improves and sw_improves
            else "WRF_O3RAD_REAL_WRF_COMPONENT_GATE_FALSIFIED"
        ),
        "is_self_compare": False,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "status_porcelain": _git("status", "--porcelain", "--untracked-files=no"),
        },
        "execution": {
            "platform": jax.default_backend(),
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_solver_invocations": {"lw": 2, "sw": 2},
        },
        "authority": {
            "tracked": tracked,
            "oracle_dir": str(oracle_dir),
            "oracle_manifest_sha256": EXPECTED["oracle_manifest"],
            "oracle_file_sha256": oracle_hashes,
            "source_run": manifest.get("source_run"),
            "physics_options": manifest.get("physics_options"),
            "wrf_source": source,
            "wrfinput": {
                "path": str(wrfinput),
                "sha256": EXPECTED["wrfinput_d01"],
                "start_date": start_date,
                "julyr": julyr,
                "julday": julday,
                "utc_minute": minute,
                "o3input": 2,
                "o3input_authority": "pinned Registry.EM_COMMON default; source run does not override it",
            },
        },
        "o3rad": {
            "shape": list(ozone_np.shape),
            "dtype": str(ozone_np.dtype),
            "payload_sha256": hashlib.sha256(ozone_np.tobytes()).hexdigest(),
            "minimum_vmr": float(np.min(ozone_np)),
            "maximum_vmr": float(np.max(ozone_np)),
            "same_leaf_supplied_to_lw_and_sw": True,
            "diagnostic_pressure": "authenticated WRF hydrostatic P3D",
            "latitude": "authenticated WRF XLAT, bitwise equal to wrfinput",
        },
        "paired_ab": {
            "only_changed_input": "LW+SW RRTMGColumnState.ozone_vmr",
            "accepted_composition": "QML + static LW top pressure + annual INIRAD ozone",
            "candidate_composition": "QML + static LW top pressure + exact WRF O3RAD",
            "dynamic_p3d_p8w_t8w_population": False,
            "run_date_ghg_population": False,
        },
        "real_wrf_glw": glw,
        "real_wrf_swdnb": swdnb,
        "omitted_leaf_regression": omitted,
        "gates": {
            "authenticated_real_wrf_reference": True,
            "wrf_source_expressions_present": True,
            "cam_table_extraction_exact": True,
            "same_o3rad_to_lw_and_sw": True,
            "real_wrf_glw_strictly_improves": lw_improves,
            "real_wrf_swdnb_strictly_improves": sw_improves,
            "omitted_leaves_byte_identical_to_9bfe1e30": omitted["byte_identical"],
            "no_empirical_tuning": True,
            "no_synthetic_reference": True,
        },
    }
    record["canonical_payload_sha256"] = _canonical(record)
    _atomic_json(output, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--wrfinput", required=True, type=Path)
    parser.add_argument("--wrf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        record = run(
            args.oracle_dir.resolve(),
            args.wrfinput.resolve(),
            args.wrf_root.resolve(),
            args.output.resolve(),
        )
        print(
            json.dumps(
                {
                    "verdict": record["verdict"],
                    "glw_annual_rms": record["real_wrf_glw"][
                        "accepted_static_top_annual_ozone"
                    ]["rms"],
                    "glw_o3rad_rms": record["real_wrf_glw"][
                        "candidate_static_top_o3rad"
                    ]["rms"],
                    "swdnb_annual_rms": record["real_wrf_swdnb"][
                        "accepted_midpoint_annual_ozone"
                    ]["rms"],
                    "swdnb_o3rad_rms": record["real_wrf_swdnb"][
                        "candidate_midpoint_o3rad"
                    ]["rms"],
                },
                sort_keys=True,
            )
        )
        return 0 if record["status"] == "PASS" else 1
    except Exception as exc:  # noqa: BLE001 - proof must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
