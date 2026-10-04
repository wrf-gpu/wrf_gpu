#!/usr/bin/env python3
"""Authenticated real-WRF A/B for the v0234 RRTMG-LW top-buffer fix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.rrtmg_lw import (
    RRTMGLWColumnState,
    _lw_buffer_layer_count,
    _lw_extended_pressure_profiles,
    solve_rrtmg_lw_column,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the oracle's pinned manifest
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_authenticated_oracle(root: Path) -> tuple[dict[str, Any], dict[tuple[str, str, str], np.ndarray], dict[str, str]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    hashes = {"manifest.json": _sha256(manifest_path)}
    for field in manifest["fields"]:
        if field["scheme"] != "rrtmg_lw":
            continue
        path = root / field["file"]
        observed_md5 = _md5(path)
        if observed_md5 != field["md5"]:
            raise RuntimeError(
                f"oracle MD5 drift for {field['file']}: {observed_md5} != {field['md5']}"
            )
        raw = np.fromfile(path, dtype=">f8")
        if raw.size * raw.dtype.itemsize != int(field["bytes"]):
            raise RuntimeError(f"oracle byte-count drift for {field['file']}")
        arrays[(field["scheme"], field["tag"], field["name"])] = np.ascontiguousarray(
            raw.reshape(tuple(field["shape"]))
        )
        hashes[field["file"]] = _sha256(path)
    return manifest, arrays, hashes


def _source_authority(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    required = {
        "deltap_4mb": r"real\s*,\s*PARAMETER\s*::\s*deltap\s*=\s*4\.",
        "nlayers_from_ptop": r"NLAYERS\s*=\s*kme\s*\+\s*nint\(p_top\*0\.01/deltap\)\s*-\s*1",
        "four_mb_ladder": r"plev\(ncol,L\+1\)\s*=\s*plev\(ncol,L\)\s*-\s*deltap",
        "forced_zero_toa": r"plev\(ncol,nlayers\+1\)\s*=\s*0\.00",
        "temperature_profile": r"tlev\(ncol,L\)\s*=\s*varint\(L\)\s*\+\s*\(tlev\(ncol,kte\)\s*-\s*varint\(kte\)\)",
    }
    matches = {name: bool(re.search(pattern, text, flags=re.IGNORECASE)) for name, pattern in required.items()}
    if not all(matches.values()):
        raise RuntimeError(f"pristine WRF source-expression gate failed: {matches}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "required_expression_matches": matches,
    }


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise RuntimeError(f"shape mismatch {candidate.shape} != {reference.shape}")
    delta = candidate - reference
    return {
        "shape": list(candidate.shape),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "bias": float(np.mean(delta)),
        "candidate_min": float(np.min(candidate)),
        "candidate_max": float(np.max(candidate)),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "finite": bool(np.all(np.isfinite(candidate))),
    }


def run(oracle_dir: Path, wrf_source: Path, out: Path, top_pressure_pa: float) -> dict[str, Any]:
    manifest, arrays, oracle_hashes = _load_authenticated_oracle(oracle_dir)
    source = _source_authority(wrf_source)

    def col3(name: str) -> jax.Array:
        array = arrays[("rrtmg_lw", "in", name)]
        nj, nk, ni = array.shape
        return jnp.asarray(np.moveaxis(array, 1, 2).reshape(nj * ni, nk), dtype=jnp.float64)

    def surface(tag: str, name: str) -> jax.Array:
        return jnp.asarray(arrays[("rrtmg_lw", tag, name)].reshape(-1), dtype=jnp.float64)

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
    tsk = surface("in", "tsk")
    emiss = surface("in", "emiss")

    common = dict(
        T=T,
        p=p,
        qv=qv,
        qc=qc,
        qi=qi,
        qs=qs,
        qg=zero,
        cloud_fraction=cloud_fraction,
        surface_temperature=tsk,
        surface_emissivity=emiss,
        dz=dz,
        rho=rho,
    )
    legacy_state = RRTMGLWColumnState(**common)
    candidate_state = RRTMGLWColumnState(**common, top_pressure_pa=top_pressure_pa)

    legacy = solve_rrtmg_lw_column(legacy_state, debug=False)
    candidate = solve_rrtmg_lw_column(candidate_state, debug=False)
    jax.block_until_ready((legacy.surface_down, candidate.surface_down))

    glw_reference = np.asarray(surface("out", "glw"))
    pi3d = np.asarray(col3("pi3d"))
    rthraten = arrays[("rrtmg_lw", "out", "rthratenlw")]
    nj, nk, ni = rthraten.shape
    heating_reference = np.moveaxis(rthraten, 1, 2).reshape(nj * ni, nk) * pi3d
    legacy_glw = _metrics(np.asarray(legacy.surface_down), glw_reference)
    candidate_glw = _metrics(np.asarray(candidate.surface_down), glw_reference)
    legacy_heating = _metrics(np.asarray(legacy.heating_rate), heating_reference)
    candidate_heating = _metrics(np.asarray(candidate.heating_rate), heating_reference)

    buffer_count = _lw_buffer_layer_count(top_pressure_pa)
    _, pressure_interfaces, buffer_pressure = _lw_extended_pressure_profiles(
        p[:1], top_pressure_pa
    )
    model_layers = int(p.shape[-1])
    buffer_interfaces = np.asarray(pressure_interfaces)[0, model_layers:]
    expected_interfaces = np.concatenate(
        (
            np.asarray([top_pressure_pa], dtype=np.float64),
            top_pressure_pa
            - 100.0 * 4.0 * np.arange(1, buffer_count, dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        )
    )
    buffer_exact = bool(np.array_equal(buffer_interfaces, expected_interfaces))
    improved = bool(candidate_glw["rms"] < legacy_glw["rms"])
    record = {
        "schema": "v0234-rrtmg-lw-top-buffer-real-wrf-ab-v1",
        "status": "PASS" if improved and buffer_exact else "FAIL",
        "is_self_compare": False,
        "execution": {
            "platform": jax.default_backend(),
            "gpu_used": False,
            "wrf_or_mpi_executed": False,
        },
        "authority": {
            "oracle_dir": str(oracle_dir),
            "oracle_manifest_sha256": oracle_hashes.pop("manifest.json"),
            "oracle_file_sha256": oracle_hashes,
            "source_run": manifest.get("source_run"),
            "physics_options": manifest.get("physics_options"),
            "wrf_source": source,
        },
        "buffer_source_replay": {
            "top_pressure_pa": top_pressure_pa,
            "deltap_mb": 4.0,
            "buffer_layer_count": buffer_count,
            "model_layer_count": model_layers,
            "buffer_interface_pressure_pa": buffer_interfaces.tolist(),
            "buffer_layer_pressure_pa": np.asarray(buffer_pressure)[0].tolist(),
            "expected_interface_pressure_pa": expected_interfaces.tolist(),
            "bitwise_exact": buffer_exact,
        },
        "real_wrf_glw": {
            "legacy_one_layer": legacy_glw,
            "wrf_top_buffer": candidate_glw,
            "rms_ratio_candidate_over_legacy": candidate_glw["rms"] / legacy_glw["rms"],
            "rms_improvement_fraction": 1.0 - candidate_glw["rms"] / legacy_glw["rms"],
            "strictly_improves": improved,
        },
        "real_wrf_lw_heating": {
            "legacy_one_layer": legacy_heating,
            "wrf_top_buffer": candidate_heating,
            "rms_ratio_candidate_over_legacy": candidate_heating["rms"] / legacy_heating["rms"],
        },
        "public_layout": {
            "legacy_flux_shape": list(legacy.flux_down.shape),
            "candidate_flux_shape": list(candidate.flux_down.shape),
            "expected_last_axis": model_layers + 2,
            "backward_compatible": bool(
                legacy.flux_down.shape == candidate.flux_down.shape
                and candidate.flux_down.shape[-1] == model_layers + 2
            ),
        },
    }
    if not record["public_layout"]["backward_compatible"]:
        record["status"] = "FAIL"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--wrf-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-pressure-pa", type=float, default=5000.0)
    args = parser.parse_args()
    record = run(args.oracle_dir, args.wrf_source, args.out, args.top_pressure_pa)
    print(json.dumps(record["real_wrf_glw"], indent=2, sort_keys=True))
    if record["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
