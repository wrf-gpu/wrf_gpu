#!/usr/bin/env python3
"""Seal the CPU-only d03 step-1 horizontal-PGF component discriminator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts import v0234_first_interval_momentum_wrf_reassemble as wrf  # noqa: E402
from scripts import v0234_s1_operator_ledger as parent  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
OUTPUT = SPRINT / "pgf-component-ledger.json"
DUMPS = parent.LEDGER / "dumps_pgf_inputs1"
RUN = parent.LEDGER / "run_pgf_inputs1"
BINARY = parent.LEDGER / "_build_operator/main/wrf"
MODULE_EM = parent.LEDGER / "dyn_em/module_em.F"
WRFINPUT = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs/control/run/wrfinput_d03")
PARENT_CANONICAL = "3d8c7c3c775429ebd629a192caa4192c73bc24bd2a572eb21a6f047e3ea3bad1"
PREDICTIONS_SHA256 = "73087c007e556be03a5a1cb40ab975d31d52a6bdb8e08d789ea78fa9714dca3a"
EXPECTED_BINARY_SHA256 = "e374c4ae8990b874f772f09e32ab9936ad0abcf12b9f89e02f414104c08047b9"
EXPECTED_MODULE_SHA256 = "97f66c1d23b299c04f407250dc94de595e06b645f32034178b2982c0512d36e4"
REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_row(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(value.shape),
        "finite": bool(np.isfinite(value).all()),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        "rms": float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))),
        "max_abs": float(np.max(np.abs(value))),
    }


def reassemble3d(tag: str, stagger: str, nz: int, ranks: list[dict]) -> np.ndarray:
    ny = wrf.JDE if stagger == "v" else wrf.JDE - 1
    nx = wrf.IDE if stagger == "u" else wrf.IDE - 1
    out = np.full((nz, ny, nx), np.nan, dtype=np.float64)
    covered = np.zeros((ny, nx), dtype=bool)
    for meta in ranks:
        ims, ime, jms, jme, kms, kme = meta["ims_ime_jms_jme_kms_kme"]
        ips, ipe, jps, jpe, _kps, _kpe = meta["ips_ipe_jps_jpe_kps_kpe"]
        path = meta["dir"] / f"step000001_{tag}.f64"
        memory = wrf._read_dump(
            path, (jme - jms + 1, kme - kms + 1, ime - ims + 1)
        )
        if stagger == "u":
            gi0, gi1 = ips, ipe
            gj0, gj1 = jps, min(jpe, wrf.JDE - 1)
        elif stagger == "v":
            gi0, gi1 = ips, min(ipe, wrf.IDE - 1)
            gj0, gj1 = jps, jpe
        else:
            gi0, gi1 = ips, min(ipe, wrf.IDE - 1)
            gj0, gj1 = jps, min(jpe, wrf.JDE - 1)
        block = memory[
            gj0 - jms : gj1 - jms + 1,
            wrf.KDS - kms : wrf.KDS - kms + nz,
            gi0 - ims : gi1 - ims + 1,
        ].transpose(1, 0, 2)
        if covered[gj0 - 1 : gj1, gi0 - 1 : gi1].any():
            raise RuntimeError(f"double coverage: {tag}")
        out[:, gj0 - 1 : gj1, gi0 - 1 : gi1] = block
        covered[gj0 - 1 : gj1, gi0 - 1 : gi1] = True
    if not covered.all() or not np.isfinite(out).all():
        raise RuntimeError(f"coverage/nonfinite failure: {tag}")
    return out


def reassemble2d(tag: str, stagger: str, ranks: list[dict]) -> np.ndarray:
    ny = wrf.JDE if stagger == "v" else wrf.JDE - 1
    nx = wrf.IDE if stagger == "u" else wrf.IDE - 1
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    covered = np.zeros_like(out, dtype=bool)
    for meta in ranks:
        ims, ime, jms, jme = meta["ims_ime_jms_jme_kms_kme"][:4]
        ips, ipe, jps, jpe = meta["ips_ipe_jps_jpe_kps_kpe"][:4]
        path = meta["dir"] / f"step000001_{tag}.f64"
        memory = wrf._read_dump(path, (jme - jms + 1, ime - ims + 1))
        if stagger == "u":
            gi0, gi1 = ips, ipe
            gj0, gj1 = jps, min(jpe, wrf.JDE - 1)
        elif stagger == "v":
            gi0, gi1 = ips, min(ipe, wrf.IDE - 1)
            gj0, gj1 = jps, jpe
        else:
            gi0, gi1 = ips, min(ipe, wrf.IDE - 1)
            gj0, gj1 = jps, min(jpe, wrf.JDE - 1)
        block = memory[
            gj0 - jms : gj1 - jms + 1,
            gi0 - ims : gi1 - ims + 1,
        ]
        if covered[gj0 - 1 : gj1, gi0 - 1 : gi1].any():
            raise RuntimeError(f"double coverage: {tag}")
        out[gj0 - 1 : gj1, gi0 - 1 : gi1] = block
        covered[gj0 - 1 : gj1, gi0 - 1 : gi1] = True
    if not covered.all() or not np.isfinite(out).all():
        raise RuntimeError(f"coverage/nonfinite failure: {tag}")
    return out


def dpn_faces(pair_sum: np.ndarray, constants: dict[str, Any]) -> np.ndarray:
    nz = pair_sum.shape[0]
    value = np.zeros((nz + 1,) + pair_sum.shape[1:], dtype=np.float64)
    value[0] = 0.5 * (
        constants["cf1"] * pair_sum[0]
        + constants["cf2"] * pair_sum[1]
        + constants["cf3"] * pair_sum[2]
    )
    value[1:nz] = 0.5 * (
        constants["fnm"][1:, None, None] * pair_sum[1:]
        + constants["fnp"][1:, None, None] * pair_sum[:-1]
    )
    if constants["top_lid"]:
        value[nz] = 0.5 * (
            constants["cfn"] * pair_sum[-1]
            + constants["cfn1"] * pair_sum[-2]
        )
    return value


def pgf_components(fields: dict[str, np.ndarray], c: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    ph, alt, p, pb, al, php = (fields[name] for name in ("ph", "alt", "p", "pb", "al", "php"))
    mu, muu, muv = fields["mu"], fields["muu"], fields["muv"]
    c1h = c["c1h"][:, None, None]
    c2h = c["c2h"][:, None, None]
    rdnw = c["rdnw"][:, None, None]

    result = {
        term: {"u": np.zeros_like(fields["cqu"]), "v": np.zeros_like(fields["cqv"])}
        for term in ("geopotential", "pressure", "base_pressure", "nonhydrostatic")
    }

    # U owned faces 1:-1.
    mass_u = c1h * muu[None, :, 1:-1] + c2h
    scale_u = (
        -fields["cqu"][:, :, 1:-1]
        * (c["msfux"][:, 1:-1] / c["msfuy"][:, 1:-1])[None, :, :]
        * 0.5 * c["rdx"] * mass_u
    )
    result["geopotential"]["u"][:, :, 1:-1] = scale_u * (
        ph[1:, :, 1:] - ph[1:, :, :-1] + ph[:-1, :, 1:] - ph[:-1, :, :-1]
    )
    result["pressure"]["u"][:, :, 1:-1] = scale_u * (
        (alt[:, :, 1:] + alt[:, :, :-1]) * (p[:, :, 1:] - p[:, :, :-1])
    )
    result["base_pressure"]["u"][:, :, 1:-1] = scale_u * (
        (al[:, :, 1:] + al[:, :, :-1]) * (pb[:, :, 1:] - pb[:, :, :-1])
    )
    dpn_u = dpn_faces(p[:, :, 1:] + p[:, :, :-1], c)
    result["nonhydrostatic"]["u"][:, :, 1:-1] = (
        -fields["cqu"][:, :, 1:-1]
        * (c["msfux"][:, 1:-1] / c["msfuy"][:, 1:-1])[None, :, :]
        * c["rdx"]
        * (php[:, :, 1:] - php[:, :, :-1])
        * (
            rdnw * (dpn_u[1:] - dpn_u[:-1])
            - 0.5 * c1h * (mu[None, :, 1:] + mu[None, :, :-1])
        )
    )

    # V owned faces 1:-1.
    mass_v = c1h * muv[None, 1:-1, :] + c2h
    scale_v = (
        -fields["cqv"][:, 1:-1, :]
        * (c["msfvy"][1:-1, :] / c["msfvx"][1:-1, :])[None, :, :]
        * 0.5 * c["rdy"] * mass_v
    )
    result["geopotential"]["v"][:, 1:-1, :] = scale_v * (
        ph[1:, 1:, :] - ph[1:, :-1, :] + ph[:-1, 1:, :] - ph[:-1, :-1, :]
    )
    result["pressure"]["v"][:, 1:-1, :] = scale_v * (
        (alt[:, 1:, :] + alt[:, :-1, :]) * (p[:, 1:, :] - p[:, :-1, :])
    )
    result["base_pressure"]["v"][:, 1:-1, :] = scale_v * (
        (al[:, 1:, :] + al[:, :-1, :]) * (pb[:, 1:, :] - pb[:, :-1, :])
    )
    dpn_v = dpn_faces(p[:, 1:, :] + p[:, :-1, :], c)
    result["nonhydrostatic"]["v"][:, 1:-1, :] = (
        -fields["cqv"][:, 1:-1, :]
        * (c["msfvy"][1:-1, :] / c["msfvx"][1:-1, :])[None, :, :]
        * c["rdy"]
        * (php[:, 1:, :] - php[:, :-1, :])
        * (
            rdnw * (dpn_v[1:] - dpn_v[:-1])
            - 0.5 * c1h * (mu[None, 1:, :] + mu[None, :-1, :])
        )
    )
    return result


def add_components(components: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: sum((row[name] for row in components.values()), start=np.zeros_like(next(iter(components.values()))[name]))
        for name in ("u", "v")
    }


def mask_for(value: np.ndarray, lo: int = 1, hi: int = 999) -> np.ndarray:
    return (parent.cpu.distance_to_edge(value.shape) >= lo) & (
        parent.cpu.distance_to_edge(value.shape) <= hi
    )


def rmse(fields: dict[str, np.ndarray], lo: int = 1, hi: int = 999) -> float:
    sse = count = 0
    for name in ("u", "v"):
        mask = mask_for(fields[name], lo, hi)
        sse += float(np.sum(np.square(fields[name][mask], dtype=np.float64)))
        count += int(mask.sum())
    return math.sqrt(sse / count)


def score(target: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, float]:
    before = rmse(target)
    after = rmse({name: target[name] - prediction[name] for name in ("u", "v")})
    return {
        "rmse_target": before,
        "rmse_prediction": rmse(prediction),
        "rmse_target_minus_prediction": after,
        "explained_sse": 1.0 - (after * after) / (before * before),
    }


def main() -> int:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual!r}")
    parent_proof = json.loads(parent.OUTPUT.read_text())
    if parent_proof.get("proof_sha256") != PARENT_CANONICAL:
        raise RuntimeError("parent ledger identity changed")
    if sha256_file(parent.PREDICTIONS_OUTPUT) != PREDICTIONS_SHA256:
        raise RuntimeError("sealed prediction archive changed")
    if sha256_file(BINARY) != EXPECTED_BINARY_SHA256 or sha256_file(MODULE_EM) != EXPECTED_MODULE_SHA256:
        raise RuntimeError("fresh PGF diagnostic source/binary changed")
    if "SUCCESS COMPLETE WRF" not in (RUN / "rsl.error.0000").read_text(errors="replace"):
        raise RuntimeError("fresh PGF-input WRF did not complete")

    ranks = wrf.load_ranks(DUMPS)
    wrf_fields = {
        name: reassemble3d(f"op_pgf_in__{name}", "mass", 45 if name == "ph" else 44, ranks)
        for name in ("ph", "alt", "p", "pb", "al", "php")
    }
    wrf_fields.update({
        "cqu": reassemble3d("op_pgf_in__cqu", "u", 44, ranks),
        "cqv": reassemble3d("op_pgf_in__cqv", "v", 44, ranks),
        "mu": reassemble2d("op_pgf_in__mu", "mass", ranks),
        "muu": reassemble2d("op_pgf_in__muu", "u", ranks),
        "muv": reassemble2d("op_pgf_in__muv", "v", ranks),
    })

    for lane in ("adv", "pgf"):
        wrf.FIELD_STAGGER[f"op_{lane}__ru_tend"] = "u"
        wrf.FIELD_STAGGER[f"op_{lane}__rv_tend"] = "v"
    wrf_lane = {
        name: wrf.reassemble3d(f"op_pgf__r{name}_tend", 1, ranks)
        - wrf.reassemble3d(f"op_adv__r{name}_tend", 1, ranks)
        for name in ("u", "v")
    }

    with Dataset(WRFINPUT) as dataset:
        one = lambda name: np.asarray(dataset.variables[name][0], dtype=np.float64)
        constants = {
            "c1h": one("C1H").reshape(-1), "c2h": one("C2H").reshape(-1),
            "fnm": one("FNM").reshape(-1), "fnp": one("FNP").reshape(-1),
            "rdnw": one("RDNW").reshape(-1),
            "cf1": float(one("CF1")), "cf2": float(one("CF2")), "cf3": float(one("CF3")),
            "cfn": float(one("CFN")), "cfn1": float(one("CFN1")),
            "msfux": one("MAPFAC_UX"), "msfuy": one("MAPFAC_UY"),
            "msfvx": one("MAPFAC_VX"), "msfvy": one("MAPFAC_VY"),
            "rdx": 0.001, "rdy": 0.001, "top_lid": True,
        }

    import jax
    import gpuwrf.contracts.state as state_contract
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime
    from gpuwrf.dynamics.acoustic_wrf import moisture_coupling_factors
    from gpuwrf.dynamics.core.rk_addtend_dry import _absolute_diagnostics
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    scratch = Path(tempfile.mkdtemp(prefix="v0234-s1-pgf-components-"))
    try:
        tree, *_ = ordinary.load_corrected_tree(scratch)
        with parent.STEP0.open("rb") as stream:
            carry = pickle.load(stream)
        namelist = tree.domains["d03"].namelist
        state = runtime.apply_halo(carry.state, runtime.halo_spec(namelist.grid))
        ph, p, al, alt, php = _absolute_diagnostics(
            state, namelist.metrics,
            hypsometric_opt=int(namelist.hypsometric_opt), base_state=carry.base_state,
        )
        cqu, cqv = moisture_coupling_factors(state)
        mu = np.asarray(state.mu_perturbation, dtype=np.float64)
        mut = np.asarray(state.mu_total, dtype=np.float64)
        gpu_fields = {
            "ph": np.asarray(ph), "alt": np.asarray(alt), "p": np.asarray(p),
            "pb": np.asarray(state.p_total - state.p_perturbation),
            "al": np.asarray(al), "php": np.asarray(php),
            "cqu": np.asarray(cqu), "cqv": np.asarray(cqv), "mu": mu,
            "muu": 0.5 * (np.pad(mut, ((0, 0), (1, 1)), mode="edge")[:, :-1]
                          + np.pad(mut, ((0, 0), (1, 1)), mode="edge")[:, 1:]),
            "muv": 0.5 * (np.pad(mut, ((1, 1), (0, 0)), mode="edge")[:-1, :]
                          + np.pad(mut, ((1, 1), (0, 0)), mode="edge")[1:, :]),
        }
        runtime_lane_device = runtime.large_step_horizontal_pgf(
            state, namelist.metrics,
            dx_m=float(namelist.grid.projection.dx_m),
            dy_m=float(namelist.grid.projection.dy_m),
            non_hydrostatic=True, top_lid=bool(namelist.top_lid),
            hypsometric_opt=int(namelist.hypsometric_opt), base_state=carry.base_state,
        )
        runtime_lane = {"u": np.asarray(runtime_lane_device[0]), "v": np.asarray(runtime_lane_device[1])}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    wrf_components = pgf_components(wrf_fields, constants)
    gpu_components = pgf_components(gpu_fields, constants)
    wrf_reconstructed = add_components(wrf_components)
    gpu_reconstructed = add_components(gpu_components)
    source_prediction = {
        name: gpu_reconstructed[name] - wrf_reconstructed[name] for name in ("u", "v")
    }
    with np.load(parent.PREDICTIONS_OUTPUT) as archive:
        sealed_prediction = {name: np.asarray(archive[f"pgf_{name}"]) for name in ("u", "v")}
    component_deltas = {
        term: {name: gpu_components[term][name] - wrf_components[term][name] for name in ("u", "v")}
        for term in wrf_components
    }

    raw_rows = []
    for path in sorted(item for item in DUMPS.rglob("*") if item.is_file()):
        raw_rows.append({
            "relative_path": str(path.relative_to(DUMPS)),
            "bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
        })
    proof = {
        "schema": "gpuwrf.v0234.s1-pgf-component-ledger.v1",
        "authority": {
            "parent_operator_ledger_canonical": PARENT_CANONICAL,
            "sealed_predictions_sha256": PREDICTIONS_SHA256,
            "step0_sha256": parent.STEP0_SHA256,
            "wrfinput": {"path": str(WRFINPUT), "file_sha256": sha256_file(WRFINPUT)},
        },
        "fresh_wrf": {
            "binary_sha256": EXPECTED_BINARY_SHA256,
            "module_em_sha256": EXPECTED_MODULE_SHA256,
            "success_complete_wrf": True,
            "observed_live_rank_affinity": {"ranks": 12, "cpus_allowed_list": "13-15,29-31"},
            "raw_manifest": {"root": str(DUMPS), "files": len(raw_rows), "rows_sha256": canonical(raw_rows), "rows": raw_rows},
        },
        "reconstruction": {
            "wrf_component_sum_minus_dumped_lane_rmse": rmse({name: wrf_reconstructed[name] - wrf_lane[name] for name in ("u", "v")}),
            "gpu_numpy_sum_minus_runtime_lane_rmse": rmse({name: gpu_reconstructed[name] - runtime_lane[name] for name in ("u", "v")}),
            "component_sum_prediction_minus_sealed_prediction_rmse": rmse({name: source_prediction[name] - sealed_prediction[name] for name in ("u", "v")}),
        },
        "input_deltas_gpu_minus_wrf": {
            name: array_row(gpu_fields[name] - wrf_fields[name]) for name in wrf_fields
        },
        "component_deltas_gpu_minus_wrf": {
            term: {"arrays": {name: array_row(value) for name, value in fields.items()}, "score_against_sealed_pgf_prediction": score(sealed_prediction, fields)}
            for term, fields in component_deltas.items()
        },
        "component_sum_score_against_sealed_pgf_prediction": score(sealed_prediction, source_prediction),
        "negative_results": {
            "eos_pressure_reinversion_u_v_rmse": [42.07720548546835, 47.52525724273365],
            "calc_alt_only_candidate_zone_rmse": {
                "nonspec": 0.795649296743509,
                "relax_rows_1_4": 0.7826326785388102,
                "interior_ge_5": 0.7979978379588765,
                "ring_1": 0.7806499253217077,
            },
        },
        "resource_attestation": {
            "gpu_queries": 0, "gpu_locks": 0, "gpu_compiles": 0, "gpu_dispatches": 0,
            "cpu_only": True, "historical_namespaces_mutated": 0,
        },
    }
    proof["proof_sha256"] = canonical(proof)
    OUTPUT.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "proof_sha256": proof["proof_sha256"],
        "wrf_reconstruction_rmse": proof["reconstruction"]["wrf_component_sum_minus_dumped_lane_rmse"],
        "prediction_reconstruction_rmse": proof["reconstruction"]["component_sum_prediction_minus_sealed_prediction_rmse"],
        "component_scores": {term: row["score_against_sealed_pgf_prediction"]["explained_sse"] for term, row in proof["component_deltas_gpu_minus_wrf"].items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
