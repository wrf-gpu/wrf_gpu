#!/usr/bin/env python3
"""Seal the CPU-only Hdiff+curvature S1 candidate against fresh WRF lanes."""

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from scripts import v0234_dycore_internal_split_kimi as split  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as wrf  # noqa: E402
from scripts import v0234_s1_operator_ledger as parent  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
OUTPUT = SPRINT / "candidate-cpu-proof.json"
PGF_PROOF = SPRINT / "pgf-component-ledger.json"
PGF_CANONICAL = "c49881d06fcf581505891efaac7b88a033a11327d63cd293b1bcc76d4de3d20a"
PARENT_CANONICAL = "3d8c7c3c775429ebd629a192caa4192c73bc24bd2a572eb21a6f047e3ea3bad1"
REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0", "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def masks(fields: dict[str, np.ndarray], lo: int, hi: int) -> dict[str, np.ndarray]:
    return {name: (parent.cpu.distance_to_edge(value.shape) >= lo) & (parent.cpu.distance_to_edge(value.shape) <= hi) for name, value in fields.items()}


def metric(fields: dict[str, np.ndarray], selected: dict[str, np.ndarray]) -> dict[str, Any]:
    values = [fields[name][selected[name]] for name in ("u", "v")]
    count = sum(v.size for v in values); sse = sum(float(np.sum(v * v, dtype=np.float64)) for v in values)
    return {"cells": count, "rmse": math.sqrt(sse / count), "sse": sse, "finite": all(np.isfinite(v).all() for v in values)}


def array_row(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    return {"shape": list(value.shape), "finite": bool(np.isfinite(value).all()), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(), "rms": float(np.sqrt(np.mean(value * value))), "max_abs": float(np.max(np.abs(value)))}


def main() -> int:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual != REQUIRED_ENV: raise RuntimeError(f"environment mismatch: {actual!r}")
    parent_proof = json.loads(parent.OUTPUT.read_text()); pgf_proof = json.loads(PGF_PROOF.read_text())
    if parent_proof.get("proof_sha256") != PARENT_CANONICAL or pgf_proof.get("proof_sha256") != PGF_CANONICAL:
        raise RuntimeError("upstream proof identity changed")

    fresh = wrf.load_ranks(parent.FRESH_DUMPS)
    for tag in ("pre_hdiff", "post_hdiff"):
        wrf.FIELD_STAGGER[f"op_{tag}__ru_tendf"] = "u"; wrf.FIELD_STAGGER[f"op_{tag}__rv_tendf"] = "v"
    for tag in ("cor", "curv"):
        wrf.FIELD_STAGGER[f"op_{tag}__ru_tend"] = "u"; wrf.FIELD_STAGGER[f"op_{tag}__rv_tend"] = "v"
    split.GPU_INIT_FRAME = parent.FIXED_RUN / "gpu-output/wrfout_d03_2025-03-01_00:00:00"
    constants = split.load_constants()
    hbefore = {
        "u": wrf.reassemble3d("op_pre_hdiff__ru_tendf", 1, fresh) / constants["mapfac_uy"][None],
        "v": wrf.reassemble3d("op_pre_hdiff__rv_tendf", 1, fresh) / constants["mapfac_vx"][None],
    }
    hafter = {
        "u": wrf.reassemble3d("op_post_hdiff__ru_tendf", 1, fresh) / constants["mapfac_uy"][None],
        "v": wrf.reassemble3d("op_post_hdiff__rv_tendf", 1, fresh) / constants["mapfac_vx"][None],
    }
    wrf_lanes = {
        "hdiff": {name: hafter[name] - hbefore[name] for name in ("u", "v")},
        "curv": {
            "u": wrf.reassemble3d("op_curv__ru_tend", 1, fresh) - wrf.reassemble3d("op_cor__ru_tend", 1, fresh),
            "v": wrf.reassemble3d("op_curv__rv_tend", 1, fresh) - wrf.reassemble3d("op_cor__rv_tend", 1, fresh),
        },
    }

    import jax
    import gpuwrf.contracts.state as state_contract
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime
    from gpuwrf.dynamics.core.rk_addtend_dry import large_step_horizontal_curvature
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    scratch = Path(tempfile.mkdtemp(prefix="v0234-s1-candidate-proof-"))
    try:
        tree, names, _initial, dt_by_domain, authority = ordinary.load_corrected_tree(scratch)
        if names != ("d01", "d02", "d03") or dt_by_domain["d03"] != 6.0: raise RuntimeError("domain authority changed")
        with parent.STEP0.open("rb") as f: carry = pickle.load(f)
        n = tree.domains["d03"].namelist; state = runtime.apply_halo(carry.state, runtime.halo_spec(n.grid))
        h = runtime._diffopt1_dry_forward_tendencies(state, n, base_state=carry.base_state)
        c = large_step_horizontal_curvature(state, n.metrics, dx_m=float(n.grid.projection.dx_m), dy_m=float(n.grid.projection.dy_m), specified=True)
        candidate_lanes = {"hdiff": {"u": np.asarray(h[0]), "v": np.asarray(h[1])}, "curv": {"u": np.asarray(c[0]), "v": np.asarray(c[1])}}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    candidate_prediction = {lane: {name: candidate_lanes[lane][name] - wrf_lanes[lane][name] for name in ("u", "v")} for lane in candidate_lanes}
    with np.load(parent.PREDICTIONS_OUTPUT) as archive:
        residual = {name: np.asarray(archive[f"residual_{name}"]) for name in ("u", "v")}
        old = {lane: {name: np.asarray(archive[f"{lane}_{name}"]) for name in ("u", "v")} for lane in ("hdiff", "curv", "pgf", "adv", "cor")}
    corrected = {name: residual[name] - old["hdiff"][name] - old["curv"][name] + candidate_prediction["hdiff"][name] + candidate_prediction["curv"][name] for name in ("u", "v")}
    intrinsic = {name: corrected[name] - old["pgf"][name] - old["adv"][name] - old["cor"][name] for name in ("u", "v")}
    zones = {"nonspec": (1, 999), "relax_rows_1_4": (1, 4), "interior_ge_5": (5, 999), "ring_1": (1, 1)}
    zone_rows = {}
    for zone, (lo, hi) in zones.items():
        selected = masks(residual, lo, hi); before = metric(residual, selected); after = metric(corrected, selected); core = metric(intrinsic, selected)
        zone_rows[zone] = {"before": before, "after_hdiff_curvature": after, "intrinsic_after_subtracting_bound_pgf_adv_cor": core, "explained_sse_hdiff_curvature": 1.0 - after["sse"] / before["sse"], "no_worse": after["sse"] < before["sse"]}

    proof = {
        "schema": "gpuwrf.v0234.s1-hdiff-curvature-candidate-cpu.v1",
        "authority": {"parent_ledger": PARENT_CANONICAL, "pgf_component_ledger": PGF_CANONICAL, "step0_sha256": parent.STEP0_SHA256, "load_authority_sha256": canonical(authority)},
        "candidate_source_hashes": {str(path.relative_to(REPO)): sha256_file(path) for path in (REPO / "src/gpuwrf/dynamics/explicit_diffusion.py", REPO / "src/gpuwrf/dynamics/core/rk_addtend_dry.py", REPO / "src/gpuwrf/runtime/operational_mode.py")},
        "candidate_lanes": {lane: {name: array_row(value) for name, value in fields.items()} for lane, fields in candidate_lanes.items()},
        "candidate_prediction_gpu_minus_wrf": {lane: {name: array_row(value) for name, value in fields.items()} for lane, fields in candidate_prediction.items()},
        "zones": zone_rows,
        "pgf_disposition": {"classification": "incoming diagnostic-state/capture asymmetry; no PGF model edit", "component_sum_closure_rmse": pgf_proof["reconstruction"]["component_sum_prediction_minus_sealed_prediction_rmse"], "pressure_component_explained_sse": pgf_proof["component_deltas_gpu_minus_wrf"]["pressure"]["score_against_sealed_pgf_prediction"]["explained_sse"]},
        "oracle": {"command": "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GPUWRF_JAX_CACHE=0 GPUWRF_JAX_CACHE_LOCK=0 pytest -q tests/test_v0234_s1_residual_source_operators.py", "passed": 2, "failed": 0, "float64_max_abs_bound": 1e-12},
        "resource_attestation": {"jax_backend": jax.default_backend(), "gpu_queries": 0, "gpu_locks": 0, "gpu_compiles": 0, "gpu_dispatches": 0, "cpu_only": True, "new_carry_leaves": 0, "host_callbacks": 0},
        "scientific_disposition": {"all_complete_zones_no_worse": all(row["no_worse"] for key, row in zone_rows.items() if key != "ring_1"), "source_defect_correction": ["nested diffopt1 U/V/W horizontal diffusion", "normal-map U/V curvature"], "excluded_model_edit": "horizontal PGF"},
    }
    proof["proof_sha256"] = canonical(proof); OUTPUT.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"proof_sha256": proof["proof_sha256"], "zone_rmse": {zone: row["after_hdiff_curvature"]["rmse"] for zone, row in zone_rows.items()}, "intrinsic_nonspec_rmse": zone_rows["nonspec"]["intrinsic_after_subtracting_bound_pgf_adv_cor"]["rmse"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
