#!/usr/bin/env python3
"""Seal the CPU-only authentic d03 step-1 momentum operator ledger.

The result is deliberately source ordered.  It reassembles the independent
WRF rank dumps, reconstructs the frozen post-diff6 S1 residual, evaluates the
current production GPUWRF operators on CPU from the authenticated step-0
carry, and scores unscaled ``P = GPU_lane - WRF_lane`` predictions.

No candidate model is evaluated here and no GPU API, lock, query, compile, or
dispatch is permitted.
"""

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
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts import v0234_dycore_internal_split_kimi as split  # noqa: E402
from scripts import v0234_dycore_suboperator_gpt_cpu_analysis as cpu  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as wrf  # noqa: E402
from scripts import v0234_s1_dyn_attribution_fable5 as s1  # noqa: E402

SCHEMA = "gpuwrf.v0234.s1-authentic-operator-ledger.v1"
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
OUTPUT = SPRINT / "operator-ledger-proof.json"
FIXED_RUN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
)
FIXED_TERMINAL = FIXED_RUN / "s1-diff6-validation-terminal-proof.json"
CONTROL_DUMPS = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs/control/momsp_dumps"
)
LEDGER = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_s1_residual_closure_gpt/operator_ledger1"
)
PREDICTIONS_OUTPUT = LEDGER / "current-source-predictions-step1.npz"
FRESH_DUMPS = LEDGER / "dumps_step_fixed"
FRESH_BINARY = LEDGER / "install_operator/bin/wrf"
FRESH_NAMELIST = LEDGER / "run_operator/namelist.input"
FRESH_RSL = LEDGER / "run_operator/rsl.error.0000"
FRESH_MODULE_EM = LEDGER / "dyn_em/module_em.F"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_1500_science_60659a2e_terminal2/"
    "science-runtime/failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
TERMINAL_SELF_SHA256 = "6e8ed620932459a6d663feafb8a64cdd3bf5cbd3aaa65a6e878dcdb1e70c677e"
FRESH_ROWS_SHA256 = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"
EXPECTED_METRICS = {
    "nonspec": 3.22385592467145,
    "relax_rows_1_4": 7.765706866136884,
    "interior_ge_5": 1.1401580540399108,
}
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


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def authenticate_json(path: Path, expected: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    observed = canonical({key: value for key, value in payload.items() if key != "proof_sha256"})
    if observed != payload.get("proof_sha256") or observed != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    return payload


def array_row(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": sha256_array(value),
        "rms": float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))),
        "max_abs": float(np.max(np.abs(value))),
    }


def zone_masks(fields: Mapping[str, np.ndarray], lo: int, hi: int) -> dict[str, np.ndarray]:
    return {
        name: (cpu.distance_to_edge(value.shape) >= lo)
        & (cpu.distance_to_edge(value.shape) <= hi)
        for name, value in fields.items()
    }


def score(
    residual: Mapping[str, np.ndarray],
    prediction: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    sse_r = sum(
        float(np.sum(np.square(residual[name][masks[name]], dtype=np.float64), dtype=np.float64))
        for name in ("u", "v")
    )
    after = {name: residual[name] - prediction[name] for name in ("u", "v")}
    sse_after = sum(
        float(np.sum(np.square(after[name][masks[name]], dtype=np.float64), dtype=np.float64))
        for name in ("u", "v")
    )
    sse_p = sum(
        float(np.sum(np.square(prediction[name][masks[name]], dtype=np.float64), dtype=np.float64))
        for name in ("u", "v")
    )
    dot = sum(
        float(np.sum(residual[name][masks[name]] * prediction[name][masks[name]], dtype=np.float64))
        for name in ("u", "v")
    )
    cells = sum(int(np.count_nonzero(masks[name])) for name in ("u", "v"))
    return {
        "cells": cells,
        "rmse_R": math.sqrt(sse_r / cells),
        "rmse_P": math.sqrt(sse_p / cells),
        "rmse_R_minus_P": math.sqrt(sse_after / cells),
        "explained_sse": 1.0 - sse_after / sse_r,
        "signed_correlation": dot / math.sqrt(sse_r * sse_p) if sse_p else None,
    }


def component_score(
    residual: np.ndarray, prediction: np.ndarray, mask: np.ndarray,
) -> dict[str, Any]:
    r = residual[mask]
    p = prediction[mask]
    sse_r = float(np.sum(np.square(r, dtype=np.float64), dtype=np.float64))
    sse_p = float(np.sum(np.square(p, dtype=np.float64), dtype=np.float64))
    sse_after = float(np.sum(np.square(r - p, dtype=np.float64), dtype=np.float64))
    cells = int(r.size)
    dot = float(np.sum(r * p, dtype=np.float64))
    return {
        "cells": cells,
        "rmse_R": math.sqrt(sse_r / cells),
        "rmse_P": math.sqrt(sse_p / cells),
        "rmse_R_minus_P": math.sqrt(sse_after / cells),
        "explained_sse": 1.0 - sse_after / sse_r,
        "signed_correlation": dot / math.sqrt(sse_r * sse_p) if sse_p else None,
    }


def selected_rmse(fields: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray]) -> float:
    sse = sum(
        float(np.sum(np.square(fields[name][masks[name]], dtype=np.float64), dtype=np.float64))
        for name in ("u", "v")
    )
    cells = sum(int(np.count_nonzero(masks[name])) for name in ("u", "v"))
    return math.sqrt(sse / cells)


def raw_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
            }
        )
    return {"root": str(root), "files": len(rows), "rows_sha256": canonical(rows), "rows": rows}


def load_frozen_residual() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    terminal = authenticate_json(FIXED_TERMINAL, TERMINAL_SELF_SHA256)
    rows = terminal["savepoint_manifest"]["rows"]
    if len(rows) != 180 or canonical(rows) != FRESH_ROWS_SHA256:
        raise RuntimeError("fixed GPU savepoint manifest changed")
    step_rows = {(row["tag"], row["field"]): row for row in rows if row["step"] == 1}
    for row in step_rows.values():
        path = Path(row["path"])
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"fixed GPU savepoint drift: {path}")

    def load(tag: str, field: str) -> np.ndarray:
        return np.load(step_rows[(tag, field)]["path"], allow_pickle=False)

    gpu = {
        "ru_tend": load("l1_rk1_tend", "ru_tend"),
        "rv_tend": load("l1_rk1_tend", "rv_tend"),
        "ru_tendf": load("sp3_tendf", "ru_tendf"),
        "rv_tendf": load("sp3_tendf", "rv_tendf"),
        "u_save": load("l1_rk1_relax", "u_save"),
        "v_save": load("l1_rk1_relax", "v_save"),
        "u_sp1": load("sp1_entry", "u"),
        "v_sp1": load("sp1_entry", "v"),
    }
    old_ranks = wrf.load_ranks(CONTROL_DUMPS)
    control = {
        "ru_tend": wrf.reassemble3d("l1_rk1_tend__ru_tend", 1, old_ranks),
        "rv_tend": wrf.reassemble3d("l1_rk1_tend__rv_tend", 1, old_ranks),
        "ru_tendf": wrf.reassemble3d("sp3_tendf__ru_tendf", 1, old_ranks),
        "rv_tendf": wrf.reassemble3d("sp3_tendf__rv_tendf", 1, old_ranks),
        "u_save": wrf.reassemble3d("l1_rk1_tend__u_save", 1, old_ranks),
        "v_save": wrf.reassemble3d("l1_rk1_tend__v_save", 1, old_ranks),
    }
    split.GPU_INIT_FRAME = FIXED_RUN / "gpu-output/wrfout_d03_2025-03-01_00:00:00"
    constants = split.load_constants()
    residual = s1.s1_residual(gpu, control, constants)
    reproduced = {
        "nonspec": split.combined_rmse(residual, zone_masks(residual, 1, 999)),
        "relax_rows_1_4": split.combined_rmse(residual, zone_masks(residual, 1, 4)),
        "interior_ge_5": split.combined_rmse(residual, zone_masks(residual, 5, 999)),
    }
    if reproduced != EXPECTED_METRICS:
        raise RuntimeError(f"frozen residual changed: {reproduced!r}")
    return gpu, control, residual, {"constants": constants, "metrics": reproduced}


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    if "jax" in sys.modules or "gpuwrf" in sys.modules:
        raise RuntimeError("JAX/gpuwrf imported before CPU environment validation")
    if sha256_file(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated step-0 carry changed")

    gpu, control, residual, frozen = load_frozen_residual()
    constants = frozen.pop("constants")

    fresh_ranks = wrf.load_ranks(FRESH_DUMPS)
    old_ranks = wrf.load_ranks(CONTROL_DUMPS)
    unchanged_fields = [
        "sp1_entry__u", "sp1_entry__v",
        "sp3_tendf__ru_tendf", "sp3_tendf__rv_tendf",
        "l1_rk1_tend__ru_tend", "l1_rk1_tend__rv_tend",
        "l1_rk1_tend__u_save", "l1_rk1_tend__v_save",
        "l2_rk1_fin__u", "l2_rk1_fin__v",
        "l3_rk2_fin__u", "l3_rk2_fin__v",
        "l4_rk3_fin__u", "l4_rk3_fin__v",
        "l5_prebdry__u", "l5_prebdry__v",
    ]
    neutrality = []
    for field in unchanged_fields:
        fresh = wrf.reassemble3d(field, 1, fresh_ranks)
        historical = wrf.reassemble3d(field, 1, old_ranks)
        neutrality.append(
            {
                "field": field,
                "bitwise_equal": bool(np.array_equal(fresh, historical)),
                "array_sha256": sha256_array(fresh),
            }
        )
    if not all(row["bitwise_equal"] for row in neutrality):
        raise RuntimeError("fresh WRF instrumentation changed a historical ladder array")

    for lane in ("adv", "pgf", "cor", "curv"):
        wrf.FIELD_STAGGER[f"op_{lane}__ru_tend"] = "u"
        wrf.FIELD_STAGGER[f"op_{lane}__rv_tend"] = "v"
    for lane in ("pre_hdiff", "post_hdiff", "post_diff6"):
        wrf.FIELD_STAGGER[f"op_{lane}__ru_tendf"] = "u"
        wrf.FIELD_STAGGER[f"op_{lane}__rv_tendf"] = "v"

    cumulative = {
        lane: {
            "u": wrf.reassemble3d(f"op_{lane}__ru_tend", 1, fresh_ranks),
            "v": wrf.reassemble3d(f"op_{lane}__rv_tend", 1, fresh_ranks),
        }
        for lane in ("adv", "pgf", "cor", "curv")
    }
    wrf_lanes = {
        "adv": cumulative["adv"],
        "pgf": {name: cumulative["pgf"][name] - cumulative["adv"][name] for name in ("u", "v")},
        "cor": {name: cumulative["cor"][name] - cumulative["pgf"][name] for name in ("u", "v")},
        "curv": {name: cumulative["curv"][name] - cumulative["cor"][name] for name in ("u", "v")},
    }
    tendf = {}
    for tag in ("pre_hdiff", "post_hdiff", "post_diff6"):
        tendf[tag] = {
            "u": wrf.reassemble3d(f"op_{tag}__ru_tendf", 1, fresh_ranks)
            / constants["mapfac_uy"][None, :, :],
            "v": wrf.reassemble3d(f"op_{tag}__rv_tendf", 1, fresh_ranks)
            / constants["mapfac_vx"][None, :, :],
        }
    wrf_lanes["hdiff"] = {
        name: tendf["post_hdiff"][name] - tendf["pre_hdiff"][name] for name in ("u", "v")
    }
    wrf_lanes["diff6"] = {
        name: tendf["post_diff6"][name] - tendf["post_hdiff"][name] for name in ("u", "v")
    }

    # Independent WRF source-ledger reconstruction of the retained S1 algebra.
    wrf_dyn = {
        "u": control["ru_tend"] - control["ru_tendf"] / constants["mapfac_uy"][None] - control["u_save"],
        "v": control["rv_tend"] - control["rv_tendf"] / constants["mapfac_vx"][None] - control["v_save"],
    }
    wrf_reconstructed = {
        name: cumulative["curv"][name] + wrf_lanes["hdiff"][name] + wrf_lanes["diff6"][name]
        for name in ("u", "v")
    }
    wrf_reconstruction_error = {
        name: wrf_dyn[name] - wrf_reconstructed[name] for name in ("u", "v")
    }

    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected JAX backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    scratch = Path(tempfile.mkdtemp(prefix="v0234-s1-operator-ledger-"))
    try:
        tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(scratch)
        if names != ("d01", "d02", "d03") or dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
            raise RuntimeError("canonical domain tree changed")
        with STEP0.open("rb") as stream:
            carry = pickle.load(stream)
        namelist = tree.domains["d03"].namelist
        origin = runtime.apply_halo(carry.state, runtime.halo_spec(namelist.grid))
        metrics = namelist.metrics
        dx = float(namelist.grid.projection.dx_m)
        dy = float(namelist.grid.projection.dy_m)
        muu = runtime._u_face_average_2d(origin.mu_total)
        muv = runtime._v_face_average_2d(origin.mu_total)
        mass_u = metrics.c1h[:, None, None] * muu[None] + metrics.c2h[:, None, None]
        mass_v = metrics.c1h[:, None, None] * muv[None] + metrics.c2h[:, None, None]
        velocities = runtime._stage_transport_velocities(origin, namelist)
        gpu_adv = {
            "u": namelist.tendencies.u * mass_u + runtime.advect_u_flux(
                origin.u, velocities, rdx=1.0 / dx, rdy=1.0 / dy,
                rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
            ),
            "v": namelist.tendencies.v * mass_v + runtime.advect_v_flux(
                origin.v, velocities, rdx=1.0 / dx, rdy=1.0 / dy,
                rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
            ),
        }
        pgf = runtime.large_step_horizontal_pgf(
            origin, metrics, dx_m=dx, dy_m=dy, non_hydrostatic=True,
            top_lid=bool(namelist.top_lid), hypsometric_opt=int(namelist.hypsometric_opt),
            base_state=carry.base_state,
        )
        cor = runtime.large_step_coriolis(origin, metrics, specified=bool(namelist.run_boundary))
        hdiff = runtime._diffopt1_dry_forward_tendencies(
            origin, namelist, base_state=carry.base_state,
        )
        gpu_lanes_device = {
            "adv": gpu_adv,
            "pgf": {"u": pgf[0], "v": pgf[1]},
            "cor": {"u": cor[0], "v": cor[1]},
            "curv": {"u": np.zeros(origin.u.shape), "v": np.zeros(origin.v.shape)},
            "hdiff": {"u": hdiff[0], "v": hdiff[1]},
        }
        gpu_lanes = {
            lane: {
                name: np.asarray(jax.device_get(value), dtype=np.float64)
                for name, value in fields.items()
            }
            for lane, fields in gpu_lanes_device.items()
        }
        step0_identity = {
            "u_equals_fixed_gpu_sp1": bool(np.array_equal(np.asarray(carry.state.u), gpu["u_sp1"])),
            "v_equals_fixed_gpu_sp1": bool(np.array_equal(np.asarray(carry.state.v), gpu["v_sp1"])),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    predictions = {
        lane: {name: gpu_lanes[lane][name] - wrf_lanes[lane][name] for name in ("u", "v")}
        for lane in ("hdiff", "curv", "adv", "pgf", "cor")
    }
    prediction_arrays = {
        f"{lane}_{name}": np.asarray(value, dtype=np.float64)
        for lane, fields in predictions.items()
        for name, value in fields.items()
    }
    prediction_arrays.update(
        {f"residual_{name}": np.asarray(value, dtype=np.float64) for name, value in residual.items()}
    )
    np.savez_compressed(PREDICTIONS_OUTPUT, **prediction_arrays)
    zones = {
        "nonspec": zone_masks(residual, 1, 999),
        "relax_rows_1_4": zone_masks(residual, 1, 4),
        "interior_ge_5": zone_masks(residual, 5, 999),
        "ring_1": zone_masks(residual, 1, 1),
    }
    lane_scores = {
        lane: {zone: score(residual, prediction, masks) for zone, masks in zones.items()}
        for lane, prediction in predictions.items()
    }
    combinations = {}
    for lanes in (
        ("hdiff", "pgf"),
        ("hdiff", "pgf", "curv"),
        ("hdiff", "adv", "pgf", "cor", "curv"),
    ):
        prediction = {
            name: sum(predictions[lane][name] for lane in lanes) for name in ("u", "v")
        }
        combinations["+".join(lanes)] = {
            zone: score(residual, prediction, masks) for zone, masks in zones.items()
        }

    _, ny_u, nx_u = residual["u"].shape
    _, ny_v, nx_v = residual["v"].shape
    u_east = zones["ring_1"]["u"] & (
        np.broadcast_to(np.arange(nx_u)[None, None, ::-1], residual["u"].shape) == 1
    )
    v_north = zones["ring_1"]["v"] & (
        np.broadcast_to(np.arange(ny_v)[None, ::-1, None], residual["v"].shape) == 1
    )
    directional = {
        "hdiff_u_east_ring1": component_score(residual["u"], predictions["hdiff"]["u"], u_east),
        "hdiff_v_north_ring1": component_score(residual["v"], predictions["hdiff"]["v"], v_north),
    }

    manifest = raw_manifest(FRESH_DUMPS)
    if manifest["files"] != 456:
        raise RuntimeError(f"fresh raw dump inventory changed: {manifest['files']}")
    rsl_text = FRESH_RSL.read_text(errors="replace")
    source_hashes = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in (
            REPO / "src/gpuwrf/runtime/operational_mode.py",
            REPO / "src/gpuwrf/dynamics/explicit_diffusion.py",
            REPO / "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
        )
    }
    proof = {
        "schema": SCHEMA,
        "authority": {
            "step0": {"path": str(STEP0), "file_sha256": STEP0_SHA256},
            "fixed_terminal": {"path": str(FIXED_TERMINAL), "canonical_self_hash": TERMINAL_SELF_SHA256},
            "fixed_savepoint_rows": 180,
            "fixed_savepoint_rows_sha256": FRESH_ROWS_SHA256,
            "frozen_metrics": frozen["metrics"],
            "step0_identity": step0_identity,
            "load_authority_sha256": canonical(load_authority),
        },
        "fresh_wrf": {
            "binary": {"path": str(FRESH_BINARY), "file_sha256": sha256_file(FRESH_BINARY)},
            "namelist": {"path": str(FRESH_NAMELIST), "file_sha256": sha256_file(FRESH_NAMELIST)},
            "module_em": {"path": str(FRESH_MODULE_EM), "file_sha256": sha256_file(FRESH_MODULE_EM)},
            "success_complete_wrf": "SUCCESS COMPLETE WRF" in rsl_text,
            "raw_manifest": manifest,
            "instrumentation_neutrality": neutrality,
            "resource_command": (
                "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 13-15,29-31 "
                "mpirun --bind-to none --oversubscribe -np 12 <fresh-wrf>"
            ),
            "observed_live_rank_affinity": {
                "wrf_ranks": 12,
                "cpus_allowed_list_each": "13-15,29-31",
            },
        },
        "wrf_source_ledger": {
            "lanes": {
                lane: {name: array_row(value) for name, value in fields.items()}
                for lane, fields in wrf_lanes.items()
            },
            "reconstruction_error": {
                zone: selected_rmse(wrf_reconstruction_error, masks)
                for zone, masks in zones.items()
            },
        },
        "current_gpu_cpu_lanes": {
            "source_hashes": source_hashes,
            "jax_backend": jax.default_backend(),
            "lanes": {
                lane: {name: array_row(value) for name, value in fields.items()}
                for lane, fields in gpu_lanes.items()
            },
        },
        "sealed_current_predictions": {
            "path": str(PREDICTIONS_OUTPUT),
            "file_sha256": sha256_file(PREDICTIONS_OUTPUT),
            "arrays": {
                name: array_row(value) for name, value in prediction_arrays.items()
            },
        },
        "unscaled_prediction_scores": lane_scores,
        "source_ordered_combinations": combinations,
        "directional_scores": directional,
        "scientific_disposition": {
            "conditioning_claim_rejected": True,
            "h1_hdiff_ring1_source_attributed": lane_scores["hdiff"]["ring_1"]["explained_sse"] >= 0.95,
            "h1_hdiff_relax_source_attributed": lane_scores["hdiff"]["relax_rows_1_4"]["explained_sse"] >= 0.95,
            "h4_pgf_second_mechanism": lane_scores["pgf"]["interior_ge_5"]["explained_sse"] > 0.0,
            "h2_curvature_source_visible": lane_scores["curv"]["nonspec"]["explained_sse"] > 0.0,
            "complete_source_ordered_closure": combinations[
                "hdiff+adv+pgf+cor+curv"
            ]["nonspec"]["explained_sse"] > 0.999999,
            "production_edit_requires_committed_amendment": True,
        },
        "resource_attestation": {
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "cpu_jax_only": True,
            "historical_namespaces_mutated": 0,
        },
    }
    proof["proof_sha256"] = canonical(proof)
    OUTPUT.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "proof": str(OUTPUT),
        "proof_sha256": proof["proof_sha256"],
        "full_closure_nonspec_rmse": combinations["hdiff+adv+pgf+cor+curv"]["nonspec"]["rmse_R_minus_P"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
