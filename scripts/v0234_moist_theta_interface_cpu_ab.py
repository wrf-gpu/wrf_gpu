"""Two-step real-d03 CPU A/B for the complete moist-theta MP interface.

Step 1 produces settled ``h_diabatic`` after post-RK Thompson.  Step 2 is the
minimum dispatch that consumes that retained source through every RK stage and
the final-stage cancellation, so this arm tests the coupled mechanism omitted
by the already-falsified isolated Thompson correction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-17-v0234-moist-theta-interface"
OUT = SPRINT / "complete-interface-cpu-ab-proof.json"
FAILURE = SPRINT / "complete-interface-cpu-ab-blocker.json"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_scalar_sixth_order_18d97595_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
FRAME_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
FRAMES = {
    "current": (
        FRAME_ROOT
        / "corrected_ni_rca_max_22c2bd7a/"
        "nested_stage_omega_transport_470e6111_full18h_discriminator1/"
        "output/wrfout_d03_2025-03-01_00:20:00",
        "6113fbbe68531925c5c14ecd1de8897f93e3780d35d7bb664fb4a3c008bf6153",
    ),
    "cpu_wrf": (
        FRAME_ROOT / "run/wrf/wrfout_d03_2025-03-01_00:20:00",
        "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1",
    ),
    "retry20": (
        FRAME_ROOT
        / "gpu_validation_retry20_relative_rmse_3ee02c19/"
        "pair-snapshots/20250301T002000/gpu.nc",
        "70e09cf3ca22711c66ef529e716ead53fb998d2f548e9939c35d7767ac3f9e62",
    ),
}
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
    "GPUWRF_MOIST_THETA_INTERFACE": "1",
}
ALLOWED_CPU_CUSTOM_TARGETS = {"lapack_dgtsv_ffi"}
STATE_TO_HISTORY = {
    "theta": "THM",
    "qv": "QVAPOR",
    "qc": "QCLOUD",
    "qr": "QRAIN",
    "qi": "QICE",
    "qs": "QSNOW",
    "qg": "QGRAUP",
    "Ni": "QNICE",
    "Nr": "QNRAIN",
    "Ns": "QNSNOW",
    "Ng": "QNGRAUPEL",
    "p_perturbation": "P",
    "ph_perturbation": "PH",
    "u": "U",
    "v": "V",
    "w": "W",
    "rain_acc": "RAINNC",
    "snow_acc": "SNOWNC",
    "graupel_acc": "GRAUPELNC",
    "ice_acc": "I_RAINNC",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _path_name(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for key in path:
        for attr in ("name", "key", "idx"):
            if hasattr(key, attr):
                parts.append(str(getattr(key, attr)))
                break
        else:
            parts.append(str(key))
    return ".".join(parts)


def _manifest(jax, np, value: Any) -> dict[str, Any]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        array = np.asarray(leaf)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
        rows.append(
            {
                "path": _path_name(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": digest.hexdigest(),
                "finite": bool(
                    not np.issubdtype(array.dtype, np.floating)
                    or np.all(np.isfinite(array))
                ),
            }
        )
    return {
        "leaf_count": len(rows),
        "leaves": rows,
        "sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _run_arm(runtime, jax, jnp, carry, namelist, clock) -> tuple[Any, dict[str, Any]]:
    runtime._advance_chunk_fori.clear_cache()
    jax.clear_caches()
    native_step = jnp.asarray(1, dtype=jnp.int32)
    started = time.perf_counter()
    lowered = runtime._advance_chunk_fori.lower(
        carry,
        namelist,
        native_step,
        clock,
        n_steps=2,
        cadence=int(namelist.radiation_cadence_steps),
    )
    lower_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    targets = sorted(
        set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', stablehlo))
    )
    lowered_text = stablehlo.lower()
    forbidden = sorted(
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
            "send_to_host",
            "recv_from_host",
        )
        if token in lowered_text
    )
    input_tree = jax.tree_util.tree_structure(carry)
    output_tree = jax.tree_util.tree_structure(lowered.out_info)
    input_avals = [
        (tuple(value.shape), str(value.dtype))
        for value in jax.tree_util.tree_leaves(carry)
    ]
    output_avals = [
        (tuple(value.shape), str(value.dtype))
        for value in jax.tree_util.tree_leaves(lowered.out_info)
    ]
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = executable(
        carry,
        namelist,
        native_step,
        clock,
        n_steps=2,
        cadence=int(namelist.radiation_cadence_steps),
    )
    result = jax.device_get(result)
    dispatch_seconds = time.perf_counter() - started
    return result, {
        "completed_native_steps": [1, 2],
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "dispatch_seconds": dispatch_seconds,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
        "stablehlo_bytes": len(stablehlo.encode()),
        "custom_call_targets": targets,
        "unknown_custom_call_targets": sorted(set(targets) - ALLOWED_CPU_CUSTOM_TARGETS),
        "forbidden_tokens": forbidden,
        "interface_identity": bool(
            input_tree == output_tree and input_avals == output_avals
        ),
        "leaf_count": len(input_avals),
    }


def _load_frames(np) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from netCDF4 import Dataset

    requested = {
        "T",
        "THM",
        "QVAPOR",
        "QCLOUD",
        "QRAIN",
        "QICE",
        "QSNOW",
        "QGRAUP",
        "QNICE",
        "QNRAIN",
        "QNSNOW",
        "QNGRAUPEL",
        "P",
        "PH",
        "PSFC",
        "U",
        "V",
        "W",
        "RAINNC",
        "SNOWNC",
        "GRAUPELNC",
        "I_RAINNC",
        "LANDMASK",
    }
    loaded: dict[str, dict[str, Any]] = {}
    authority: dict[str, Any] = {}
    for name, (path, expected) in FRAMES.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frame hash mismatch {name}: {actual}")
        with Dataset(path) as dataset:
            loaded[name] = {
                field: np.asarray(dataset.variables[field][0], dtype=np.float64)
                for field in requested
                if field in dataset.variables
            }
        authority[name] = {"path": str(path), "sha256": actual}
    return loaded, authority


def _horizontal_class(np, landmask, shape: tuple[int, int]):
    ny, nx = landmask.shape
    if shape == (ny, nx):
        return landmask >= 0.5
    if shape == (ny, nx + 1):
        face = np.empty(shape, dtype=np.float64)
        face[:, 0] = landmask[:, 0]
        face[:, -1] = landmask[:, -1]
        face[:, 1:-1] = 0.5 * (landmask[:, :-1] + landmask[:, 1:])
        return face >= 0.5
    if shape == (ny + 1, nx):
        face = np.empty(shape, dtype=np.float64)
        face[0, :] = landmask[0, :]
        face[-1, :] = landmask[-1, :]
        face[1:-1, :] = 0.5 * (landmask[:-1, :] + landmask[1:, :])
        return face >= 0.5
    raise ValueError(f"unsupported horizontal shape {shape}")


def _masks(np, landmask, shape: tuple[int, int]) -> dict[str, Any]:
    land = _horizontal_class(np, landmask, shape)
    yy, xx = np.indices(shape)
    distance = np.minimum.reduce((yy, xx, shape[0] - 1 - yy, shape[1] - 1 - xx))
    inner = distance >= 5
    return {
        "all": np.ones(shape, dtype=bool),
        "land": land,
        "sea": ~land,
        "inner5": inner,
        "land_inner5": land & inner,
        "sea_inner5": (~land) & inner,
    }


def _rmse(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _projection(np, runtime, baseline, candidate, namelist, frames) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for state_name, history_name in STATE_TO_HISTORY.items():
        left = getattr(baseline.state, state_name)
        right = getattr(candidate.state, state_name)
        if left is None or right is None:
            continue
        delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
        if np.count_nonzero(delta):
            deltas[history_name] = delta
    psfc_a = np.asarray(runtime._psfc_from_state(baseline.state, namelist.metrics))
    psfc_b = np.asarray(runtime._psfc_from_state(candidate.state, namelist.metrics))
    if np.count_nonzero(psfc_b - psfc_a):
        deltas["PSFC"] = psfc_b - psfc_a

    current = frames["current"]
    projected: dict[str, Any] = {}
    for field, delta in deltas.items():
        if field in current:
            projected[field] = current[field] + delta
    if "THM" in projected or "QVAPOR" in projected:
        projected_thm = projected.get("THM", current["THM"])
        projected_qv = projected.get("QVAPOR", current["QVAPOR"])
        projected["T"] = (projected_thm + 300.0) / (
            1.0 + (461.6 / 287.0) * projected_qv
        ) - 300.0
        deltas["T"] = projected["T"] - current["T"]

    landmask = current["LANDMASK"]
    fields: dict[str, Any] = {}
    no_worse_failures: list[dict[str, Any]] = []
    unavailable_oracles: dict[str, list[str]] = {}
    for field in sorted(projected):
        delta = deltas[field]
        shape = projected[field].shape[-2:]
        field_masks = _masks(np, landmask, shape)
        row = {
            "changed_values": int(np.count_nonzero(delta)),
            "delta_rms": _rmse(np, delta),
            "delta_max_abs": float(np.max(np.abs(delta))),
            "anchors": {},
        }
        missing = [
            anchor for anchor in ("cpu_wrf", "retry20")
            if field not in frames[anchor]
        ]
        if missing:
            unavailable_oracles[field] = missing
        for anchor in ("cpu_wrf", "retry20"):
            if field not in frames[anchor]:
                continue
            anchor_rows = {}
            for mask_name, mask in field_masks.items():
                if not np.any(mask):
                    raise RuntimeError(f"empty mask {field}/{mask_name}")
                before = _rmse(
                    np, current[field][..., mask] - frames[anchor][field][..., mask]
                )
                after = _rmse(
                    np, projected[field][..., mask] - frames[anchor][field][..., mask]
                )
                green = bool(after <= before)
                anchor_rows[mask_name] = {
                    "current_rmse": before,
                    "projected_rmse": after,
                    "no_worse": green,
                }
                if not green:
                    no_worse_failures.append(
                        {
                            "field": field,
                            "anchor": anchor,
                            "mask": mask_name,
                            "current_rmse": before,
                            "projected_rmse": after,
                        }
                    )
            row["anchors"][anchor] = anchor_rows
        fields[field] = row
    return {
        "method": (
            "Add the complete two-step candidate-minus-candidate-off delta to the "
            "authenticated Stage-Omega 00:20 frame.  This is a directional gate, "
            "not a claimed 200-step trajectory substitute."
        ),
        "fields": fields,
        "unavailable_oracles": unavailable_oracles,
        "no_worse_failures": no_worse_failures,
        "green": not no_worse_failures,
    }


def main() -> int:
    try:
        actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if _sha256(STEP0) != STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend {jax.default_backend()}")
        if jax.config.values.get("jax_cpu_enable_async_dispatch") is not False:
            raise RuntimeError("CPU async dispatch remained enabled")
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != [12]:
            raise RuntimeError(f"CPU affinity changed: {affinity}")

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-moist-theta-interface-ab-"))
        try:
            load_dir = scratch / "load"
            load_dir.mkdir()
            tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
                load_dir
            )
            if names != ("d01", "d02", "d03") or dt_by_domain != {
                "d01": 54.0,
                "d02": 18.0,
                "d03": 6.0,
            }:
                raise RuntimeError("canonical domain hierarchy changed")
            with STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            if hasattr(carry, "h_diabatic"):
                raise RuntimeError("retained pre-candidate carry unexpectedly owns h_diabatic")
            # Step0 source authority is exact: pristine WRF initializes h to
            # zero, and the retained carry is the completed cold-start step 0.
            object.__setattr__(carry, "h_diabatic", jnp.zeros_like(carry.state.theta))

            candidate_namelist = tree.domains["d03"].namelist
            if not runtime._moist_theta_interface_active(candidate_namelist):
                raise RuntimeError("complete candidate inactive in canonical loader")
            if int(candidate_namelist.mp_physics) != 8 or float(
                candidate_namelist.mp_tend_lim
            ) != 10.0:
                raise RuntimeError("authenticated mp=8/mp_tend_lim authority changed")
            baseline_namelist = dataclasses.replace(
                candidate_namelist, moist_theta_interface=False
            )
            clock = runtime.build_clock_base(candidate_namelist)

            print("MOIST_THETA_INTERFACE baseline A lower/compile/2-dispatch", flush=True)
            baseline, baseline_audit = _run_arm(
                runtime, jax, jnp, carry, baseline_namelist, clock
            )
            print("MOIST_THETA_INTERFACE candidate B lower/compile/2-dispatch", flush=True)
            candidate, candidate_audit = _run_arm(
                runtime, jax, jnp, carry, candidate_namelist, clock
            )

            baseline_manifest = _manifest(jax, np, baseline)
            candidate_manifest = _manifest(jax, np, candidate)
            frames, frame_authority = _load_frames(np)
            projection = _projection(
                np, runtime, baseline, candidate, candidate_namelist, frames
            )
            same_structure = all(
                left["path"] == right["path"]
                and left["shape"] == right["shape"]
                and left["dtype"] == right["dtype"]
                for left, right in zip(
                    baseline_manifest["leaves"],
                    candidate_manifest["leaves"],
                    strict=True,
                )
            )
            nonnegative = {}
            for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng"):
                values = np.asarray(getattr(candidate.state, name), dtype=np.float64)
                nonnegative[name] = {
                    "finite": bool(np.all(np.isfinite(values))),
                    "min": float(np.min(values)),
                    "nonnegative": bool(np.min(values) >= 0.0),
                }
            h = np.asarray(candidate.h_diabatic, dtype=np.float64)
            h_active = bool(np.all(np.isfinite(h)) and np.max(np.abs(h)) > 0.0)
            changed_state_fields = sorted(
                name
                for name in STATE_TO_HISTORY
                if getattr(baseline.state, name) is not None
                and getattr(candidate.state, name) is not None
                and not np.array_equal(
                    np.asarray(getattr(baseline.state, name)),
                    np.asarray(getattr(candidate.state, name)),
                )
            )
            checks = {
                "authenticated_step0_loader_and_frames": True,
                "step0_h_zero_source_authorized": bool(
                    np.count_nonzero(np.asarray(carry.h_diabatic)) == 0
                ),
                "two_completed_steps_activate_produce_then_consume_h": h_active,
                "both_interfaces_identical": baseline_audit["interface_identity"]
                and candidate_audit["interface_identity"],
                "complete_structure_identity": same_structure
                and baseline_manifest["leaf_count"] == candidate_manifest["leaf_count"],
                "both_complete_carries_finite": all(
                    row["finite"] for row in baseline_manifest["leaves"]
                )
                and all(row["finite"] for row in candidate_manifest["leaves"]),
                "callback_transfer_unknown_free": not baseline_audit["forbidden_tokens"]
                and not candidate_audit["forbidden_tokens"]
                and not baseline_audit["unknown_custom_call_targets"]
                and not candidate_audit["unknown_custom_call_targets"],
                "complete_output_changed": baseline_manifest["sha256"]
                != candidate_manifest["sha256"],
                "source_effect_reaches_thermo_pressure_momentum": all(
                    name in changed_state_fields
                    for name in (
                        "theta",
                        "qv",
                        "qc",
                        "p_perturbation",
                        "ph_perturbation",
                        "u",
                        "v",
                        "w",
                    )
                ),
                "all_species_finite_nonnegative": all(
                    row["finite"] and row["nonnegative"]
                    for row in nonnegative.values()
                ),
                "all_affected_available_field_masks_no_worse": projection["green"],
            }
            proof = {
                "schema": "gpuwrf.v0234.complete-moist-theta-interface-cpu-ab.v1",
                "environment": actual_env,
                "effective_runtime": {
                    "backend": jax.default_backend(),
                    "jax_cpu_enable_async_dispatch": False,
                    "cpu_affinity": affinity,
                    "gpu_queries": 0,
                    "gpu_commands": 0,
                },
                "input": {
                    "path": str(STEP0),
                    "sha256": STEP0_SHA256,
                    "completed_step": 0,
                    "seeded_h_diabatic": "source-authorized exact zero",
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "mechanism": (
                    "A retains step-entry legacy Thompson. B performs dry prep/literal "
                    "moist finish after RK, persists h from step1, injects it on every "
                    "RK stage of step2, cancels only final RK, and refreshes pressure."
                ),
                "baseline_A": {**baseline_audit, "manifest": baseline_manifest},
                "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
                "candidate_h_diabatic": {
                    "finite": bool(np.all(np.isfinite(h))),
                    "min": float(np.min(h)),
                    "max": float(np.max(h)),
                    "nonzero_values": int(np.count_nonzero(h)),
                },
                "candidate_species": nonnegative,
                "changed_state_fields": changed_state_fields,
                "affected_field_projection": projection,
                "checks": checks,
                "verdict": (
                    "MOIST_THETA_INTERFACE_CPU_AB_GREEN"
                    if all(checks.values())
                    else "MOIST_THETA_INTERFACE_CPU_AB_RED"
                ),
            }
            proof["proof_sha256"] = _canonical(proof)
            _atomic(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": proof["verdict"],
                        "proof_sha256": proof["proof_sha256"],
                        "failed_checks": [
                            name for name, green in checks.items() if not green
                        ],
                        "no_worse_failures": projection["no_worse_failures"],
                        "changed_state_fields": changed_state_fields,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0 if all(checks.values()) else 3
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Exception as exc:
        import traceback

        blocker = {
            "schema": "gpuwrf.v0234.complete-moist-theta-interface-cpu-ab-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": {name: os.environ.get(name) for name in REQUIRED_ENV},
            "gpu_queries": 0,
            "gpu_commands": 0,
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic(FAILURE, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
