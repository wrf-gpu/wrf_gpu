"""Authenticated real-Step0 CPU A/B for per-substep WRF ``t_2save`` ownership."""

from __future__ import annotations

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
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-t2save-ownership-complete-cpu-ab-proof.json"
FAILURE = SPRINT / "nested-t2save-ownership-complete-cpu-ab-blocker.json"
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
    "current_395": (
        FRAME_ROOT
        / "corrected_ni_rca_max_22c2bd7a/nested_h_sca_order_395fb800_full18h_discriminator1/"
        "output/wrfout_d03_2025-03-01_00:20:00",
        "3235053155927d04ee11971e34ddce71ba81572bf9f1b94a043e59c14d0c6c10",
    ),
    "immutable_448": (
        FRAME_ROOT
        / "corrected_ni_rca_max_22c2bd7a/nested_boundary_final_4484be85_full18h_owner_override1/"
        "output/wrfout_d03_2025-03-01_00:20:00",
        "79492ab0b0f8482809a2486f72969973f13dbd35f374801cd787156238a72f7c",
    ),
    "partial_wind_2c": (
        FRAME_ROOT
        / "corrected_ni_rca_max_22c2bd7a/nested_advection_degrade_2c13b731_full18h_discriminator1/"
        "output/wrfout_d03_2025-03-01_00:20:00",
        "3cf8dcda34ae469df86028209f5f8c2ed59ef4fb1cfd38e2148ee33bcc5ab599",
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
RETAINED_395_CPU_HLO_SHA256 = "adb3c3eb4946fd78a8855bca5713cdef6d0fcf7bdfe72d2a94f6655e614732e6"
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}
ALLOWED_CPU_CUSTOM_TARGETS = {"lapack_dgtsv_ffi"}


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _rms(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _ring(np, ny: int, nx: int, distance: int = 1):
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == distance


def _path_name(path: tuple[Any, ...]) -> str:
    parts = []
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
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    lower_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo_lower = stablehlo.lower()
    custom_targets = sorted(
        set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', stablehlo))
    )
    unknown_targets = sorted(set(custom_targets) - ALLOWED_CPU_CUSTOM_TARGETS)
    forbidden_tokens = sorted(
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
        if token in stablehlo_lower
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
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    result = jax.device_get(result)
    dispatch_seconds = time.perf_counter() - started
    return result, {
        "native_step": 1,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "dispatch_seconds": dispatch_seconds,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
        "stablehlo_bytes": len(stablehlo.encode()),
        "custom_call_targets": custom_targets,
        "unknown_custom_call_targets": unknown_targets,
        "forbidden_tokens": forbidden_tokens,
        "interface_identity": bool(
            input_tree == output_tree
            and input_avals == output_avals
            and len(input_avals) == 106
        ),
    }


def _state_delta(np, left, right) -> dict[str, Any]:
    rows = []
    for name in sorted(set(dir(left.state)) & set(dir(right.state))):
        if name.startswith("_"):
            continue
        a0 = getattr(left.state, name, None)
        b0 = getattr(right.state, name, None)
        if a0 is None or b0 is None or not hasattr(a0, "shape"):
            continue
        a = np.asarray(a0)
        b = np.asarray(b0)
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
            continue
        delta = b.astype(np.float64) - a.astype(np.float64)
        changed = int(np.count_nonzero(delta))
        if not changed:
            continue
        row = {
            "field": name,
            "changed_values": changed,
            "rms": _rms(np, delta),
            "max_abs": float(np.max(np.abs(delta))),
        }
        if delta.ndim >= 2 and delta.shape[-2:] == right.state.theta.shape[-2:]:
            mask = _ring(np, delta.shape[-2], delta.shape[-1])
            row["ring1_rms"] = _rms(np, delta[..., mask])
        rows.append(row)
    return {"changed_fields": [row["field"] for row in rows], "fields": rows}


def _literal_t2ave(np, row: dict[str, Any]) -> Any:
    return (
        0.5
        * (
            (1.0 + row["epssm"]) * row["t_after"]
            + (1.0 - row["epssm"]) * row["t_before"]
        )
        + row["c1h"][:, None, None] * row["muave"][None] * 300.0
    ) / (
        (row["c1h"][:, None, None] * row["muts"][None] + row["c2h"][:, None, None])
        * (300.0 + row["t1"])
    )


def _real_recurrence_oracle(
    acoustic, runtime, jax, jnp, np, carry, namelist
) -> dict[str, Any]:
    base = runtime._acoustic_core_state(carry, namelist)
    # Pickled retained carries materialize as NumPy arrays. The production JIT
    # accepts those as inputs, while this eager CPU-only recurrence calls the
    # array-update helpers directly and therefore needs ordinary JAX CPU arrays.
    base = jax.tree_util.tree_map(
        lambda value: (
            jnp.asarray(value)
            if value is not None and hasattr(value, "shape")
            else value
        ),
        base,
        is_leaf=lambda value: value is None,
    )
    base = base.replace(
        theta_coupled_work=base.theta,
        t_2ave=base.theta,
        theta_ave=base.theta,
    )
    cfg = acoustic.AcousticCoreConfig(
        dt=float(namelist.dt_s) / float(namelist.acoustic_substeps),
        dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        w_damping=int(namelist.w_damping),
        damp_opt=int(namelist.damp_opt),
        dampcoef=float(namelist.dampcoef),
        zdamp=float(namelist.zdamp),
        dt_full=float(namelist.dt_s),
        nested_frozen_wrf_boundary_bundle=True,
        periodic_x=False,
        specified=False,
        nested=True,
        spec_zone=int(namelist.boundary_config.spec_zone),
    )
    original_selector = acoustic._t_2save_for_advance_w
    original_advance_w = acoustic.advance_w_wrf
    arms: dict[str, Any] = {}
    try:
        for label, candidate in (("retained_A", False), ("candidate_B", True)):
            for substeps in (5, 10):
                selected_rows: list[dict[str, Any]] = []

                def selector(pre, prior, *, _candidate=candidate):
                    selected_rows.append(
                        {
                            "t_before": np.asarray(jax.device_get(pre), dtype=np.float64),
                            "prior": np.asarray(jax.device_get(prior), dtype=np.float64),
                        }
                    )
                    return pre if _candidate else prior

                def observed_advance_w(**kwargs):
                    result = original_advance_w(**kwargs)
                    row = selected_rows[-1]
                    row.update(
                        {
                            "actual_input": np.asarray(
                                jax.device_get(kwargs["t_2ave"]), dtype=np.float64
                            ),
                            "t_after": np.asarray(
                                jax.device_get(kwargs["t_2"]), dtype=np.float64
                            ),
                            "t1": np.asarray(jax.device_get(kwargs["t_1"]), dtype=np.float64),
                            "c1h": np.asarray(jax.device_get(kwargs["c1h"]), dtype=np.float64),
                            "c2h": np.asarray(jax.device_get(kwargs["c2h"]), dtype=np.float64),
                            "muave": np.asarray(
                                jax.device_get(kwargs["muave"]), dtype=np.float64
                            ),
                            "muts": np.asarray(jax.device_get(kwargs["muts"]), dtype=np.float64),
                            "epssm": float(kwargs["epssm"]),
                            "actual_output": np.asarray(
                                jax.device_get(result[2]), dtype=np.float64
                            ),
                        }
                    )
                    return result

                acoustic._t_2save_for_advance_w = selector
                acoustic.advance_w_wrf = observed_advance_w
                acoustic.acoustic_scan_core(
                    base,
                    namelist.metrics,
                    cfg,
                    substeps=substeps,
                )
                rows = []
                for index, row in enumerate(selected_rows, start=1):
                    expected = _literal_t2ave(np, row)
                    rows.append(
                        {
                            "sound_step": index,
                            "selected_pre_theta_exact": bool(
                                np.array_equal(row["actual_input"], row["t_before"])
                            ),
                            "prior_equals_pre": bool(
                                np.array_equal(row["prior"], row["t_before"])
                            ),
                            "literal_max_abs_error": float(
                                np.max(np.abs(row["actual_output"] - expected))
                            ),
                        }
                    )
                arms[f"{label}_rk{2 if substeps == 5 else 3}"] = rows
    finally:
        acoustic._t_2save_for_advance_w = original_selector
        acoustic.advance_w_wrf = original_advance_w

    candidate_rows = arms["candidate_B_rk2"] + arms["candidate_B_rk3"]
    retained_rows = arms["retained_A_rk2"] + arms["retained_A_rk3"]
    return {
        "arms": arms,
        "candidate_all_substeps_select_pre_theta": all(
            row["selected_pre_theta_exact"] for row in candidate_rows
        ),
        "candidate_literal_max_abs_error": max(
            row["literal_max_abs_error"] for row in candidate_rows
        ),
        "retained_first_substeps_inactive": bool(
            arms["retained_A_rk2"][0]["prior_equals_pre"]
            and arms["retained_A_rk3"][0]["prior_equals_pre"]
        ),
        "retained_active_after_first": all(
            not row["prior_equals_pre"]
            for key in ("retained_A_rk2", "retained_A_rk3")
            for row in arms[key][1:]
        ),
    }


def _load_frames(np) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from netCDF4 import Dataset

    loaded: dict[str, dict[str, Any]] = {}
    authority = {}
    for name, (path, expected) in FRAMES.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frame hash mismatch {name}: {actual}")
        with Dataset(path) as dataset:
            loaded[name] = {
                field: np.asarray(dataset.variables[field][0], dtype=np.float64)
                for field in ("T", "THM", "QVAPOR", "U")
            }
        authority[name] = {"path": str(path), "sha256": actual}
    return loaded, authority


def _projection(np, retained, candidate, frames) -> dict[str, Any]:
    theta_delta = np.asarray(candidate.state.theta, dtype=np.float64) - np.asarray(
        retained.state.theta, dtype=np.float64
    )
    qv_delta = np.asarray(candidate.state.qv, dtype=np.float64) - np.asarray(
        retained.state.qv, dtype=np.float64
    )
    u_delta = np.asarray(candidate.state.u, dtype=np.float64) - np.asarray(
        retained.state.u, dtype=np.float64
    )
    current = frames["current_395"]
    rvovrd = 461.6 / 287.0
    projected_thm = current["THM"] + theta_delta
    projected_qv = current["QVAPOR"] + qv_delta
    projected_t = (projected_thm + 300.0) / (
        1.0 + rvovrd * np.maximum(projected_qv, 0.0)
    ) - 300.0
    projected_u = current["U"] + u_delta
    t_mask = _ring(np, projected_t.shape[-2], projected_t.shape[-1])
    u_mask = _ring(np, projected_u.shape[-2], projected_u.shape[-1])

    result = {
        "one_step_delta": {
            "THM_ring1_rms_K": _rms(np, theta_delta[:, t_mask]),
            "QVAPOR_ring1_rms": _rms(np, qv_delta[:, t_mask]),
            "U_ring1_rms_m_s": _rms(np, u_delta[:, u_mask]),
        },
        "anchors": {},
    }
    for anchor in ("cpu_wrf", "retry20"):
        result["anchors"][anchor] = {
            "T_current_ring1_rmse_K": _rms(
                np, current["T"][:, t_mask] - frames[anchor]["T"][:, t_mask]
            ),
            "T_projected_ring1_rmse_K": _rms(
                np, projected_t[:, t_mask] - frames[anchor]["T"][:, t_mask]
            ),
            "THM_current_ring1_rmse_K": _rms(
                np, current["THM"][:, t_mask] - frames[anchor]["THM"][:, t_mask]
            ),
            "THM_projected_ring1_rmse_K": _rms(
                np, projected_thm[:, t_mask] - frames[anchor]["THM"][:, t_mask]
            ),
            "U_current_ring1_rmse_m_s": _rms(
                np, current["U"][:, u_mask] - frames[anchor]["U"][:, u_mask]
            ),
            "U_projected_ring1_rmse_m_s": _rms(
                np, projected_u[:, u_mask] - frames[anchor]["U"][:, u_mask]
            ),
            "U_partial_2c_ring1_rmse_m_s": _rms(
                np,
                frames["partial_wind_2c"]["U"][:, u_mask]
                - frames[anchor]["U"][:, u_mask],
            ),
            "U_immutable_448_ring1_rmse_m_s": _rms(
                np,
                frames["immutable_448"]["U"][:, u_mask]
                - frames[anchor]["U"][:, u_mask],
            ),
        }
    return result


def main() -> int:
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if _sha256(STEP0) != STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")

        import jax
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]

        import gpuwrf.dynamics.core.acoustic as acoustic
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-t2save-cpu-ab-"))
        try:
            load_dir = scratch / "load"
            load_dir.mkdir()
            tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
                load_dir
            )
            if names != ("d01", "d02", "d03"):
                raise RuntimeError(f"domain order changed: {names!r}")
            if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
                raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
            with STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            namelist = tree.domains["d03"].namelist
            clock = runtime.build_clock_base(namelist)
            if int(namelist.acoustic_substeps) != 10:
                raise RuntimeError("canonical acoustic substep count changed")

            recurrence = _real_recurrence_oracle(
                acoustic, runtime, jax, jnp, np, carry, namelist
            )

            original_selector = acoustic._t_2save_for_advance_w

            def retained_selector(_pre, prior):
                return prior

            acoustic._t_2save_for_advance_w = retained_selector
            try:
                print("T2SAVE_CPU_AB retained A lower/compile/dispatch", flush=True)
                retained, retained_audit = _run_arm(
                    runtime, jax, jnp, carry, namelist, clock
                )
            finally:
                acoustic._t_2save_for_advance_w = original_selector

            print("T2SAVE_CPU_AB candidate B lower/compile/dispatch", flush=True)
            candidate, candidate_audit = _run_arm(
                runtime, jax, jnp, carry, namelist, clock
            )

            retained_manifest = _manifest(jax, np, retained)
            candidate_manifest = _manifest(jax, np, candidate)
            frames, frame_authority = _load_frames(np)
            projection = _projection(np, retained, candidate, frames)
            deltas = _state_delta(np, retained, candidate)
            structure_identity = all(
                a["path"] == b["path"]
                and a["shape"] == b["shape"]
                and a["dtype"] == b["dtype"]
                for a, b in zip(
                    retained_manifest["leaves"],
                    candidate_manifest["leaves"],
                    strict=True,
                )
            )
            t_improves = all(
                row["T_projected_ring1_rmse_K"]
                < row["T_current_ring1_rmse_K"]
                and row["THM_projected_ring1_rmse_K"]
                < row["THM_current_ring1_rmse_K"]
                for row in projection["anchors"].values()
            )
            u_retained = all(
                row["U_projected_ring1_rmse_m_s"]
                <= row["U_partial_2c_ring1_rmse_m_s"]
                for row in projection["anchors"].values()
            )
            checks = {
                "authenticated_step0_and_frames": True,
                "canonical_rk2_rk3_sound_counts": True,
                "candidate_recurrence_selects_pre_mu_theta_every_substep": recurrence[
                    "candidate_all_substeps_select_pre_theta"
                ],
                "candidate_recurrence_matches_literal_fp64": recurrence[
                    "candidate_literal_max_abs_error"
                ]
                <= 2.0e-13,
                "retained_recurrence_inactive_first_active_after": recurrence[
                    "retained_first_substeps_inactive"
                ]
                and recurrence["retained_active_after_first"],
                "retained_interface_106_identity": retained_audit["interface_identity"],
                "candidate_interface_106_identity": candidate_audit["interface_identity"],
                "complete_leaf_structure_identity": structure_identity
                and retained_manifest["leaf_count"]
                == candidate_manifest["leaf_count"]
                == 106,
                "both_all_106_finite": all(
                    row["finite"] for row in retained_manifest["leaves"]
                )
                and all(row["finite"] for row in candidate_manifest["leaves"]),
                "retained_hlo_authenticated_to_395": retained_audit[
                    "stablehlo_sha256"
                ]
                == RETAINED_395_CPU_HLO_SHA256,
                "both_callback_transfer_unknown_free": not retained_audit[
                    "forbidden_tokens"
                ]
                and not candidate_audit["forbidden_tokens"]
                and not retained_audit["unknown_custom_call_targets"]
                and not candidate_audit["unknown_custom_call_targets"],
                "complete_output_changed": retained_manifest["sha256"]
                != candidate_manifest["sha256"],
                "source_effect_reaches_THM_W_PH": all(
                    name in deltas["changed_fields"]
                    for name in ("theta", "w", "ph_perturbation")
                ),
                "T_and_THM_direction_improve_both_anchors": t_improves,
                "U_retains_partial_wind_margin": u_retained,
            }
            proof = {
                "schema": "gpuwrf.v0234.nested-t2save-ownership-complete-cpu-ab.v1",
                "environment": actual_env,
                "input": {
                    "path": str(STEP0),
                    "sha256": STEP0_SHA256,
                    "carry_completed_step": 0,
                    "dispatched_native_step": 1,
                    "leaf_count": 106,
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "source_mechanism": "per-acoustic-substep pre-advance_mu_t coupled t_2save passed immediately to advance_w",
                "real_step0_recurrence_oracle": recurrence,
                "retained_A": {**retained_audit, "manifest": retained_manifest},
                "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
                "complete_output_delta": deltas,
                "step200_directional_projection": projection,
                "checks": checks,
                "verdict": (
                    "NESTED_T2SAVE_OWNERSHIP_CPU_AB_GREEN"
                    if all(checks.values())
                    else "NESTED_T2SAVE_OWNERSHIP_CPU_AB_RED"
                ),
            }
            proof["proof_sha256"] = _canonical(proof)
            _atomic_json(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": proof["verdict"],
                        "proof_sha256": proof["proof_sha256"],
                        "failed_checks": [
                            name for name, passed in checks.items() if not passed
                        ],
                        "projection": projection["anchors"],
                        "changed_fields": deltas["changed_fields"],
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
            "schema": "gpuwrf.v0234.nested-t2save-ownership-complete-cpu-ab-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "gpu_queries": 0,
            "proof_sha256": "",
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(FAILURE, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
