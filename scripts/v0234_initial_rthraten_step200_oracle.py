"""Real-State CPU oracle for WRF's missing first-interval RTHRATEN source."""

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

from scripts import v0234_original_moist_sumflux_cpu_ab as projection_support
from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "initial-rthraten-step200-direct-oracle.json"
FAILURE = SPRINT / "initial-rthraten-step200-direct-oracle-blocker.json"
REQUIRED_ENV = {**common.REQUIRED_ENV, "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false"}
WRF_RADIATION_DRIVER = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/phys/module_radiation_driver.F"
)
WRF_RADIATION_DRIVER_SHA256 = (
    "4f0177189eb616c9220fe2022186dd6cc310666ca29977557dfe53d4409d733b"
)
ALLOWED_CPU_CUSTOM_TARGETS = {"lapack_dgtsv_ffi"}


def _ring(np, ny: int, nx: int, distance: int = 1):
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == distance


def _rms(np, value) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def main() -> int:
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if common._sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")
        if common._sha256(WRF_RADIATION_DRIVER) != WRF_RADIATION_DRIVER_SHA256:
            raise RuntimeError("pristine WRF radiation-driver hash mismatch")

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        effective_async = jax.config.values.get("jax_cpu_enable_async_dispatch")
        import jax.numpy as jnp
        import numpy as np

        affinity = sorted(os.sched_getaffinity(0))
        if (
            effective_async is not False
            or jax.default_backend() != "cpu"
            or affinity != [12]
        ):
            raise RuntimeError(
                f"runtime binding mismatch async={effective_async} "
                f"backend={jax.default_backend()} affinity={affinity}"
            )

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        from gpuwrf.coupling.physics_couplers import rrtmg_theta_tendency
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-initial-rthraten-oracle-"))
        try:
            load_dir = scratch / "load"
            load_dir.mkdir()
            tree, names, _initial, dt_by_domain, load_authority = (
                ordinary.load_corrected_tree(load_dir)
            )
            if names != ("d01", "d02", "d03") or dt_by_domain["d03"] != 6.0:
                raise RuntimeError("canonical hierarchy changed")
            namelist = tree.domains["d03"].namelist
            if (
                int(namelist.ra_sw_physics),
                int(namelist.ra_lw_physics),
                int(namelist.radiation_cadence_steps),
                int(namelist.rad_rk_tendf),
            ) != (4, 4, 300, 1):
                raise RuntimeError("canonical d03 radiation configuration changed")
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            retained_rate = np.asarray(carry.rthraten, dtype=np.float64)
            if np.count_nonzero(retained_rate) != 0:
                raise RuntimeError("retained Step0 RTHRATEN is no longer exact zero")

            radt_seconds = float(namelist.dt_s) * int(
                namelist.radiation_cadence_steps
            )
            midpoint_seconds = 0.5 * radt_seconds

            def first_call_rate(state, land_state):
                return rrtmg_theta_tendency(
                    state,
                    namelist.grid,
                    time_utc=namelist.time_utc,
                    lead_seconds=midpoint_seconds,
                    radiation_static=namelist.radiation_static,
                    topo_shading=int(namelist.topo_shading),
                    slope_rad=int(namelist.slope_rad),
                    shadow_length_m=float(namelist.topo_shadow_length_m),
                    land_state=land_state,
                )

            compiled_rate = jax.jit(first_call_rate)
            started = time.perf_counter()
            lowered = compiled_rate.lower(carry.state, carry.noahmp_land)
            lower_seconds = time.perf_counter() - started
            stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
            stablehlo_lower = stablehlo.lower()
            custom_targets = sorted(
                set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', stablehlo))
            )
            unknown_targets = sorted(
                set(custom_targets) - ALLOWED_CPU_CUSTOM_TARGETS
            )
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
            started = time.perf_counter()
            executable = lowered.compile()
            compile_seconds = time.perf_counter() - started
            started = time.perf_counter()
            candidate_rate = np.asarray(
                jax.device_get(executable(carry.state, carry.noahmp_land)),
                dtype=np.float64,
            )
            dispatch_seconds = time.perf_counter() - started

            frames, frame_authority = projection_support._load_frames(np)
            current = frames["current_395"]
            qv = np.asarray(current["QVAPOR"], dtype=np.float64)
            moist_factor = 1.0 + (461.6 / 287.0) * qv
            applied_seconds = 200.0 * float(namelist.dt_s)
            dry_theta_delta = applied_seconds * candidate_rate
            moist_theta_delta = moist_factor * dry_theta_delta
            projected_thm = current["THM"] + moist_theta_delta
            projected_t = (projected_thm + 300.0) / moist_factor - 300.0
            mask = _ring(np, *projected_t.shape[-2:])
            anchors = {}
            for anchor in ("cpu_wrf", "retry20"):
                anchors[anchor] = {
                    "T_current_ring1_rmse_K": _rms(
                        np, current["T"][:, mask] - frames[anchor]["T"][:, mask]
                    ),
                    "T_projected_ring1_rmse_K": _rms(
                        np, projected_t[:, mask] - frames[anchor]["T"][:, mask]
                    ),
                    "THM_current_ring1_rmse_K": _rms(
                        np,
                        current["THM"][:, mask]
                        - frames[anchor]["THM"][:, mask],
                    ),
                    "THM_projected_ring1_rmse_K": _rms(
                        np,
                        projected_thm[:, mask]
                        - frames[anchor]["THM"][:, mask],
                    ),
                }
            directional_green = all(
                row["T_projected_ring1_rmse_K"]
                < row["T_current_ring1_rmse_K"]
                and row["THM_projected_ring1_rmse_K"]
                < row["THM_current_ring1_rmse_K"]
                for row in anchors.values()
            )
            finite_nonzero = bool(
                np.all(np.isfinite(candidate_rate))
                and np.count_nonzero(candidate_rate) > 0
            )
            checks = {
                "fresh_synchronous_cpu12": effective_async is False
                and affinity == [12],
                "authenticated_step0_and_loader": True,
                "retained_step0_rthraten_exact_zero": True,
                "pristine_wrf_first_call_and_midpoint_source_bound": True,
                "candidate_rate_finite_nonzero": finite_nonzero,
                "radiation_only_hlo_callback_transfer_unknown_free": not forbidden_tokens
                and not unknown_targets,
                "T_and_THM_direct_projection_improve_both_anchors": directional_green,
            }
            proof = {
                "schema": "gpuwrf.v0234.initial-rthraten-step200-direct-oracle.v1",
                "environment": actual_env,
                "effective_runtime": {
                    "backend": jax.default_backend(),
                    "jax_cpu_enable_async_dispatch": effective_async,
                    "cpu_affinity": affinity,
                    "fresh_arm": True,
                    "resumed_from_async_checkpoint": False,
                },
                "input": {
                    "path": str(common.STEP0),
                    "sha256": common.STEP0_SHA256,
                    "completed_step": 0,
                    "retained_rthraten_nonzero": 0,
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "source_oracle": {
                    "wrf_radiation_driver": str(WRF_RADIATION_DRIVER),
                    "sha256": WRF_RADIATION_DRIVER_SHA256,
                    "wrf_calls": [1, 301],
                    "retained_gpuwrf_calls": [300, 600],
                    "radt_seconds": radt_seconds,
                    "wrf_first_call_midpoint_seconds": midpoint_seconds,
                    "step200_applied_seconds": applied_seconds,
                },
                "candidate_rthraten": {
                    "shape": list(candidate_rate.shape),
                    "dtype": str(candidate_rate.dtype),
                    "nonzero_values": int(np.count_nonzero(candidate_rate)),
                    "min_K_s": float(np.min(candidate_rate)),
                    "max_K_s": float(np.max(candidate_rate)),
                    "rms_K_s": _rms(np, candidate_rate),
                    "sha256": hashlib.sha256(
                        candidate_rate.tobytes(order="C")
                    ).hexdigest(),
                },
                "direct_step200_delta": {
                    "dry_theta_ring1_rms_K": _rms(np, dry_theta_delta[:, mask]),
                    "moist_theta_ring1_rms_K": _rms(
                        np, moist_theta_delta[:, mask]
                    ),
                },
                "projection": {"anchors": anchors},
                "radiation_only_program": {
                    "lower_seconds": lower_seconds,
                    "compile_seconds": compile_seconds,
                    "dispatch_seconds": dispatch_seconds,
                    "stablehlo_sha256": hashlib.sha256(
                        stablehlo.encode()
                    ).hexdigest(),
                    "stablehlo_bytes": len(stablehlo.encode()),
                    "custom_call_targets": custom_targets,
                    "unknown_custom_call_targets": unknown_targets,
                    "forbidden_tokens": forbidden_tokens,
                },
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "verdict": (
                    "INITIAL_RTHRATEN_STEP200_DIRECT_ORACLE_GREEN"
                    if all(checks.values())
                    else "INITIAL_RTHRATEN_STEP200_DIRECT_ORACLE_RED"
                ),
            }
            proof["proof_sha256"] = common._canonical(proof)
            common._atomic_json(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": proof["verdict"],
                        "proof_sha256": proof["proof_sha256"],
                        "failed_checks": [
                            key for key, value in checks.items() if not value
                        ],
                        "candidate_rthraten": proof["candidate_rthraten"],
                        "direct_step200_delta": proof["direct_step200_delta"],
                        "anchors": anchors,
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
            "schema": "gpuwrf.v0234.initial-rthraten-step200-direct-oracle-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": {key: os.environ.get(key) for key in REQUIRED_ENV},
            "gpu_commands": 0,
            "gpu_queries": 0,
            "proof_sha256": "",
        }
        blocker["proof_sha256"] = common._canonical(blocker)
        common._atomic_json(FAILURE, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
