"""Authenticated real-Step0 oracle for WRF stage-omega transport ownership."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from scripts import v0234_original_moist_sumflux_cpu_ab as frame_support
from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "stage-omega-transport-source-oracle.json"
FAILURE = SPRINT / "stage-omega-transport-source-oracle-blocker.json"
REQUIRED_ENV = {**common.REQUIRED_ENV, "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false"}


def _band(np, ny: int, nx: int):
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))


def main() -> int:
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if common._sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        effective_async = jax.config.values.get("jax_cpu_enable_async_dispatch")
        import jax.numpy as jnp
        import numpy as np

        affinity = sorted(os.sched_getaffinity(0))
        if effective_async is not False or jax.default_backend() != "cpu" or affinity != [12]:
            raise RuntimeError(
                f"runtime binding mismatch async={effective_async} "
                f"backend={jax.default_backend()} affinity={affinity}"
            )
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from gpuwrf.dynamics.flux_advection import advect_scalar_flux
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-stage-omega-transport-oracle-"))
        try:
            tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
                scratch
            )
            if names != ("d01", "d02", "d03") or dt_by_domain != {
                "d01": 54.0,
                "d02": 18.0,
                "d03": 6.0,
            }:
                raise RuntimeError("canonical hierarchy changed")
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            state = carry.state
            namelist = tree.domains["d03"].namelist
            if not runtime._specified_adv_degrade_active(namelist):
                raise RuntimeError("authenticated d03 specified/nested transport gate inactive")

            selector = runtime._stage_transport_omega_ownership_enabled
            runtime._stage_transport_omega_ownership_enabled = lambda: False
            try:
                retained_vel = runtime._stage_transport_velocities(state, namelist)
            finally:
                runtime._stage_transport_omega_ownership_enabled = selector
            candidate_vel = runtime._stage_transport_velocities(state, namelist)

            metrics = namelist.metrics
            theta_pert = state.theta - runtime._theta_base_offset(state.theta)
            kwargs = dict(
                mut=state.mu_total,
                c1=metrics.c1h,
                rdx=1.0 / float(namelist.grid.projection.dx_m),
                rdy=1.0 / float(namelist.grid.projection.dy_m),
                rdzw=metrics.rdnw,
                fzm=metrics.fnm,
                fzp=metrics.fnp,
            )
            retained_tendency = advect_scalar_flux(theta_pert, retained_vel, **kwargs)
            candidate_tendency = advect_scalar_flux(theta_pert, candidate_vel, **kwargs)

            old_rom = np.asarray(retained_vel.rom, dtype=np.float64)
            new_rom = np.asarray(candidate_vel.rom, dtype=np.float64)
            rom_delta = new_rom - old_rom
            tendency_delta = np.asarray(
                candidate_tendency - retained_tendency, dtype=np.float64
            )
            mass_h = (
                np.asarray(metrics.c1h, dtype=np.float64)[:, None, None]
                * np.asarray(state.mu_total, dtype=np.float64)[None]
                + np.asarray(metrics.c2h, dtype=np.float64)[:, None, None]
            )
            direct_theta_delta = float(namelist.dt_s) * tendency_delta / mass_h
            band = _band(np, *direct_theta_delta.shape[-2:])

            # The vertical flux difference has zero top and bottom flux.  In eta
            # coordinates sum(tendency/rdnw) is therefore the telescoping column
            # conservation identity independently of the field values.
            weighted_column_residual = np.sum(
                tendency_delta
                / np.asarray(metrics.rdnw, dtype=np.float64)[:, None, None],
                axis=0,
            )

            frames, frame_authority = frame_support._load_frames(np)
            current = frames["current_395"]
            projected_thm = current["THM"] + direct_theta_delta
            projected_t = (projected_thm + 300.0) / (
                1.0
                + (461.6 / 287.0)
                * np.maximum(current["QVAPOR"], 0.0)
            ) - 300.0
            ring1 = band == 1
            projection = {}
            for anchor in ("cpu_wrf", "retry20"):
                projection[anchor] = {
                    "THM_current_ring1_rmse_K": common._rms(
                        np, current["THM"][:, ring1] - frames[anchor]["THM"][:, ring1]
                    ),
                    "THM_direct_source_projected_ring1_rmse_K": common._rms(
                        np, projected_thm[:, ring1] - frames[anchor]["THM"][:, ring1]
                    ),
                    "T_current_ring1_rmse_K": common._rms(
                        np, current["T"][:, ring1] - frames[anchor]["T"][:, ring1]
                    ),
                    "T_direct_source_projected_ring1_rmse_K": common._rms(
                        np, projected_t[:, ring1] - frames[anchor]["T"][:, ring1]
                    ),
                }

            checks = {
                "fresh_synchronous_cpu12": effective_async is False and affinity == [12],
                "authenticated_step0_hierarchy_and_frames": True,
                "candidate_rom_changed": bool(np.any(rom_delta != 0.0)),
                "candidate_theta_vertical_source_changed": bool(
                    np.any(tendency_delta != 0.0)
                ),
                "top_bottom_rom_zero_both": bool(
                    np.all(old_rom[[0, -1]] == 0.0)
                    and np.all(new_rom[[0, -1]] == 0.0)
                ),
                "weighted_vertical_conservation": bool(
                    np.max(np.abs(weighted_column_residual)) < 5.0e-9
                ),
                "all_finite": bool(
                    np.all(np.isfinite(new_rom))
                    and np.all(np.isfinite(tendency_delta))
                    and np.all(np.isfinite(direct_theta_delta))
                ),
            }
            proof = {
                "schema": "gpuwrf.v0234.stage-omega-transport-source-oracle.v1",
                "environment": actual_env,
                "effective_runtime": {
                    "backend": jax.default_backend(),
                    "jax_cpu_enable_async_dispatch": effective_async,
                    "cpu_affinity": affinity,
                    "fresh_arm": True,
                },
                "input": {
                    "path": str(common.STEP0),
                    "sha256": common.STEP0_SHA256,
                    "completed_step": 0,
                    "leaf_count": 106,
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "measurements": {
                    "rom_delta_rms": common._rms(np, rom_delta),
                    "rom_delta_maxabs": float(np.max(np.abs(rom_delta))),
                    "rom_delta_ring0_rms": common._rms(np, rom_delta[:, band == 0]),
                    "rom_delta_ring1_rms": common._rms(np, rom_delta[:, band == 1]),
                    "rom_delta_interior_ge10_rms": common._rms(
                        np, rom_delta[:, band >= 10]
                    ),
                    "theta_tendency_delta_rms_coupled_per_s": common._rms(
                        np, tendency_delta
                    ),
                    "direct_theta_delta_ring1_rms_K": common._rms(
                        np, direct_theta_delta[:, ring1]
                    ),
                    "direct_theta_delta_interior_ge10_rms_K": common._rms(
                        np, direct_theta_delta[:, band >= 10]
                    ),
                    "weighted_column_conservation_maxabs": float(
                        np.max(np.abs(weighted_column_residual))
                    ),
                },
                "direct_source_projection_not_final_gate": projection,
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "verdict": (
                    "STAGE_OMEGA_TRANSPORT_SOURCE_ORACLE_GREEN"
                    if all(checks.values())
                    else "STAGE_OMEGA_TRANSPORT_SOURCE_ORACLE_RED"
                ),
            }
            proof["proof_sha256"] = common._canonical(proof)
            common._atomic_json(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": proof["verdict"],
                        "proof_sha256": proof["proof_sha256"],
                        "failed_checks": [k for k, v in checks.items() if not v],
                        "measurements": proof["measurements"],
                        "projection": projection,
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
            "schema": "gpuwrf.v0234.stage-omega-transport-source-oracle-blocker.v1",
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
