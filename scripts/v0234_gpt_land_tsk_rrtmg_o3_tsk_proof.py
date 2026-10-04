#!/usr/bin/env python3
"""Propagate exact WRF O3RAD through the frozen d03 LW/Noah-MP TSK seam."""

from __future__ import annotations

import argparse
from dataclasses import replace as dataclass_replace
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

from netCDF4 import Dataset
import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
INPUTS = STAGED / "inputs"
SEAM = STAGED / "seam-capture"
CARRY = INPUTS / "last-healthy-d03-step-0.pkl"
WRFINPUT = INPUTS / "wrfinput_d03"
WRF_TABLES = STAGED / "wrf-source/pristine/phys/noahmp/parameters"
BASE_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_tsk_proof.py"
REAL_WRF_PROOF = SPRINT / "rrtmg-o3-real-wrf-proof.json"
BUFFER_TSK_PROOF = SPRINT / "rrtmg-top-buffer-tsk-proof.json"
BUFFER_DELTA = SCRATCH / "rrtmg-buffer-surface-delta-v1.npz"
REAL_WRF_PROOF_SHA256 = "333d716259c1d012b7db6d7092717611d3c708113ddf247683161872d753e4d9"
BUFFER_TSK_PROOF_SHA256 = "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a"
BUFFER_DELTA_SHA256 = "f50ac95bb2e22d5e52a12f4fa4634140dd6df0a273a7183b58f167291cb98d0c"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"proof helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("v0234_o3_tsk_base", BASE_SCRIPT)
provenance = base.provenance


class O3TSKFailure(RuntimeError):
    """Fail-closed authority or paired-replay failure."""


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
        raise O3TSKFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = _git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise O3TSKFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": provenance.sha256_bytes(disk),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _load_delta(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if provenance.sha256_file(path) != expected_sha256:
        raise O3TSKFailure(f"DELTA_HASH:{path}")
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"{name}_delta" for name in DELTA_FIELDS}
        if set(archive.files) != expected:
            raise O3TSKFailure(f"DELTA_SCHEMA:{path}")
        values = {
            name: np.asarray(archive[f"{name}_delta"], dtype=np.float64)
            for name in DELTA_FIELDS
        }
    if any(
        value.shape != (93, 111) or not np.isfinite(value).all()
        for value in values.values()
    ):
        raise O3TSKFailure(f"DELTA_ARRAY:{path}")
    return values


def _validate_authority() -> dict[str, Any]:
    upstream = base._validate_upstream_proofs()
    if provenance.sha256_file(REAL_WRF_PROOF) != REAL_WRF_PROOF_SHA256:
        raise O3TSKFailure("REAL_WRF_PROOF_HASH")
    real_wrf = json.loads(REAL_WRF_PROOF.read_text(encoding="utf-8"))
    if not (
        real_wrf.get("status") == "PASS"
        and real_wrf.get("is_self_compare") is False
        and real_wrf.get("gates", {}).get("wrf_source_expressions_present") is True
        and real_wrf.get("gates", {}).get("cam_table_extraction_exact") is True
        and real_wrf.get("gates", {}).get("real_wrf_glw_strictly_improves") is True
        and real_wrf.get("gates", {}).get("real_wrf_swdnb_strictly_improves") is True
        and real_wrf.get("gates", {}).get(
            "omitted_leaves_byte_identical_to_9bfe1e30"
        )
        is True
    ):
        raise O3TSKFailure("REAL_WRF_PROOF_AUTHORITY")
    if provenance.sha256_file(BUFFER_TSK_PROOF) != BUFFER_TSK_PROOF_SHA256:
        raise O3TSKFailure("BUFFER_TSK_PROOF_HASH")
    buffer = json.loads(BUFFER_TSK_PROOF.read_text(encoding="utf-8"))
    if not (
        buffer.get("passed") is True
        and buffer.get("tsk_parity", {}).get("land_rms_and_max_strictly_improve")
        is True
        and buffer.get("tsk_parity", {}).get("water_bitwise_invariant") is True
    ):
        raise O3TSKFailure("BUFFER_TSK_PROOF_AUTHORITY")
    return {
        **upstream,
        "o3_real_wrf_proof_sha256": REAL_WRF_PROOF_SHA256,
        "o3_real_wrf_canonical_sha256": real_wrf["canonical_payload_sha256"],
        "buffer_tsk_proof_sha256": BUFFER_TSK_PROOF_SHA256,
        "buffer_delta_sha256": BUFFER_DELTA_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delta-output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    delta_output = args.delta_output.resolve()

    try:
        resources = provenance.resource_gate()
        _manifest_check, static_authority = provenance.validate_static_inputs()
        upstream = _validate_authority()
        qml_delta = base._load_qml_delta()
        buffer_delta = _load_delta(BUFFER_DELTA, BUFFER_DELTA_SHA256)
        tracked = {
            "script": _tracked(Path(__file__).resolve()),
            "cam_ozone": _tracked(REPO / "src/gpuwrf/physics/wrf_cam_ozone.py"),
            "rrtmg_lw": _tracked(REPO / "src/gpuwrf/physics/rrtmg_lw.py"),
            "rrtmg_sw": _tracked(REPO / "src/gpuwrf/physics/rrtmg_sw.py"),
            "coupler": _tracked(REPO / "src/gpuwrf/coupling/physics_couplers.py"),
        }

        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
        from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.io.noahmp_land_init import (
            build_noahmp_land_state,
            build_noahmp_params,
        )
        from gpuwrf.io.radiation_static import load_radiation_static
        from gpuwrf.physics.noah_mp import (
            mavail_from_prescribed_fields,
            roughness_from_prescribed_fields,
        )
        from gpuwrf.physics.noahmp_coupler import (
            assemble_noahmp_forcing,
            noahmp_surface_adapter,
        )
        from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column
        from gpuwrf.runtime.operational_mode import _NoahMPClock, _NoahMPRadiation

        if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
            raise O3TSKFailure("JAX_EXECUTION_MODE")

        manifest = json.loads((SEAM / "manifest.json").read_text(encoding="utf-8"))
        seam_state = provenance.load_seam_state(manifest, State, jax)
        with CARRY.open("rb") as stream:
            carry = pickle.load(stream)
        if type(carry).__name__ != "OperationalCarry" or carry.noahmp_rad is None:
            raise O3TSKFailure("CARRY_SCHEMA")

        run = Gen2Run(INPUTS)
        grid = dataclass_replace(
            run.grid("d03").as_grid_spec(),
            metrics=load_wrfinput_metrics(WRFINPUT),
        )
        if float(grid.vertical.top_pressure_pa) != 5000.0 or grid.metrics is None:
            raise O3TSKFailure("GRID_METADATA")
        radiation_static, radiation_static_meta = load_radiation_static(
            run, "d03", grid=grid, metrics=grid.metrics
        )
        with Dataset(WRFINPUT) as dataset:
            landmask = np.asarray(dataset["LANDMASK"][0], dtype=np.float64)
            lu_index = np.asarray(dataset["LU_INDEX"][0], dtype=np.int32)
            wrf_xland = np.asarray(dataset["XLAND"][0], dtype=np.float64)
            wrf_xlat = np.asarray(dataset["XLAT"][0], dtype=np.float64)
            start_date = str(dataset.getncattr("START_DATE"))
        if not np.array_equal(np.asarray(radiation_static.xlat_deg), wrf_xlat):
            raise O3TSKFailure("RADIATION_STATIC_XLAT")
        if start_date != "2025-03-01_00:00:00":
            raise O3TSKFailure(f"RUN_DATE:{start_date}")
        land = wrf_xland < 1.5
        if int(np.sum(land)) != 2034:
            raise O3TSKFailure("LAND_COUNT")

        winter_z0 = np.asarray(
            roughness_from_prescribed_fields(
                carry.state.xland,
                landmask,
                lu_index=lu_index,
                season=2,
            )
        )
        winter_mavail = np.asarray(
            mavail_from_prescribed_fields(
                carry.state.xland,
                landmask,
                carry.state.soil_moisture,
                lu_index=lu_index,
                season=2,
            )
        )
        replay_updates = {
            name: getattr(carry.state, name)
            for name in ("t_skin", "soil_moisture", "xland", "lakemask", "ustar")
        }
        replay_updates.update(
            roughness_m=jnp.asarray(winter_z0),
            mavail=jnp.asarray(winter_mavail),
        )
        replay_state = seam_state.replace(**replay_updates)

        _built_land, noahmp_static, _init_meta = build_noahmp_land_state(
            INPUTS, "d03", table_dir=WRF_TABLES
        )
        energy_params, rad_params, _nroot = build_noahmp_params(noahmp_static)
        view = _build_column_view(replay_state, grid)
        sw_state, lw_state, *_rest, geometry, _topography = _rrtmg_column_inputs(
            replay_state,
            grid,
            lead_seconds=0.0,
            time_utc=start_date,
            land_state=carry.noahmp_land,
            radiation_static=radiation_static,
        )
        if lw_state.top_pressure_pa != 5000.0 or lw_state.ozone_vmr is None:
            raise O3TSKFailure("PRODUCTION_O3_PLUMBING")
        if sw_state.ozone_vmr is None or not np.array_equal(
            np.asarray(lw_state.ozone_vmr), np.asarray(sw_state.ozone_vmr)
        ):
            raise O3TSKFailure("LW_SW_O3_DISAGREE")
        if any(
            value is not None
            for value in (
                lw_state.pressure_interfaces,
                lw_state.temperature_interfaces,
                lw_state.co2_vmr,
                sw_state.pressure_interfaces,
                sw_state.temperature_interfaces,
                sw_state.co2_vmr,
            )
        ):
            raise O3TSKFailure("REJECTED_RADIATION_LEAF_REPOPULATED")
        coszen = np.asarray(geometry.coszen)
        if np.any(coszen > 0.0):
            raise O3TSKFailure("D03_NOT_NIGHT")

        annual_state = lw_state.replace(ozone_vmr=None)
        legacy_state = annual_state.replace(top_pressure_pa=None)
        legacy_glw = solve_rrtmg_lw_column(legacy_state, debug=False).surface_down
        accepted_glw = solve_rrtmg_lw_column(annual_state, debug=False).surface_down
        o3_glw = solve_rrtmg_lw_column(lw_state, debug=False).surface_down
        jax.block_until_ready((legacy_glw, accepted_glw, o3_glw))
        legacy_np = np.asarray(legacy_glw, dtype=np.float64)
        accepted_np = np.asarray(accepted_glw, dtype=np.float64)
        o3_np = np.asarray(o3_glw, dtype=np.float64)

        authenticated_rad = _NoahMPRadiation(*carry.noahmp_rad)
        held_lwdn = np.asarray(authenticated_rad.lwdn, dtype=np.float64)
        legacy_vs_held = provenance.split_metrics(legacy_np, held_lwdn, land)
        if not (
            legacy_vs_held["all"]["rms"] < 2.0e-6
            and legacy_vs_held["all"]["max_abs"] < 1.0e-4
        ):
            raise O3TSKFailure("LEGACY_HELD_RADIATION_REPLAY")
        accepted_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn)
            + jnp.asarray(accepted_np - legacy_np),
            authenticated_rad.cosz,
        )
        o3_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn) + jnp.asarray(o3_np - legacy_np),
            authenticated_rad.cosz,
        )
        clock = _NoahMPClock(julian=60.0, yearlen=365.0)
        accepted_forcing = assemble_noahmp_forcing(
            view, noahmp_static, accepted_rad, clock, 6.0
        )
        o3_forcing = assemble_noahmp_forcing(
            view, noahmp_static, o3_rad, clock, 6.0
        )

        changed_forcing: list[str] = []
        forcing_comparison: dict[str, Any] = {}
        for name in accepted_forcing._fields:
            old_value = getattr(accepted_forcing, name)
            new_value = getattr(o3_forcing, name)
            if old_value is None or new_value is None:
                if old_value is not new_value:
                    raise O3TSKFailure(f"FORCING_NONE:{name}")
                continue
            comparison = provenance.metrics(
                np.asarray(new_value), np.asarray(old_value)
            )
            forcing_comparison[name] = comparison
            if comparison["bitwise_mismatch_count"]:
                changed_forcing.append(name)
        if changed_forcing != ["lwdn"]:
            raise O3TSKFailure(f"FORCING_AB:{changed_forcing}")

        def replay(forcing: Any, radiation: Any) -> dict[str, np.ndarray]:
            view_out, _land_out, blended = noahmp_surface_adapter(
                view,
                carry.noahmp_land,
                noahmp_static,
                radiation=radiation,
                clock=clock,
                dt=6.0,
                forcing=forcing,
                energy_params=energy_params,
                rad_params=rad_params,
                first_timestep=True,
            )
            return {
                "theta_flux": np.asarray(blended.theta_flux),
                "qv_flux": np.asarray(blended.qv_flux),
                "fltv": np.asarray(blended.fltv),
                "t_skin": np.asarray(view_out.t_skin),
                "ustar": np.asarray(blended.ustar),
                "tau_u": np.asarray(blended.tau_u),
                "tau_v": np.asarray(blended.tau_v),
                "rhosfc": np.asarray(blended.rhosfc),
                "roughness_m": np.asarray(view_out.roughness_m),
            }

        held_forcing = assemble_noahmp_forcing(
            view, noahmp_static, authenticated_rad, clock, 6.0
        )
        held = replay(held_forcing, authenticated_rad)
        accepted = replay(accepted_forcing, accepted_rad)
        candidate = replay(o3_forcing, o3_rad)
        reproduced_buffer_delta = {
            name: np.asarray(accepted[name], dtype=np.float64)
            - np.asarray(held[name], dtype=np.float64)
            for name in DELTA_FIELDS
        }
        buffer_replay = {
            name: provenance.metrics(reproduced_buffer_delta[name], buffer_delta[name])
            for name in DELTA_FIELDS
        }
        if any(
            comparison["rms"] > 5.0e-13 or comparison["max_abs"] > 5.0e-12
            for comparison in buffer_replay.values()
        ):
            raise O3TSKFailure(f"BUFFER_DELTA_REPLAY:{buffer_replay}")

        paired_delta = {
            name: np.asarray(candidate[name], dtype=np.float64)
            - np.asarray(accepted[name], dtype=np.float64)
            for name in DELTA_FIELDS
        }
        output_comparison: dict[str, Any] = {}
        for name in candidate:
            comparison = provenance.split_metrics(candidate[name], accepted[name], land)
            output_comparison[name] = comparison
            if comparison["water"]["bitwise_mismatch_count"] != 0:
                raise O3TSKFailure(f"WATER_CHANGED:{name}")
            if (
                name not in DELTA_FIELDS
                and comparison["all"]["bitwise_mismatch_count"] != 0
            ):
                raise O3TSKFailure(f"UNAUTHORIZED_SURFACE_CHANGE:{name}")

        wrf_tsk = provenance.load_wrf_tsk()
        accepted_tsk = (
            np.asarray(seam_state.t_skin, dtype=np.float64)
            + qml_delta["t_skin"]
            + buffer_delta["t_skin"]
        )
        candidate_tsk = accepted_tsk + paired_delta["t_skin"]
        accepted_metrics = provenance.split_metrics(accepted_tsk, wrf_tsk, land)
        candidate_metrics = provenance.split_metrics(candidate_tsk, wrf_tsk, land)
        strict_improvement = bool(
            candidate_metrics["land"]["rms"] < accepted_metrics["land"]["rms"]
            and candidate_metrics["land"]["max_abs"]
            < accepted_metrics["land"]["max_abs"]
        )
        water_invariant = candidate_metrics["water"]["bitwise_mismatch_count"] == 0
        if not (
            accepted_metrics["land"]["rms"] == 0.018808123113575367
            and accepted_metrics["land"]["max_abs"] == 0.05064729526111478
            and strict_improvement
            and water_invariant
        ):
            raise O3TSKFailure(
                f"TSK_GATE:{accepted_metrics['land']}:{candidate_metrics['land']}"
            )

        delta_artifact = provenance.atomic_npz(delta_output, paired_delta)
        proof = {
            "schema": "v0234-wrf-o3rad-d03-tsk-v1",
            "verdict": "WRF_O3RAD_STRICTLY_IMPROVES_D03_LAND_TSK",
            "passed": True,
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status_porcelain": _git(
                    "status", "--porcelain", "--untracked-files=no"
                ),
            },
            "execution": {
                "resources": resources,
                "jax_backend": jax.default_backend(),
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
                "production_rrtmg_lw_invocations": 3,
                "production_rrtmg_sw_invocations": 0,
                "noahmp_replays": 3,
            },
            "authority": {
                **upstream,
                "static": static_authority,
                "tracked": tracked,
                "radiation_static": radiation_static_meta,
                "run_date": start_date,
                "authenticated_held_lwdn_sha256": provenance.array_sha(held_lwdn),
            },
            "production_composition": {
                "accepted": "QML + static LW top pressure",
                "candidate": "QML + static LW top pressure + exact WRF O3RAD on LW and SW",
                "dynamic_p3d_p8w_t8w_population": False,
                "run_date_ghg_population": False,
                "lw_sw_o3rad_bitwise_same": True,
                "o3rad_payload_sha256": provenance.array_sha(
                    np.asarray(lw_state.ozone_vmr)
                ),
                "o3rad_shape": list(lw_state.ozone_vmr.shape),
            },
            "paired_radiation": {
                "only_changed_leaf": "RRTMGLWColumnState.ozone_vmr",
                "legacy_glw_vs_authenticated_held": legacy_vs_held,
                "accepted_buffer_minus_legacy": provenance.split_metrics(
                    accepted_np, legacy_np, land
                ),
                "o3rad_minus_accepted_buffer": provenance.split_metrics(
                    o3_np, accepted_np, land
                ),
                "delta_application": "held LWDN + paired solver deltas",
            },
            "shortwave_night_branch": {
                "coszen_max": float(np.max(coszen)),
                "coszen_positive_count": int(np.count_nonzero(coszen > 0.0)),
                "soldn_held_in_both_arms": True,
                "o3rad_populated_on_sw_state": True,
                "wrf_exact_no_sw_solver_invocation": True,
            },
            "paired_noahmp": {
                "only_changed_forcing_leaf": "lwdn",
                "forcing_comparison": forcing_comparison,
                "surface_output_comparison": output_comparison,
                "sealed_buffer_delta_replay": buffer_replay,
            },
            "tsk_parity": {
                "accepted_qml_plus_static_top": accepted_metrics,
                "candidate_qml_plus_static_top_plus_o3rad": candidate_metrics,
                "land_rms_ratio": candidate_metrics["land"]["rms"]
                / accepted_metrics["land"]["rms"],
                "land_rms_improvement_fraction": 1.0
                - candidate_metrics["land"]["rms"]
                / accepted_metrics["land"]["rms"],
                "land_rms_and_max_strictly_improve": strict_improvement,
                "water_bitwise_invariant": water_invariant,
                "remaining_status": "LAND_TSK_NOT_CLOSED_AFTER_EXACT_WRF_O3RAD",
            },
            "delta_artifact": delta_artifact,
            "delta_contract": {
                "fields": list(DELTA_FIELDS),
                "application": "accepted seam + QML + static-top + paired O3RAD delta",
                "water_delta_bitwise_zero": True,
                "paired_same_inputs_except_lwdn": True,
                "synthetic_wrf_reference": False,
                "empirical_tuning": False,
            },
            "gates": {
                "source_and_real_wrf_o3_proof_pinned": True,
                "production_composition_exact": True,
                "land_tsk_rms_and_max_strictly_improve": strict_improvement,
                "water_bitwise_invariant": water_invariant,
                "shortwave_night_branch_exact": True,
                "ready_for_fresh_sp2_noise_gate": True,
            },
        }
        provenance.atomic_json(output, proof)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "accepted_land_rms": accepted_metrics["land"]["rms"],
                    "candidate_land_rms": candidate_metrics["land"]["rms"],
                    "candidate_land_max": candidate_metrics["land"]["max_abs"],
                    "candidate_land_bias": candidate_metrics["land"]["signed_bias"],
                    "delta": delta_artifact,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
