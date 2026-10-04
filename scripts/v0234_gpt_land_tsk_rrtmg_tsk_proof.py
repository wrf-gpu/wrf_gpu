#!/usr/bin/env python3
"""Propagate the WRF RRTMG-LW top-buffer delta through frozen d03 Noah-MP."""

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
QML_PROOF = SPRINT / "tsk-provenance-proof.json"
QML_DELTA = SCRATCH / "qml-source-delta-v1.npz"
BUFFER_PROOF = SPRINT / "rrtmg-top-buffer-real-wrf-proof.json"
PROVENANCE_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_provenance.py"

QML_PROOF_SHA256 = "140905f7b8998ab4b5a39fa78e2cfc1213e18baccc21479aabad052e2cda26c0"
QML_PROOF_CANONICAL = "c75c4b7ce65b64bd5fc5b2da431fce4de720fe7f6e02a407fe31ce8492cbc5a2"
QML_DELTA_SHA256 = "85dce69d3a9d6a33d699d852ad4779de53a322d96dcd738f7e9a128ae6d8a0a0"
BUFFER_PROOF_SHA256 = "049399d259e3eddbd015359e94c48476e91c409666611b69a0d939b2a9b544a8"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


SPEC = importlib.util.spec_from_file_location("v0234_qml_provenance", PROVENANCE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("v0234 provenance helper unavailable")
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


class BufferTSKFailure(RuntimeError):
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
        raise BufferTSKFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = _git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise BufferTSKFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": provenance.sha256_bytes(disk),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _validate_upstream_proofs() -> dict[str, Any]:
    if provenance.sha256_file(QML_PROOF) != QML_PROOF_SHA256:
        raise BufferTSKFailure("QML_PROOF_HASH")
    qml = json.loads(QML_PROOF.read_text(encoding="utf-8"))
    if not (
        provenance.canonical_without_self(qml) == QML_PROOF_CANONICAL
        and qml.get("canonical_payload_sha256") == QML_PROOF_CANONICAL
        and qml.get("passed") is True
        and qml.get("gates", {}).get(
            "counterfactual_frozen_tsk_land_rms_and_max_strictly_improve"
        )
        is True
    ):
        raise BufferTSKFailure("QML_PROOF_AUTHORITY")
    if provenance.sha256_file(QML_DELTA) != QML_DELTA_SHA256:
        raise BufferTSKFailure("QML_DELTA_HASH")

    if provenance.sha256_file(BUFFER_PROOF) != BUFFER_PROOF_SHA256:
        raise BufferTSKFailure("BUFFER_PROOF_HASH")
    buffer = json.loads(BUFFER_PROOF.read_text(encoding="utf-8"))
    if not (
        buffer.get("status") == "PASS"
        and buffer.get("is_self_compare") is False
        and buffer.get("buffer_source_replay", {}).get("bitwise_exact") is True
        and buffer.get("real_wrf_glw", {}).get("strictly_improves") is True
        and buffer.get("public_layout", {}).get("backward_compatible") is True
    ):
        raise BufferTSKFailure("BUFFER_PROOF_AUTHORITY")
    return {
        "qml_proof_sha256": QML_PROOF_SHA256,
        "qml_proof_canonical_sha256": QML_PROOF_CANONICAL,
        "qml_delta_sha256": QML_DELTA_SHA256,
        "buffer_real_wrf_proof_sha256": BUFFER_PROOF_SHA256,
    }


def _load_qml_delta() -> dict[str, np.ndarray]:
    with np.load(QML_DELTA, allow_pickle=False) as archive:
        expected = {f"{name}_delta" for name in DELTA_FIELDS}
        if set(archive.files) != expected:
            raise BufferTSKFailure("QML_DELTA_SCHEMA")
        values = {
            name: np.asarray(archive[f"{name}_delta"], dtype=np.float64)
            for name in DELTA_FIELDS
        }
    if any(value.shape != (93, 111) or not np.isfinite(value).all() for value in values.values()):
        raise BufferTSKFailure("QML_DELTA_ARRAY")
    return values


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
        upstream = _validate_upstream_proofs()
        script_record = _tracked_record(Path(__file__).resolve())
        rrtmg_record = _tracked_record(REPO / "src/gpuwrf/physics/rrtmg_lw.py")
        coupler_record = _tracked_record(REPO / "src/gpuwrf/coupling/physics_couplers.py")
        qml_delta = _load_qml_delta()

        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
        from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
        from gpuwrf.physics.noah_mp import mavail_from_prescribed_fields, roughness_from_prescribed_fields
        from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing, noahmp_surface_adapter
        from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column
        from gpuwrf.runtime.operational_mode import _NoahMPClock, _NoahMPRadiation

        if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
            raise BufferTSKFailure("JAX_EXECUTION_MODE")

        manifest = json.loads((SEAM / "manifest.json").read_text(encoding="utf-8"))
        seam_state = provenance.load_seam_state(manifest, State, jax)
        with CARRY.open("rb") as stream:
            carry = pickle.load(stream)
        if type(carry).__name__ != "OperationalCarry" or carry.noahmp_rad is None:
            raise BufferTSKFailure("CARRY_SCHEMA")

        run = Gen2Run(INPUTS)
        grid = dataclass_replace(
            run.grid("d03").as_grid_spec(),
            metrics=load_wrfinput_metrics(WRFINPUT),
        )
        if float(grid.vertical.top_pressure_pa) != 5000.0:
            raise BufferTSKFailure(f"GRID_TOP:{grid.vertical.top_pressure_pa}")
        with Dataset(WRFINPUT) as dataset:
            landmask = np.asarray(dataset["LANDMASK"][0], dtype=np.float64)
            lu_index = np.asarray(dataset["LU_INDEX"][0], dtype=np.int32)
            wrf_xland = np.asarray(dataset["XLAND"][0], dtype=np.float64)
        land = wrf_xland < 1.5
        if int(np.sum(land)) != 2034:
            raise BufferTSKFailure("LAND_COUNT")

        winter_z0 = np.asarray(
            roughness_from_prescribed_fields(
                carry.state.xland, landmask, lu_index=lu_index, season=2
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
            roughness_m=jnp.asarray(winter_z0), mavail=jnp.asarray(winter_mavail)
        )
        replay_state = seam_state.replace(**replay_updates)

        _built_land, noahmp_static, _init_meta = build_noahmp_land_state(
            INPUTS, "d03", table_dir=WRF_TABLES
        )
        energy_params, rad_params, _nroot = build_noahmp_params(noahmp_static)
        view = _build_column_view(replay_state, grid)

        # Production column construction supplies exact grid p_top.  The A arm
        # selects the former one-layer branch explicitly, so this is a paired
        # same-input kernel discriminator rather than a different-state replay.
        _sw_state, lw_state, *_rest = _rrtmg_column_inputs(
            replay_state,
            grid,
            lead_seconds=0.0,
            land_state=carry.noahmp_land,
        )
        if lw_state.top_pressure_pa != 5000.0:
            raise BufferTSKFailure("PRODUCTION_PTOP_PLUMBING")
        legacy_glw = solve_rrtmg_lw_column(
            lw_state.replace(top_pressure_pa=None), debug=False
        ).surface_down
        buffer_glw = solve_rrtmg_lw_column(lw_state, debug=False).surface_down
        jax.block_until_ready((legacy_glw, buffer_glw))
        legacy_glw_np = np.asarray(legacy_glw, dtype=np.float64)
        buffer_glw_np = np.asarray(buffer_glw, dtype=np.float64)
        glw_delta = buffer_glw_np - legacy_glw_np

        authenticated_rad = _NoahMPRadiation(*carry.noahmp_rad)
        held_lwdn = np.asarray(authenticated_rad.lwdn, dtype=np.float64)
        legacy_vs_held = provenance.split_metrics(legacy_glw_np, held_lwdn, land)
        if not (
            legacy_vs_held["all"]["rms"] < 2.0e-6
            and legacy_vs_held["all"]["max_abs"] < 1.0e-4
        ):
            raise BufferTSKFailure("LEGACY_HELD_RADIATION_REPLAY")

        # Preserve the authenticated held-radiation base and add only the paired
        # source delta, avoiding the pre-existing ~1e-6 CPU reconstruction floor.
        candidate_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn) + jnp.asarray(glw_delta),
            authenticated_rad.cosz,
        )
        clock = _NoahMPClock(julian=60.0, yearlen=365.0)
        old_forcing = assemble_noahmp_forcing(
            view, noahmp_static, authenticated_rad, clock, 6.0
        )
        new_forcing = assemble_noahmp_forcing(
            view, noahmp_static, candidate_rad, clock, 6.0
        )
        forcing_comparison: dict[str, Any] = {}
        changed_forcing: list[str] = []
        for name in old_forcing._fields:
            old_value = getattr(old_forcing, name)
            new_value = getattr(new_forcing, name)
            if old_value is None or new_value is None:
                if old_value is not new_value:
                    raise BufferTSKFailure(f"FORCING_NONE:{name}")
                continue
            record = provenance.metrics(np.asarray(new_value), np.asarray(old_value))
            forcing_comparison[name] = record
            if record["bitwise_mismatch_count"]:
                changed_forcing.append(name)
        if changed_forcing != ["lwdn"]:
            raise BufferTSKFailure(f"FORCING_AB:{changed_forcing}")

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

        old = replay(old_forcing, authenticated_rad)
        candidate = replay(new_forcing, candidate_rad)
        deltas = {
            name: np.asarray(candidate[name], dtype=np.float64)
            - np.asarray(old[name], dtype=np.float64)
            for name in DELTA_FIELDS
        }
        output_comparison: dict[str, Any] = {}
        for name in candidate:
            record = provenance.split_metrics(candidate[name], old[name], land)
            output_comparison[name] = record
            if record["water"]["bitwise_mismatch_count"] != 0:
                raise BufferTSKFailure(f"WATER_CHANGED:{name}")
            if name not in DELTA_FIELDS and record["all"]["bitwise_mismatch_count"] != 0:
                raise BufferTSKFailure(f"UNAUTHORIZED_SURFACE_CHANGE:{name}")

        wrf_tsk = provenance.load_wrf_tsk()
        qml_counterfactual = (
            np.asarray(seam_state.t_skin, dtype=np.float64) + qml_delta["t_skin"]
        )
        combined_counterfactual = qml_counterfactual + deltas["t_skin"]
        qml_metrics = provenance.split_metrics(qml_counterfactual, wrf_tsk, land)
        combined_metrics = provenance.split_metrics(combined_counterfactual, wrf_tsk, land)
        paired_shift = provenance.split_metrics(candidate["t_skin"], old["t_skin"], land)
        if not (
            qml_metrics["land"]["rms"] == 0.020818276319712437
            and qml_metrics["land"]["max_abs"] == 0.05318624541740746
            and combined_metrics["land"]["rms"] < qml_metrics["land"]["rms"]
            and combined_metrics["land"]["max_abs"] < qml_metrics["land"]["max_abs"]
            and combined_metrics["water"]["bitwise_mismatch_count"] == 0
        ):
            raise BufferTSKFailure("TSK_IMPROVEMENT_GATE")

        delta_artifact = provenance.atomic_npz(delta_output, deltas)
        proof = {
            "schema": "v0234-rrtmg-lw-top-buffer-d03-tsk-v1",
            "verdict": "WRF_RRTMG_LW_TOP_BUFFER_D03_TSK_CANDIDATE_PROVEN",
            "passed": True,
            "git": {
                "head": _git("rev-parse", "HEAD"),
                "status_porcelain": _git("status", "--porcelain", "--untracked-files=no"),
            },
            "execution": {
                "resources": resources,
                "jax_backend": jax.default_backend(),
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
                "production_pbl_adapter_invocations": 0,
                "production_rrtmg_lw_invocations_per_arm": 1,
                "noahmp_replays_per_arm": 1,
            },
            "authority": {
                **upstream,
                "static": static_authority,
                "script": script_record,
                "rrtmg_lw_source": rrtmg_record,
                "production_coupler_source": coupler_record,
                "authenticated_held_lwdn_sha256": provenance.array_sha(held_lwdn),
            },
            "paired_radiation": {
                "only_changed_kernel_metadata": "RRTMGLWColumnState.top_pressure_pa None -> 5000.0",
                "legacy_glw_vs_authenticated_held": legacy_vs_held,
                "candidate_minus_legacy_glw": provenance.split_metrics(
                    buffer_glw_np, legacy_glw_np, land
                ),
                "delta_application": "authenticated held LWDN + paired candidate-minus-legacy GLW",
                "no_empirical_tuning": True,
                "no_mask_selected_correction": True,
            },
            "paired_noahmp": {
                "qair_arm": "source-correct q/(1+q) in both arms",
                "only_changed_forcing_leaf": "lwdn",
                "forcing_comparison": forcing_comparison,
                "surface_output_comparison": output_comparison,
                "paired_tsk_shift": paired_shift,
            },
            "tsk_parity": {
                "qml_only_counterfactual": qml_metrics,
                "qml_plus_rrtmg_buffer_counterfactual": combined_metrics,
                "land_rms_ratio": combined_metrics["land"]["rms"]
                / qml_metrics["land"]["rms"],
                "land_rms_improvement_fraction": 1.0
                - combined_metrics["land"]["rms"] / qml_metrics["land"]["rms"],
                "land_rms_and_max_strictly_improve": True,
                "water_bitwise_invariant": True,
                "remaining_status": "LAND_TSK_NOT_CLOSED_AFTER_RRTMG_TOP_BUFFER",
            },
            "delta_artifact": delta_artifact,
            "delta_contract": {
                "fields": list(DELTA_FIELDS),
                "application": "authenticated seam + accepted QML delta + paired RRTMG-buffer delta",
                "water_delta_bitwise_zero": True,
                "paired_same_inputs_except_lwdn": True,
                "synthetic_wrf_reference": False,
                "empirical_tuning": False,
            },
        }
        provenance.atomic_json(output, proof)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "qml_land_rms": qml_metrics["land"]["rms"],
                    "combined_land_rms": combined_metrics["land"]["rms"],
                    "combined_land_max": combined_metrics["land"]["max_abs"],
                    "delta": delta_artifact,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof must emit a fail-closed reason
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
