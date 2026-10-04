#!/usr/bin/env python3
"""Prove production WRF hydrostatic P3D/P8W and phy_prep T8W in d03 LW/TSK."""

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
BUFFER_TSK_PROOF = SPRINT / "rrtmg-top-buffer-tsk-proof.json"
BUFFER_DELTA = SCRATCH / "rrtmg-buffer-surface-delta-v1.npz"
WRF_FIRST_RK = SCRATCH / "pristine-wrf-source/dyn_em/module_first_rk_step_part1.F"
WRF_PHY_PREP = SCRATCH / "pristine-wrf-source/dyn_em/module_big_step_utilities_em.F"
REAL_WRF_PROOF = SPRINT / "rrtmg-interface-real-wrf-proof.json"

BUFFER_TSK_PROOF_SHA256 = "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a"
BUFFER_DELTA_SHA256 = "f50ac95bb2e22d5e52a12f4fa4634140dd6df0a273a7183b58f167291cb98d0c"
WRF_FIRST_RK_SHA256 = "8c666fe88c46b04e297fe7b7289f55ec74fa133287b234a02f10e05cbbd11841"
WRF_PHY_PREP_SHA256 = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
REAL_WRF_PROOF_SHA256 = "e2a1733ff7a63e4063017ae1a8d5e704bb90e2a7ed1f6446432868c5cf09bc0c"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


SPEC = importlib.util.spec_from_file_location("v0234_rrtmg_tsk_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("v0234 RRTMG TSK proof helper unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
provenance = base.provenance


class InterfaceFailure(RuntimeError):
    """Fail-closed interface discriminator failure."""


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
        raise InterfaceFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = _git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise InterfaceFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": provenance.sha256_bytes(disk),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _load_delta(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    if provenance.sha256_file(path) != expected_sha:
        raise InterfaceFailure(f"DELTA_HASH:{path}")
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"{name}_delta" for name in DELTA_FIELDS}
        if set(archive.files) != expected:
            raise InterfaceFailure(f"DELTA_SCHEMA:{path}")
        values = {
            name: np.asarray(archive[f"{name}_delta"], dtype=np.float64)
            for name in DELTA_FIELDS
        }
    if any(value.shape != (93, 111) or not np.isfinite(value).all() for value in values.values()):
        raise InterfaceFailure(f"DELTA_ARRAY:{path}")
    return values


def _source_gate() -> dict[str, Any]:
    expected = {
        WRF_FIRST_RK: WRF_FIRST_RK_SHA256,
        WRF_PHY_PREP: WRF_PHY_PREP_SHA256,
    }
    records: dict[str, Any] = {}
    for path, digest in expected.items():
        actual = provenance.sha256_file(path)
        if actual != digest:
            raise InterfaceFailure(f"WRF_SOURCE_HASH:{path}:{actual}")
        records[path.name] = {"path": str(path), "sha256": actual}

    first_rk = WRF_FIRST_RK.read_text(encoding="utf-8", errors="strict")
    phy_prep = WRF_PHY_PREP.read_text(encoding="utf-8", errors="strict")
    required_first_rk = (
        "P8W=grid%p_hyd_w",
        "P=grid%p_hyd",
        "T8W=t8w",
        "T=grid%t_phy",
    )
    required_phy_prep = (
        "t8w(i,k,j) = fzm(k)*t_phy(i,k,j)+fzp(k)*t_phy(i,k-1,j)",
        "t8w(i,kde,j) = w1*t_phy(i,kde-1,j)+w2*t_phy(i,kde-2,j)",
        "p_hyd_w(i,kte,j) = p_top",
        "p_hyd_w(i,k,j) = p_hyd_w(i,k+1,j) - (1.+qtot)*(c1(k)*MUT(i,j)+c2(k))*dnw(k)",
        "p_hyd(i,k,j) = 0.5*(p_hyd_w(i,k,j)+p_hyd_w(i,k+1,j))",
    )
    if not all(token in first_rk for token in required_first_rk):
        raise InterfaceFailure("WRF_RADIATION_CALL_SOURCE")
    if not all(token in phy_prep for token in required_phy_prep):
        raise InterfaceFailure("WRF_PHY_PREP_SOURCE")
    return {
        "files": records,
        "radiation_call": list(required_first_rk),
        "phy_prep": list(required_phy_prep),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delta-output", required=True, type=Path)
    parser.add_argument(
        "--candidate-arm",
        choices=("production", "pressure", "temperature"),
        default="production",
        help="Source-faithful interface component to compare with the accepted arm",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    delta_output = args.delta_output.resolve()

    try:
        resources = provenance.resource_gate()
        _manifest_check, static_authority = provenance.validate_static_inputs()
        upstream = base._validate_upstream_proofs()
        source_authority = _source_gate()
        if provenance.sha256_file(BUFFER_TSK_PROOF) != BUFFER_TSK_PROOF_SHA256:
            raise InterfaceFailure("BUFFER_TSK_PROOF_HASH")
        buffer_proof = json.loads(BUFFER_TSK_PROOF.read_text(encoding="utf-8"))
        if not (
            buffer_proof.get("passed") is True
            and buffer_proof.get("tsk_parity", {}).get("land_rms_and_max_strictly_improve") is True
        ):
            raise InterfaceFailure("BUFFER_TSK_PROOF_AUTHORITY")
        qml_delta = base._load_qml_delta()
        buffer_delta = _load_delta(BUFFER_DELTA, BUFFER_DELTA_SHA256)
        script_record = _tracked_record(Path(__file__).resolve())
        rrtmg_record = _tracked_record(REPO / "src/gpuwrf/physics/rrtmg_lw.py")
        coupler_record = _tracked_record(REPO / "src/gpuwrf/coupling/physics_couplers.py")
        if provenance.sha256_file(REAL_WRF_PROOF) != REAL_WRF_PROOF_SHA256:
            raise InterfaceFailure("REAL_WRF_PROOF_HASH")
        real_wrf = json.loads(REAL_WRF_PROOF.read_text(encoding="utf-8"))
        if not (
            real_wrf.get("status") == "PASS"
            and real_wrf.get("is_self_compare") is False
            and real_wrf.get("gates", {}).get("authenticated_p3d_t3d_dz8w_bitwise_replay") is True
            and real_wrf.get("gates", {}).get("real_wrf_glw_strictly_improves") is True
            and real_wrf.get("gates", {}).get("public_layout_unchanged") is True
        ):
            raise InterfaceFailure("REAL_WRF_PROOF_AUTHORITY")

        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
        from gpuwrf.coupling.physics_couplers import (
            _rrtmg_column_inputs,
            _to_columns,
        )
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
        from gpuwrf.physics.noah_mp import mavail_from_prescribed_fields, roughness_from_prescribed_fields
        from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing, noahmp_surface_adapter
        from gpuwrf.physics.rrtmg_lw import (
            _pressure_interfaces,
            _temperature_interfaces,
            solve_rrtmg_lw_column,
        )
        from gpuwrf.runtime.operational_mode import _NoahMPClock, _NoahMPRadiation

        if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
            raise InterfaceFailure("JAX_EXECUTION_MODE")

        manifest = json.loads((SEAM / "manifest.json").read_text(encoding="utf-8"))
        seam_state = provenance.load_seam_state(manifest, State, jax)
        with CARRY.open("rb") as stream:
            carry = pickle.load(stream)
        if type(carry).__name__ != "OperationalCarry" or carry.noahmp_rad is None:
            raise InterfaceFailure("CARRY_SCHEMA")

        run = Gen2Run(INPUTS)
        grid = dataclass_replace(run.grid("d03").as_grid_spec(), metrics=load_wrfinput_metrics(WRFINPUT))
        if float(grid.vertical.top_pressure_pa) != 5000.0 or grid.metrics is None:
            raise InterfaceFailure("GRID_METADATA")
        with Dataset(WRFINPUT) as dataset:
            landmask = np.asarray(dataset["LANDMASK"][0], dtype=np.float64)
            lu_index = np.asarray(dataset["LU_INDEX"][0], dtype=np.int32)
            wrf_xland = np.asarray(dataset["XLAND"][0], dtype=np.float64)
        land = wrf_xland < 1.5
        if int(np.sum(land)) != 2034:
            raise InterfaceFailure("LAND_COUNT")

        winter_z0 = np.asarray(
            roughness_from_prescribed_fields(carry.state.xland, landmask, lu_index=lu_index, season=2)
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
        replay_updates.update(roughness_m=jnp.asarray(winter_z0), mavail=jnp.asarray(winter_mavail))
        replay_state = seam_state.replace(**replay_updates)

        _built_land, noahmp_static, _init_meta = build_noahmp_land_state(
            INPUTS, "d03", table_dir=WRF_TABLES
        )
        energy_params, rad_params, _nroot = build_noahmp_params(noahmp_static)
        view = _build_column_view(replay_state, grid)
        _sw_state, lw_state, *_rest = _rrtmg_column_inputs(
            replay_state, grid, lead_seconds=0.0, land_state=carry.noahmp_land
        )
        if lw_state.top_pressure_pa != 5000.0:
            raise InterfaceFailure("PRODUCTION_PTOP_PLUMBING")

        if lw_state.pressure_interfaces is None or lw_state.temperature_interfaces is None:
            raise InterfaceFailure("PRODUCTION_INTERFACE_PLUMBING")
        # Recreate the immediately preceding accepted top-buffer arm exactly:
        # nonhydrostatic mass p and kernel midpoint P8W/T8W reconstruction.
        # The B arm is the current production state returned by the coupler.
        pre_interface_state = lw_state.replace(
            p=_to_columns(replay_state.p),
            pressure_interfaces=None,
            temperature_interfaces=None,
        )
        if args.candidate_arm == "pressure":
            candidate_state = lw_state.replace(
                temperature_interfaces=_temperature_interfaces(lw_state.T)
            )
        elif args.candidate_arm == "temperature":
            candidate_state = pre_interface_state.replace(
                pressure_interfaces=_pressure_interfaces(pre_interface_state.p),
                temperature_interfaces=lw_state.temperature_interfaces,
            )
        else:
            candidate_state = lw_state
        legacy_glw = solve_rrtmg_lw_column(
            pre_interface_state.replace(top_pressure_pa=None), debug=False
        ).surface_down
        buffer_glw = solve_rrtmg_lw_column(
            pre_interface_state, debug=False
        ).surface_down
        exact_glw = solve_rrtmg_lw_column(candidate_state, debug=False).surface_down
        jax.block_until_ready((legacy_glw, buffer_glw, exact_glw))
        legacy_np = np.asarray(legacy_glw, dtype=np.float64)
        buffer_np = np.asarray(buffer_glw, dtype=np.float64)
        exact_np = np.asarray(exact_glw, dtype=np.float64)

        authenticated_rad = _NoahMPRadiation(*carry.noahmp_rad)
        held_lwdn = np.asarray(authenticated_rad.lwdn, dtype=np.float64)
        legacy_vs_held = provenance.split_metrics(legacy_np, held_lwdn, land)
        if not (
            legacy_vs_held["all"]["rms"] < 2.0e-6
            and legacy_vs_held["all"]["max_abs"] < 1.0e-4
        ):
            raise InterfaceFailure("LEGACY_HELD_RADIATION_REPLAY")

        current_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn) + jnp.asarray(buffer_np - legacy_np),
            authenticated_rad.cosz,
        )
        exact_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn) + jnp.asarray(exact_np - legacy_np),
            authenticated_rad.cosz,
        )
        clock = _NoahMPClock(julian=60.0, yearlen=365.0)
        current_forcing = assemble_noahmp_forcing(view, noahmp_static, current_rad, clock, 6.0)
        exact_forcing = assemble_noahmp_forcing(view, noahmp_static, exact_rad, clock, 6.0)

        changed_forcing: list[str] = []
        forcing_comparison: dict[str, Any] = {}
        for name in current_forcing._fields:
            old_value = getattr(current_forcing, name)
            new_value = getattr(exact_forcing, name)
            if old_value is None or new_value is None:
                if old_value is not new_value:
                    raise InterfaceFailure(f"FORCING_NONE:{name}")
                continue
            record = provenance.metrics(np.asarray(new_value), np.asarray(old_value))
            forcing_comparison[name] = record
            if record["bitwise_mismatch_count"]:
                changed_forcing.append(name)
        if changed_forcing != ["lwdn"]:
            raise InterfaceFailure(f"FORCING_AB:{changed_forcing}")

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
            }

        current = replay(current_forcing, current_rad)
        exact = replay(exact_forcing, exact_rad)
        paired_delta = {
            name: np.asarray(exact[name], dtype=np.float64) - np.asarray(current[name], dtype=np.float64)
            for name in DELTA_FIELDS
        }
        for name, delta in paired_delta.items():
            if np.count_nonzero(delta[~land]) != 0:
                raise InterfaceFailure(f"WATER_CHANGED:{name}")

        # Confirm this process reproduces the already sealed buffer arm before
        # applying the new paired delta to that authenticated accepted state.
        current_vs_old_tsk = replay(
            assemble_noahmp_forcing(view, noahmp_static, authenticated_rad, clock, 6.0),
            authenticated_rad,
        )["t_skin"]
        reproduced_buffer_delta = np.asarray(current["t_skin"], dtype=np.float64) - np.asarray(
            current_vs_old_tsk, dtype=np.float64
        )
        buffer_replay = provenance.metrics(reproduced_buffer_delta, buffer_delta["t_skin"])
        if buffer_replay["rms"] > 5.0e-13 or buffer_replay["max_abs"] > 5.0e-12:
            raise InterfaceFailure(f"BUFFER_DELTA_REPLAY:{buffer_replay}")

        wrf_tsk = provenance.load_wrf_tsk()
        accepted_tsk = (
            np.asarray(seam_state.t_skin, dtype=np.float64)
            + qml_delta["t_skin"]
            + buffer_delta["t_skin"]
        )
        exact_tsk = accepted_tsk + paired_delta["t_skin"]
        accepted_metrics = provenance.split_metrics(accepted_tsk, wrf_tsk, land)
        exact_metrics = provenance.split_metrics(exact_tsk, wrf_tsk, land)
        strictly_improves = (
            exact_metrics["land"]["rms"] < accepted_metrics["land"]["rms"]
            and exact_metrics["land"]["max_abs"] < accepted_metrics["land"]["max_abs"]
        )
        water_invariant = exact_metrics["water"]["bitwise_mismatch_count"] == 0
        if not strictly_improves or not water_invariant:
            raise InterfaceFailure(
                f"TSK_ACCEPTANCE:improves={strictly_improves}:water={water_invariant}"
            )

        p_state_delta = provenance.split_metrics(
            np.moveaxis(np.asarray(candidate_state.p), -1, 0),
            np.moveaxis(np.asarray(pre_interface_state.p), -1, 0),
            land,
        )
        delta_artifact = provenance.atomic_npz(delta_output, paired_delta)
        proof = {
            "schema": f"v0234-rrtmg-wrf-interface-{args.candidate_arm}-tsk-v3",
            "verdict": (
                "WRF_HYDROSTATIC_P8W_P3D_AND_PHY_PREP_T8W_PRODUCTION_TSK_PROVEN"
                if args.candidate_arm == "production"
                else f"WRF_RRTMG_LW_{args.candidate_arm.upper()}_COMPONENT_TSK_PROVEN"
            ),
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
                "proof_only_traced_wrapper": False,
                "production_rrtmg_lw_invocations": 3,
                "noahmp_replays": 3,
            },
            "authority": {
                **upstream,
                "static": static_authority,
                "script": script_record,
                "rrtmg_lw_source": rrtmg_record,
                "production_coupler_source": coupler_record,
                "wrf_source": source_authority,
                "buffer_tsk_proof_sha256": BUFFER_TSK_PROOF_SHA256,
                "buffer_delta_sha256": BUFFER_DELTA_SHA256,
                "real_wrf_interface_proof_sha256": REAL_WRF_PROOF_SHA256,
            },
            "source_boundary": {
                "wrf_mass_pressure": "grid%p_hyd",
                "wrf_pressure_interfaces": "grid%p_hyd_w",
                "wrf_mass_temperature": "grid%t_phy",
                "wrf_temperature_interfaces": "phy_prep t8w",
                "port_before": "state.p plus midpoint-reconstructed pressure/temperature interfaces",
                "production_after": "float32 phy_prep hydrostatic P3D/P8W plus exact fnm/fnp and z-boundary T8W",
                "candidate_arm": args.candidate_arm,
                "candidate_changes": {
                    "production": ["P3D", "P8W", "T8W"],
                    "pressure": ["P3D", "P8W"],
                    "temperature": ["T8W"],
                }[args.candidate_arm],
                "authenticated_real_wrf_input_replay": True,
            },
            "paired_radiation": {
                "legacy_glw_vs_authenticated_held": legacy_vs_held,
                "accepted_buffer_minus_legacy": provenance.split_metrics(buffer_np, legacy_np, land),
                "exact_interfaces_minus_accepted_buffer": provenance.split_metrics(exact_np, buffer_np, land),
                "mass_pressure_hydrostatic_minus_port": p_state_delta,
            },
            "paired_noahmp": {
                "only_changed_forcing_leaf": "lwdn",
                "forcing_comparison": forcing_comparison,
                "surface_delta": {
                    name: provenance.split_metrics(exact[name], current[name], land)
                    for name in DELTA_FIELDS
                },
                "sealed_buffer_delta_replay": buffer_replay,
            },
            "tsk_parity": {
                "accepted_qml_plus_top_buffer": accepted_metrics,
                "candidate_plus_wrf_interfaces": exact_metrics,
                "land_rms_ratio": exact_metrics["land"]["rms"] / accepted_metrics["land"]["rms"],
                "land_rms_improvement_fraction": 1.0
                - exact_metrics["land"]["rms"] / accepted_metrics["land"]["rms"],
                "land_rms_and_max_strictly_improve": strictly_improves,
                "water_bitwise_invariant": water_invariant,
                "remaining_status": "LAND_TSK_NOT_CLOSED_AFTER_WRF_LW_INTERFACES",
            },
            "delta_artifact": delta_artifact,
            "delta_contract": {
                "fields": list(DELTA_FIELDS),
                "application": "accepted seam + QML delta + top-buffer delta + paired production-interface delta",
                "water_delta_bitwise_zero": True,
                "paired_same_inputs_except_lwdn": True,
                "synthetic_wrf_reference": False,
                "empirical_tuning": False,
            },
            "decision": {
                "production_change_authorized": args.candidate_arm == "production",
                "accepted_pending_fresh_sp2_noise_gate": True,
                "no_empirical_tuning": True,
                "no_synthetic_wrf_reference": True,
            },
        }
        provenance.atomic_json(output, proof)
        print(json.dumps({
            "verdict": proof["verdict"],
            "accepted_land_rms": accepted_metrics["land"]["rms"],
            "candidate_land_rms": exact_metrics["land"]["rms"],
            "candidate_land_max": exact_metrics["land"]["max_abs"],
            "glw_delta_land": proof["paired_radiation"]["exact_interfaces_minus_accepted_buffer"]["land"],
        }, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - proof must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
