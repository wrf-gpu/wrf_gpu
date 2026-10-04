#!/usr/bin/env python3
"""Authenticated real-WRF GLW A/B for run-date CAM greenhouse gases."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
REAL_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_interface_real_wrf_proof.py"
CAM_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_co2_year.py"
INTERFACE_PROOF = SPRINT / "rrtmg-interface-real-wrf-proof.json"
INTERFACE_PROOF_SHA256 = "e2a1733ff7a63e4063017ae1a8d5e704bb90e2a7ed1f6446432868c5cf09bc0c"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"proof helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("v0234_rrtmg_interface_real", REAL_SCRIPT)
cam = _load("v0234_rrtmg_cam_tsk", CAM_SCRIPT)


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
        raise RuntimeError(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = _git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise RuntimeError(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": cam.provenance.sha256_bytes(disk),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _interface_authority() -> dict[str, Any]:
    actual = cam.provenance.sha256_file(INTERFACE_PROOF)
    if actual != INTERFACE_PROOF_SHA256:
        raise RuntimeError(f"INTERFACE_PROOF_HASH:{actual}")
    proof = json.loads(INTERFACE_PROOF.read_text(encoding="utf-8"))
    if not (
        proof.get("status") == "PASS"
        and proof.get("is_self_compare") is False
        and proof.get("gates", {}).get("real_wrf_glw_strictly_improves") is True
        and proof.get("gates", {}).get(
            "authenticated_p3d_t3d_dz8w_bitwise_replay"
        )
        is True
    ):
        raise RuntimeError("INTERFACE_PROOF_AUTHORITY")
    return {
        "path": str(INTERFACE_PROOF),
        "sha256": actual,
        "sealed_dynamic_glw": proof["real_wrf_glw"]["wrf_p8w_t8w_interfaces"],
        "sealed_precomposition_glw": proof["real_wrf_glw"][
            "accepted_top_buffer_midpoint_interfaces"
        ],
    }


def _run(
    oracle_dir: Path,
    wrfinput: Path,
    wrf_root: Path,
    output: Path,
) -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree is not clean")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX_EXECUTION_MODE")
    script_record = _tracked_record(Path(__file__).resolve())
    cam_script_record = _tracked_record(CAM_SCRIPT)
    rrtmg_record = _tracked_record(REPO / "src/gpuwrf/physics/rrtmg_lw.py")
    constants_record = _tracked_record(REPO / "src/gpuwrf/physics/rrtmg_constants.py")
    clwrf_record = _tracked_record(REPO / "src/gpuwrf/physics/wrf_clwrf_ghg.py")
    interface_authority = _interface_authority()

    manifest, fields, oracle_hashes = base._load_oracle(oracle_dir)
    interface_source = base._source_gate(
        wrf_root / "dyn_em/module_first_rk_step_part1.F",
        wrf_root / "dyn_em/module_big_step_utilities_em.F",
    )
    p8w_zyx, t8w_zyx, reconstructed = base._wrf_interface_inputs(wrfinput, fields)

    with Dataset(wrfinput) as dataset:
        start_date = str(dataset.START_DATE)
        ghg_input = int(dataset.GHG_INPUT)
        julyr = int(dataset.JULYR)
        julday = int(dataset.JULDAY)
    timestamp = datetime.strptime(start_date, "%Y-%m-%d_%H:%M:%S")
    target_julian = float(timestamp.timetuple().tm_yday) + (
        timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    ) / 86400.0
    if not (
        ghg_input == 1
        and julyr == timestamp.year == 2026
        and julday == timestamp.timetuple().tm_yday == 118
        and target_julian == 118.75
    ):
        raise RuntimeError(
            f"SOURCE_RUN_CLOCK:{start_date}:{ghg_input}:{julyr}:{julday}:{target_julian}"
        )
    target_cam = cam._cam_gases(julyr, target_julian)

    def col3(name: str) -> jax.Array:
        array = fields[("in", name)]
        nj, nk, ni = array.shape
        return jnp.asarray(
            np.moveaxis(array, 1, 2).reshape(nj * ni, nk), dtype=jnp.float64
        )

    def surface(tag: str, name: str) -> jax.Array:
        return jnp.asarray(fields[(tag, name)].reshape(-1), dtype=jnp.float64)

    T = col3("t")
    p = col3("p")
    qv = col3("qv")
    qc = col3("qc")
    qi = col3("qi")
    qs = col3("qs")
    zero = jnp.zeros_like(qv)
    common = dict(
        T=T,
        p=p,
        qv=qv,
        qc=qc,
        qi=qi,
        qs=qs,
        qg=zero,
        cloud_fraction=col3("cldfra"),
        surface_temperature=surface("in", "tsk"),
        surface_emissivity=surface("in", "emiss"),
        dz=col3("dz8w"),
        rho=col3("rho"),
        top_pressure_pa=5000.0,
    )
    p8w_columns = np.moveaxis(p8w_zyx, 0, -1).reshape(-1, p8w_zyx.shape[0])
    t8w_columns = np.moveaxis(t8w_zyx, 0, -1).reshape(-1, t8w_zyx.shape[0])
    state = base.RRTMGLWColumnState(
        **common,
        pressure_interfaces=jnp.asarray(p8w_columns, dtype=jnp.float64),
        temperature_interfaces=jnp.asarray(t8w_columns, dtype=jnp.float64),
    )

    import gpuwrf.physics.rrtmg_lw as rrtmg_lw
    from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time

    original_cfc = np.asarray(rrtmg_lw._CFC_VMR, dtype=np.float64)
    original_gases = {
        "co2": float(rrtmg_lw.CO2_VMR),
        "n2o": float(rrtmg_lw.N2O_VMR),
        "ch4": float(rrtmg_lw.CH4_VMR),
        "cfc11": float(original_cfc[1]),
        "cfc12": float(original_cfc[2]),
    }
    target_gases = target_cam["vmr"]
    production_gases = clwrf_ssp245_gases_for_time(start_date)
    production_gas_dict = {
        "co2": production_gases.co2_vmr,
        "n2o": production_gases.n2o_vmr,
        "ch4": production_gases.ch4_vmr,
        "cfc11": production_gases.cfc11_vmr,
        "cfc12": production_gases.cfc12_vmr,
    }
    if production_gas_dict != target_gases:
        raise RuntimeError(
            f"PRODUCTION_CLWRF_REGRESSION:{production_gas_dict}:{target_gases}"
        )
    candidate_state = state.replace(
        co2_vmr=production_gases.co2_vmr,
        n2o_vmr=production_gases.n2o_vmr,
        ch4_vmr=production_gases.ch4_vmr,
        cfc11_vmr=production_gases.cfc11_vmr,
        cfc12_vmr=production_gases.cfc12_vmr,
    )

    base.solve_rrtmg_lw_column.clear_cache()
    accepted = base.solve_rrtmg_lw_column(state, debug=False)
    jax.block_until_ready((accepted.surface_down, accepted.heating_rate))
    candidate = base.solve_rrtmg_lw_column(candidate_state, debug=False)
    jax.block_until_ready((candidate.surface_down, candidate.heating_rate))

    glw_reference = np.asarray(surface("out", "glw"))
    pi3d = np.asarray(col3("pi3d"))
    rthraten = fields[("out", "rthratenlw")]
    nj, nk, ni = rthraten.shape
    heating_reference = np.moveaxis(rthraten, 1, 2).reshape(nj * ni, nk) * pi3d
    accepted_glw = base._metrics(np.asarray(accepted.surface_down), glw_reference)
    candidate_glw = base._metrics(np.asarray(candidate.surface_down), glw_reference)
    accepted_heating = base._metrics(
        np.asarray(accepted.heating_rate), heating_reference
    )
    candidate_heating = base._metrics(
        np.asarray(candidate.heating_rate), heating_reference
    )
    sealed = interface_authority["sealed_dynamic_glw"]
    if not (
        accepted_glw["rms"] == sealed["rms"]
        and accepted_glw["max_abs"] == sealed["max_abs"]
        and accepted_glw["bias"] == sealed["bias"]
    ):
        raise RuntimeError(f"SEALED_INTERFACE_REPLAY:{accepted_glw}:{sealed}")
    incremental_improves = candidate_glw["rms"] < accepted_glw["rms"]
    precomposition_glw = interface_authority["sealed_precomposition_glw"]
    composition_improves = candidate_glw["rms"] < precomposition_glw["rms"]

    record = {
        "schema": "v0234-rrtmg-lw-wrf-clwrf-production-real-wrf-ab-v1",
        "status": "PASS",
        "verdict": (
            "WRF_CLWRF_FULL_COMPOSITION_STRICTLY_IMPROVES_REAL_WRF_GLW"
            if composition_improves
            else "WRF_CLWRF_FULL_COMPOSITION_REAL_WRF_GLW_FALSIFIED"
        ),
        "candidate_strictly_improves_targeted_glw": composition_improves,
        "is_self_compare": False,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "status_porcelain": _git(
                "status", "--porcelain", "--untracked-files=no"
            ),
        },
        "execution": {
            "platform": jax.default_backend(),
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_rrtmg_invocations_per_arm": 1,
        },
        "authority": {
            "script": script_record,
            "cam_tsk_script": cam_script_record,
            "rrtmg_lw_source": rrtmg_record,
            "rrtmg_constants_source": constants_record,
            "production_clwrf_interpolation_source": clwrf_record,
            "oracle_dir": str(oracle_dir),
            "oracle_manifest_sha256": base.EXPECTED["oracle_manifest"],
            "oracle_file_sha256": oracle_hashes,
            "source_run": manifest.get("source_run"),
            "physics_options": manifest.get("physics_options"),
            "wrf_interface_source": interface_source,
            "interface_reconstruction": reconstructed,
            "sealed_interface_proof": interface_authority,
            "wrfinput": {
                "path": str(wrfinput),
                "sha256": base.EXPECTED["wrfinput_d01"],
                "start_date": start_date,
                "ghg_input": ghg_input,
                "julyr": julyr,
                "julday": julday,
                "fractional_julian": target_julian,
            },
            "clwrf_cam_interpolation": target_cam,
        },
        "paired_radiation": {
            "same_authenticated_wrf_column_inputs": True,
            "same_hydrostatic_p3d_p8w_phy_prep_t8w": True,
            "only_changed_inputs": [
                "RRTMGLWColumnState.co2_vmr",
                "RRTMGLWColumnState.n2o_vmr",
                "RRTMGLWColumnState.ch4_vmr",
                "RRTMGLWColumnState.cfc11_vmr",
                "RRTMGLWColumnState.cfc12_vmr",
            ],
            "production_static_metadata_path": True,
            "accepted_constant_gases": original_gases,
            "candidate_wrf_cam_gases": target_gases,
            "candidate_minus_accepted_glw": base._metrics(
                np.asarray(candidate.surface_down), np.asarray(accepted.surface_down)
            ),
        },
        "real_wrf_glw": {
            "accepted_precomposition_static_interfaces_constant_gases": (
                precomposition_glw
            ),
            "accepted_dynamic_interfaces_constant_gases": accepted_glw,
            "candidate_dynamic_interfaces_cam_gases": candidate_glw,
            "gas_increment_rms_ratio": candidate_glw["rms"] / accepted_glw["rms"],
            "gas_increment_strictly_improves": incremental_improves,
            "full_composition_rms_ratio": (
                candidate_glw["rms"] / precomposition_glw["rms"]
            ),
            "full_composition_improvement_fraction": (
                1.0 - candidate_glw["rms"] / precomposition_glw["rms"]
            ),
            "full_composition_strictly_improves": composition_improves,
        },
        "real_wrf_lw_heating": {
            "accepted_dynamic_interfaces_constant_gases": accepted_heating,
            "candidate_dynamic_interfaces_cam_gases": candidate_heating,
            "rms_ratio": candidate_heating["rms"] / accepted_heating["rms"],
        },
        "gates": {
            "authenticated_real_wrf_reference": True,
            "ghg_input_one_selects_cam_branch": True,
            "ssp245_link_and_payload_pinned": True,
            "sealed_dynamic_interface_arm_exactly_replayed": True,
            "gas_increment_real_wrf_glw_strictly_improves": incremental_improves,
            "full_composition_real_wrf_glw_strictly_improves": (
                composition_improves
            ),
            "no_empirical_tuning": True,
            "no_synthetic_reference": True,
        },
    }
    cam.provenance.atomic_json(output, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--wrfinput", required=True, type=Path)
    parser.add_argument("--wrf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        record = _run(
            args.oracle_dir.resolve(),
            args.wrfinput.resolve(),
            args.wrf_root.resolve(),
            args.output.resolve(),
        )
        print(
            json.dumps(
                {
                    "verdict": record["verdict"],
                    "accepted_glw_rms": record["real_wrf_glw"][
                        "accepted_dynamic_interfaces_constant_gases"
                    ]["rms"],
                    "candidate_glw_rms": record["real_wrf_glw"][
                        "candidate_dynamic_interfaces_cam_gases"
                    ]["rms"],
                    "candidate_glw_max": record["real_wrf_glw"][
                        "candidate_dynamic_interfaces_cam_gases"
                    ]["max_abs"],
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
