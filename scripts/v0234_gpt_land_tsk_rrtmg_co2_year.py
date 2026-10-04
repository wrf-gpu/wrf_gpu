#!/usr/bin/env python3
"""Discriminate WRF's run-date CAM greenhouse gases through d03 LW/Noah-MP."""

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

from netCDF4 import Dataset, chartostring
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
WRF_LW_SOURCE = SCRATCH / "pristine-wrf-source/phys/module_ra_rrtmg_lw.F"
WRF_CLWRF_SOURCE = SCRATCH / "pristine-wrf-source/phys/module_ra_clWRF_support.F"
WRF_GHG_LINK = SCRATCH / "pristine-wrf-source/run/CAMtr_volume_mixing_ratio"
WRF_GHG_DATA = SCRATCH / "pristine-wrf-source/run/CAMtr_volume_mixing_ratio.SSP245"
INTERFACE_TSK_PROOF = SPRINT / "rrtmg-interface-tsk-proof.json"
INTERFACE_REAL_WRF_PROOF = SPRINT / "rrtmg-interface-real-wrf-proof.json"

BUFFER_TSK_PROOF_SHA256 = "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a"
BUFFER_DELTA_SHA256 = "f50ac95bb2e22d5e52a12f4fa4634140dd6df0a273a7183b58f167291cb98d0c"
WRF_LW_SOURCE_SHA256 = "c7a5238612aa8a4213c8d3af6708ec6a5248e6701e19758a80e563905d306de3"
WRF_CLWRF_SOURCE_SHA256 = "b249f472550dd2d8e594fb5d2474e280f423e5e339a9ff393d42c261098d5182"
WRF_GHG_DATA_SHA256 = "9a427fd106f8e36b30e0b29266bff1398b025b82af5b878e5a7a8e9dfe268ca7"
INTERFACE_TSK_PROOF_SHA256 = "13b8f7414a13becc3c458c81a744d15ea50af7f73a929dec0af447a4cc5184fc"
INTERFACE_REAL_WRF_PROOF_SHA256 = "e2a1733ff7a63e4063017ae1a8d5e704bb90e2a7ed1f6446432868c5cf09bc0c"
TARGET_TIME = "2025-03-01_00:00:00"
TARGET_JULIAN = 60.0
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


SPEC = importlib.util.spec_from_file_location("v0234_rrtmg_tsk_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("v0234 RRTMG TSK proof helper unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
provenance = base.provenance


class CO2YearFailure(RuntimeError):
    """Fail-closed CO2-year authority or paired-replay failure."""


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
        raise CO2YearFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _tracked_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = _git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise CO2YearFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": provenance.sha256_bytes(disk),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _load_delta(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    actual = provenance.sha256_file(path)
    if actual != expected_sha:
        raise CO2YearFailure(f"DELTA_HASH:{path}:{actual}")
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"{name}_delta" for name in DELTA_FIELDS}
        if set(archive.files) != expected:
            raise CO2YearFailure(f"DELTA_SCHEMA:{path}")
        values = {
            name: np.asarray(archive[f"{name}_delta"], dtype=np.float64)
            for name in DELTA_FIELDS
        }
    if any(
        value.shape != (93, 111) or not np.isfinite(value).all()
        for value in values.values()
    ):
        raise CO2YearFailure(f"DELTA_ARRAY:{path}")
    return values


def _mid_june_julian(year: int) -> np.float32:
    """Replays CLWRF's mondata=6 `juldata` expression in default REAL."""

    february = 29 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 28
    months = (0, 31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return np.float32(
        np.float32(sum(months[:6]))
        + np.float32(months[6]) / np.float32(2.0)
        - np.float32(0.5)
    )


def _cam_gases(target_year: int, target_julian: float) -> dict[str, Any]:
    """Replays the two-valid-year CLWRF interpolation for this fixture."""

    if WRF_GHG_LINK.resolve() != WRF_GHG_DATA.resolve():
        raise CO2YearFailure(f"WRF_GHG_LINK:{WRF_GHG_LINK.resolve()}")
    actual = provenance.sha256_file(WRF_GHG_DATA)
    if actual != WRF_GHG_DATA_SHA256:
        raise CO2YearFailure(f"WRF_GHG_DATA_HASH:{actual}")
    rows: dict[int, tuple[float, ...]] = {}
    lines = WRF_GHG_DATA.read_text(encoding="ascii", errors="strict").splitlines()
    for line in lines[2:]:
        values = line.split()
        if not values:
            continue
        if len(values) != 6:
            raise CO2YearFailure(f"WRF_GHG_ROW:{line}")
        rows[int(values[0])] = tuple(float(value) for value in values[1:])
    lower_year = target_year - 1
    upper_year = target_year
    if lower_year not in rows or upper_year not in rows:
        raise CO2YearFailure("WRF_GHG_BRACKET")

    lower_julian = _mid_june_julian(lower_year)
    upper_julian = _mid_june_julian(upper_year)
    # `interpolate_CAMgases`: place all three dates on the min-year time
    # axis, then evaluate fact1/fact2 in default REAL before multiplying the
    # REAL(r8) gas table values.
    days_lower = np.float32(366.0 if lower_year % 4 == 0 else 365.0)
    data_lower = lower_julian
    data_upper = np.float32(upper_julian + days_lower)
    model_time = np.float32(np.float32(target_julian) + days_lower)
    delta_time = np.float32(data_upper - data_lower)
    fact1 = np.float32((data_upper - model_time) / delta_time)
    fact2 = np.float32((model_time - data_lower) / delta_time)

    names = ("co2", "n2o", "ch4", "cfc11", "cfc12")
    scales = (1.0e-6, 1.0e-9, 1.0e-9, 1.0e-12, 1.0e-12)
    gases: dict[str, float] = {}
    pre_scale: dict[str, float] = {}
    for index, (name, scale) in enumerate(zip(names, scales, strict=True)):
        interpolated = (
            np.float64(rows[lower_year][index]) * np.float64(fact1)
            + np.float64(rows[upper_year][index]) * np.float64(fact2)
        )
        pre_scale[name] = float(interpolated)
        # The source scaling literals are default REAL, promoted for the
        # REAL(r8) multiplication only after their default-kind rounding.
        gases[name] = float(interpolated * np.float64(np.float32(scale)))
    return {
        "path": str(WRF_GHG_DATA),
        "link": str(WRF_GHG_LINK),
        "link_target": WRF_GHG_LINK.readlink().as_posix(),
        "sha256": actual,
        "lower_year": lower_year,
        "upper_year": upper_year,
        "lower_row": list(rows[lower_year]),
        "upper_row": list(rows[upper_year]),
        "lower_julian": float(lower_julian),
        "upper_julian": float(upper_julian),
        "target_julian": float(target_julian),
        "fact1": float(fact1),
        "fact2": float(fact2),
        "pre_scale": pre_scale,
        "vmr": gases,
    }


def _source_gate() -> dict[str, Any]:
    expected = {
        WRF_LW_SOURCE: WRF_LW_SOURCE_SHA256,
        WRF_CLWRF_SOURCE: WRF_CLWRF_SOURCE_SHA256,
    }
    records: dict[str, Any] = {}
    for path, digest in expected.items():
        actual = provenance.sha256_file(path)
        if actual != digest:
            raise CO2YearFailure(f"WRF_SOURCE_HASH:{path}:{actual}")
        records[path.name] = {"path": str(path), "sha256": actual}
    lw_source = WRF_LW_SOURCE.read_text(encoding="utf-8", errors="strict")
    clwrf_source = WRF_CLWRF_SOURCE.read_text(encoding="utf-8", errors="strict")
    lw_tokens = (
        "IF ( GHG_INPUT .EQ. 1 ) THEN",
        'CALL read_CAMgases(yr,julian,.false.,"RRTMG",co2,n2o,ch4,cfc11,cfc12)',
        "co2vmr(ncol,k) = co2",
        "n2ovmr(ncol,k) = n2o",
        "ch4vmr(ncol,k) = ch4",
        "cfc11vmr(ncol,k) = cfc11",
        "cfc12vmr(ncol,k) = cfc12",
    )
    clwrf_tokens = (
        "mondata(idata) = 6",
        "CALL valid_years(yearIN, co2r, max_years,yr1, yr2)",
        "CALL interpolate_CAMgases(yr, julian, nyrm, njulm, yr1, yr2, nyrp, njulp, max_years, co2r  , co2vmr  )",
        "interp_gas = gas(yr1)*fact1+gas(yr2)*fact2",
        "co2vmr  =co2vmr  *1.e-06",
        "n2ovmr  =n2ovmr  *1.e-09",
        "ch4vmr  =ch4vmr  *1.e-09",
        "cfc11vmr=cfc11vmr*1.e-12",
        "cfc12vmr=cfc12vmr*1.e-12",
    )
    if not all(token in lw_source for token in lw_tokens):
        raise CO2YearFailure("WRF_LW_GHG_SOURCE_TOKENS")
    if not all(token in clwrf_source for token in clwrf_tokens):
        raise CO2YearFailure("WRF_CLWRF_SOURCE_TOKENS")
    with Dataset(WRFINPUT) as dataset:
        times = chartostring(dataset["Times"][:]).tolist()
        start_date = str(dataset.START_DATE)
        ghg_input = int(dataset.GHG_INPUT)
    if times != [TARGET_TIME] or start_date != TARGET_TIME:
        raise CO2YearFailure(f"TARGET_TIME:{times}:{start_date}")
    if ghg_input != 1:
        raise CO2YearFailure(f"TARGET_GHG_INPUT:{ghg_input}")
    target_year = int(TARGET_TIME[:4])
    return {
        "files": records,
        "lw_source_tokens": list(lw_tokens),
        "clwrf_source_tokens": list(clwrf_tokens),
        "wrfinput_time": times[0],
        "target_year": target_year,
        "ghg_input": ghg_input,
        "annual_fallback_branch_selected": False,
        "cam_interpolation": _cam_gases(target_year, TARGET_JULIAN),
    }


def _interface_authority() -> dict[str, Any]:
    """Validate the sealed WRF P3D/P8W/T8W source and real-WRF proof."""

    expected = {
        INTERFACE_TSK_PROOF: INTERFACE_TSK_PROOF_SHA256,
        INTERFACE_REAL_WRF_PROOF: INTERFACE_REAL_WRF_PROOF_SHA256,
    }
    records: dict[str, Any] = {}
    for path, digest in expected.items():
        actual = provenance.sha256_file(path)
        if actual != digest:
            raise CO2YearFailure(f"INTERFACE_PROOF_HASH:{path}:{actual}")
        records[path.name] = {"path": str(path), "sha256": actual}
    tsk_proof = json.loads(INTERFACE_TSK_PROOF.read_text(encoding="utf-8"))
    real_wrf = json.loads(INTERFACE_REAL_WRF_PROOF.read_text(encoding="utf-8"))
    if not (
        tsk_proof.get("passed") is True
        and tsk_proof.get("source_boundary", {}).get(
            "authenticated_real_wrf_input_replay"
        )
        is True
        and tsk_proof.get("tsk_parity", {}).get(
            "land_rms_and_max_strictly_improve"
        )
        is True
        and real_wrf.get("status") == "PASS"
        and real_wrf.get("is_self_compare") is False
        and real_wrf.get("gates", {}).get(
            "authenticated_p3d_t3d_dz8w_bitwise_replay"
        )
        is True
        and real_wrf.get("gates", {}).get("real_wrf_glw_strictly_improves")
        is True
    ):
        raise CO2YearFailure("INTERFACE_PROOF_AUTHORITY")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delta-output", required=True, type=Path)
    parser.add_argument(
        "--candidate-arm",
        choices=("cam", "cam-interfaces", "production"),
        default="cam",
        help=(
            "Compare CAM gases alone, the exploratory interface composition, "
            "or the current coupler-populated production composition"
        ),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    delta_output = args.delta_output.resolve()

    original_gases: dict[str, float] | None = None
    original_cfc: np.ndarray | None = None
    solve_rrtmg_lw_column = None
    try:
        resources = provenance.resource_gate()
        _manifest_check, static_authority = provenance.validate_static_inputs()
        upstream = base._validate_upstream_proofs()
        source_authority = _source_gate()
        interface_authority = (
            _interface_authority()
            if args.candidate_arm in {"cam-interfaces", "production"}
            else None
        )
        if provenance.sha256_file(BUFFER_TSK_PROOF) != BUFFER_TSK_PROOF_SHA256:
            raise CO2YearFailure("BUFFER_TSK_PROOF_HASH")
        buffer_proof = json.loads(BUFFER_TSK_PROOF.read_text(encoding="utf-8"))
        if not (
            buffer_proof.get("passed") is True
            and buffer_proof.get("tsk_parity", {}).get(
                "land_rms_and_max_strictly_improve"
            )
            is True
        ):
            raise CO2YearFailure("BUFFER_TSK_PROOF_AUTHORITY")
        qml_delta = base._load_qml_delta()
        buffer_delta = _load_delta(BUFFER_DELTA, BUFFER_DELTA_SHA256)
        script_record = _tracked_record(Path(__file__).resolve())
        rrtmg_record = _tracked_record(REPO / "src/gpuwrf/physics/rrtmg_lw.py")
        constants_record = _tracked_record(REPO / "src/gpuwrf/physics/rrtmg_constants.py")
        coupler_record = _tracked_record(REPO / "src/gpuwrf/coupling/physics_couplers.py")
        clwrf_record = _tracked_record(REPO / "src/gpuwrf/physics/wrf_clwrf_ghg.py")

        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
        from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs
        from gpuwrf.coupling.physics_couplers import (
            _to_columns,
            _wrf_hydrostatic_pressure_profiles_from_state,
            _wrf_phy_prep_temperature_interfaces,
        )
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
        from gpuwrf.physics.noah_mp import (
            mavail_from_prescribed_fields,
            roughness_from_prescribed_fields,
        )
        from gpuwrf.physics.noahmp_coupler import (
            assemble_noahmp_forcing,
            noahmp_surface_adapter,
        )
        import gpuwrf.physics.rrtmg_lw as rrtmg_lw
        from gpuwrf.runtime.operational_mode import _NoahMPClock, _NoahMPRadiation

        solve_rrtmg_lw_column = rrtmg_lw.solve_rrtmg_lw_column
        original_cfc = np.asarray(rrtmg_lw._CFC_VMR, dtype=np.float64).copy()
        original_gases = {
            "co2": float(rrtmg_lw.CO2_VMR),
            "n2o": float(rrtmg_lw.N2O_VMR),
            "ch4": float(rrtmg_lw.CH4_VMR),
            "cfc11": float(original_cfc[1]),
            "cfc12": float(original_cfc[2]),
        }
        expected_current = {
            "co2": 431.3824884728998e-6,
            "n2o": 319.0e-9,
            "ch4": 1774.0e-9,
            "cfc11": 0.251e-9,
            "cfc12": 0.538e-9,
        }
        if original_gases != expected_current:
            raise CO2YearFailure(f"PORT_GHG_CONSTANTS:{original_gases}")
        target_gases = source_authority["cam_interpolation"]["vmr"]
        if set(target_gases) != set(original_gases):
            raise CO2YearFailure(f"TARGET_GHG_SCHEMA:{target_gases}")
        if target_gases == original_gases:
            raise CO2YearFailure("GHG_ARMS_IDENTICAL")
        if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
            raise CO2YearFailure("JAX_EXECUTION_MODE")

        manifest = json.loads((SEAM / "manifest.json").read_text(encoding="utf-8"))
        seam_state = provenance.load_seam_state(manifest, State, jax)
        with CARRY.open("rb") as stream:
            carry = pickle.load(stream)
        if type(carry).__name__ != "OperationalCarry" or carry.noahmp_rad is None:
            raise CO2YearFailure("CARRY_SCHEMA")

        run = Gen2Run(INPUTS)
        grid = dataclass_replace(
            run.grid("d03").as_grid_spec(), metrics=load_wrfinput_metrics(WRFINPUT)
        )
        if float(grid.vertical.top_pressure_pa) != 5000.0:
            raise CO2YearFailure("GRID_TOP")
        with Dataset(WRFINPUT) as dataset:
            landmask = np.asarray(dataset["LANDMASK"][0], dtype=np.float64)
            lu_index = np.asarray(dataset["LU_INDEX"][0], dtype=np.int32)
            wrf_xland = np.asarray(dataset["XLAND"][0], dtype=np.float64)
        land = wrf_xland < 1.5
        if int(np.sum(land)) != 2034:
            raise CO2YearFailure("LAND_COUNT")

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
        assembled_sw_state, assembled_lw_state, *_rest = _rrtmg_column_inputs(
            replay_state,
            grid,
            time_utc=TARGET_TIME if args.candidate_arm == "production" else None,
            lead_seconds=0.0,
            land_state=carry.noahmp_land,
        )
        # Reconstruct the admitted pre-composition path exactly: phy_prep T3D,
        # nonhydrostatic mass pressure, no explicit P8W/T8W, historical gases,
        # and the accepted exact model-top buffer.
        lw_state = assembled_lw_state.replace(
            p=_to_columns(replay_state.p),
            pressure_interfaces=None,
            temperature_interfaces=None,
            co2_vmr=None,
            n2o_vmr=None,
            ch4_vmr=None,
            cfc11_vmr=None,
            cfc12_vmr=None,
            ozone_vmr=None,
        )
        if not (
            lw_state.top_pressure_pa == 5000.0
            and lw_state.pressure_interfaces is None
            and lw_state.temperature_interfaces is None
            and lw_state.co2_vmr is None
            and lw_state.ozone_vmr is None
        ):
            raise CO2YearFailure("ACCEPTED_PRODUCTION_ARM")

        target_state = lw_state
        if args.candidate_arm == "production":
            target_state = assembled_lw_state
            assembled_lw_gases = {
                "co2": target_state.co2_vmr,
                "n2o": target_state.n2o_vmr,
                "ch4": target_state.ch4_vmr,
                "cfc11": target_state.cfc11_vmr,
                "cfc12": target_state.cfc12_vmr,
            }
            assembled_sw_gases = {
                "co2": assembled_sw_state.co2_vmr,
                "n2o": assembled_sw_state.n2o_vmr,
                "ch4": assembled_sw_state.ch4_vmr,
            }
            if not (
                target_state.pressure_interfaces is not None
                and target_state.temperature_interfaces is not None
                and assembled_sw_state.pressure_interfaces is not None
                and assembled_sw_state.temperature_interfaces is not None
                and assembled_lw_gases == target_gases
                and assembled_sw_gases
                == {name: target_gases[name] for name in ("co2", "n2o", "ch4")}
            ):
                raise CO2YearFailure(
                    f"PRODUCTION_COMPOSITION:{assembled_lw_gases}:{assembled_sw_gases}"
                )
        elif args.candidate_arm == "cam-interfaces":
            p_hyd, p_hyd_w, _psfc = _wrf_hydrostatic_pressure_profiles_from_state(
                replay_state, grid.metrics
            )
            t_mass = jnp.moveaxis(jnp.asarray(lw_state.T), -1, 0)
            target_state = lw_state.replace(
                p=_to_columns(p_hyd),
                pressure_interfaces=_to_columns(p_hyd_w),
                temperature_interfaces=_to_columns(
                    _wrf_phy_prep_temperature_interfaces(
                        jnp.asarray(t_mass, dtype=jnp.float32),
                        replay_state,
                        grid.metrics,
                    )
                ),
            )
            if (
                target_state.pressure_interfaces is None
                or target_state.temperature_interfaces is None
            ):
                raise CO2YearFailure("INTERFACE_CANDIDATE_CONSTRUCTION")

        def set_gases(gases: dict[str, float]) -> None:
            rrtmg_lw.CO2_VMR = gases["co2"]
            rrtmg_lw.N2O_VMR = gases["n2o"]
            rrtmg_lw.CH4_VMR = gases["ch4"]
            rrtmg_lw._CFC_VMR = np.asarray(
                [original_cfc[0], gases["cfc11"], gases["cfc12"], original_cfc[3]],
                dtype=np.float64,
            )

        # Both arms use the accepted top buffer and identical column state.  The
        # module globals are restored in the finally block; clearing the JIT
        # cache forces each trace to embed the selected gas values.
        set_gases(original_gases)
        solve_rrtmg_lw_column.clear_cache()
        legacy_glw = solve_rrtmg_lw_column(
            lw_state.replace(top_pressure_pa=None), debug=False
        ).surface_down
        current_glw = solve_rrtmg_lw_column(lw_state, debug=False).surface_down
        jax.block_until_ready((legacy_glw, current_glw))

        set_gases(
            original_gases
            if args.candidate_arm == "production"
            else target_gases
        )
        solve_rrtmg_lw_column.clear_cache()
        target_glw = solve_rrtmg_lw_column(target_state, debug=False).surface_down
        jax.block_until_ready(target_glw)

        legacy_np = np.asarray(legacy_glw, dtype=np.float64)
        current_np = np.asarray(current_glw, dtype=np.float64)
        target_np = np.asarray(target_glw, dtype=np.float64)
        authenticated_rad = _NoahMPRadiation(*carry.noahmp_rad)
        held_lwdn = np.asarray(authenticated_rad.lwdn, dtype=np.float64)
        legacy_vs_held = provenance.split_metrics(legacy_np, held_lwdn, land)
        if not (
            legacy_vs_held["all"]["rms"] < 2.0e-6
            and legacy_vs_held["all"]["max_abs"] < 1.0e-4
        ):
            raise CO2YearFailure("LEGACY_HELD_RADIATION_REPLAY")

        current_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn) + jnp.asarray(current_np - legacy_np),
            authenticated_rad.cosz,
        )
        target_rad = _NoahMPRadiation(
            authenticated_rad.soldn,
            jnp.asarray(authenticated_rad.lwdn) + jnp.asarray(target_np - legacy_np),
            authenticated_rad.cosz,
        )
        clock = _NoahMPClock(julian=60.0, yearlen=365.0)
        current_forcing = assemble_noahmp_forcing(
            view, noahmp_static, current_rad, clock, 6.0
        )
        target_forcing = assemble_noahmp_forcing(
            view, noahmp_static, target_rad, clock, 6.0
        )

        changed_forcing: list[str] = []
        forcing_comparison: dict[str, Any] = {}
        for name in current_forcing._fields:
            old_value = getattr(current_forcing, name)
            new_value = getattr(target_forcing, name)
            if old_value is None or new_value is None:
                if old_value is not new_value:
                    raise CO2YearFailure(f"FORCING_NONE:{name}")
                continue
            record = provenance.metrics(np.asarray(new_value), np.asarray(old_value))
            forcing_comparison[name] = record
            if record["bitwise_mismatch_count"]:
                changed_forcing.append(name)
        if changed_forcing != ["lwdn"]:
            raise CO2YearFailure(f"FORCING_AB:{changed_forcing}")

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

        held = replay(
            assemble_noahmp_forcing(
                view, noahmp_static, authenticated_rad, clock, 6.0
            ),
            authenticated_rad,
        )
        current = replay(current_forcing, current_rad)
        target = replay(target_forcing, target_rad)
        paired_delta = {
            name: np.asarray(target[name], dtype=np.float64)
            - np.asarray(current[name], dtype=np.float64)
            for name in DELTA_FIELDS
        }
        for name, delta in paired_delta.items():
            if np.count_nonzero(delta[~land]) != 0:
                raise CO2YearFailure(f"WATER_CHANGED:{name}")

        reproduced_buffer_delta = np.asarray(current["t_skin"], dtype=np.float64) - np.asarray(
            held["t_skin"], dtype=np.float64
        )
        buffer_replay = provenance.metrics(
            reproduced_buffer_delta, buffer_delta["t_skin"]
        )
        if buffer_replay["rms"] > 5.0e-13 or buffer_replay["max_abs"] > 5.0e-12:
            raise CO2YearFailure(f"BUFFER_DELTA_REPLAY:{buffer_replay}")

        wrf_tsk = provenance.load_wrf_tsk()
        accepted_tsk = (
            np.asarray(seam_state.t_skin, dtype=np.float64)
            + qml_delta["t_skin"]
            + buffer_delta["t_skin"]
        )
        target_tsk = accepted_tsk + paired_delta["t_skin"]
        accepted_metrics = provenance.split_metrics(accepted_tsk, wrf_tsk, land)
        target_metrics = provenance.split_metrics(target_tsk, wrf_tsk, land)
        if not (
            accepted_metrics["land"]["rms"] == 0.018808123113575367
            and accepted_metrics["land"]["max_abs"] == 0.05064729526111478
        ):
            raise CO2YearFailure(f"ACCEPTED_TSK_DRIFT:{accepted_metrics['land']}")
        strictly_improves = (
            target_metrics["land"]["rms"] < accepted_metrics["land"]["rms"]
            and target_metrics["land"]["max_abs"] < accepted_metrics["land"]["max_abs"]
        )
        water_invariant = target_metrics["water"]["bitwise_mismatch_count"] == 0
        if not water_invariant:
            raise CO2YearFailure("TSK_WATER_CHANGED")

        delta_artifact = provenance.atomic_npz(delta_output, paired_delta)
        proof = {
            "schema": (
                "v0234-rrtmg-wrf-clwrf-production-composition-tsk-v1"
                if args.candidate_arm == "production"
                else (
                    "v0234-rrtmg-wrf-cam-ghg-interfaces-tsk-discriminator-v1"
                    if args.candidate_arm == "cam-interfaces"
                    else "v0234-rrtmg-wrf-cam-ghg-tsk-discriminator-v1"
                )
            ),
            "verdict": (
                f"WRF_{args.candidate_arm.upper().replace('-', '_')}_STRICTLY_IMPROVES_D03_TSK"
                if strictly_improves
                else f"WRF_{args.candidate_arm.upper().replace('-', '_')}_D03_TSK_FALSIFIED"
            ),
            "passed": True,
            "candidate_authorizes_fix_request": strictly_improves,
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
                "noahmp_replays": 3,
            },
            "authority": {
                **upstream,
                "static": static_authority,
                "script": script_record,
                "rrtmg_lw_source": rrtmg_record,
                "rrtmg_constants_source": constants_record,
                "production_coupler_source": coupler_record,
                "production_clwrf_interpolation_source": clwrf_record,
                "wrf_source": source_authority,
                "sealed_interface_proofs": interface_authority,
                "buffer_tsk_proof_sha256": BUFFER_TSK_PROOF_SHA256,
                "buffer_delta_sha256": BUFFER_DELTA_SHA256,
                "authenticated_held_lwdn_sha256": provenance.array_sha(held_lwdn),
            },
            "paired_radiation": {
                "same_column_state": args.candidate_arm == "cam",
                "same_top_buffer": True,
                "only_changed_kernel_inputs": (
                    [
                        "CO2 VMR",
                        "N2O VMR",
                        "CH4 VMR",
                        "CFC11 VMR",
                        "CFC12 VMR",
                    ]
                    + (
                        ["hydrostatic P3D", "hydrostatic P8W", "phy_prep T8W"]
                        if args.candidate_arm in {"cam-interfaces", "production"}
                        else []
                    )
                    + (["O3RAD"] if args.candidate_arm == "production" else [])
                ),
                "production_coupler_populated_candidate": (
                    args.candidate_arm == "production"
                ),
                "candidate_arm": args.candidate_arm,
                "accepted_constant_gases": original_gases,
                "target_wrf_cam_gases": target_gases,
                "relative_scalar_changes": {
                    name: target_gases[name] / original_gases[name] - 1.0
                    for name in original_gases
                },
                "legacy_glw_vs_authenticated_held": legacy_vs_held,
                "accepted_buffer_minus_legacy": provenance.split_metrics(
                    current_np, legacy_np, land
                ),
                "target_cam_minus_accepted_glw": provenance.split_metrics(
                    target_np, current_np, land
                ),
                "delta_application": (
                    "authenticated held LWDN + accepted buffer-minus-legacy + "
                    "target-CAM-minus-accepted GLW"
                ),
            },
            "paired_noahmp": {
                "only_changed_forcing_leaf": "lwdn",
                "forcing_comparison": forcing_comparison,
                "surface_delta": {
                    name: provenance.split_metrics(target[name], current[name], land)
                    for name in DELTA_FIELDS
                },
                "sealed_buffer_delta_replay": buffer_replay,
            },
            "tsk_parity": {
                "accepted_qml_plus_top_buffer": accepted_metrics,
                "candidate_plus_wrf_cam_ghg": target_metrics,
                "land_rms_ratio": (
                    target_metrics["land"]["rms"]
                    / accepted_metrics["land"]["rms"]
                ),
                "land_rms_improvement_fraction": (
                    1.0
                    - target_metrics["land"]["rms"]
                    / accepted_metrics["land"]["rms"]
                ),
                "land_rms_and_max_strictly_improve": strictly_improves,
                "water_bitwise_invariant": water_invariant,
                "remaining_status": "LAND_TSK_NOT_CLOSED",
            },
            "delta_artifact": delta_artifact,
            "delta_contract": {
                "fields": list(DELTA_FIELDS),
                "application": (
                    "authenticated seam + accepted QML delta + accepted buffer delta + "
                    "paired target-CAM GHG delta"
                ),
                "water_delta_bitwise_zero": True,
                "paired_same_inputs_except_authorized_radiation_composition": True,
                "synthetic_wrf_reference": False,
                "empirical_tuning": False,
            },
        }
        provenance.atomic_json(output, proof)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "accepted_land_rms": accepted_metrics["land"]["rms"],
                    "candidate_land_rms": target_metrics["land"]["rms"],
                    "candidate_land_max": target_metrics["land"]["max_abs"],
                    "candidate_authorizes_fix_request": strictly_improves,
                    "delta": delta_artifact,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - proof must emit a fail-closed reason
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    finally:
        if original_gases is not None and original_cfc is not None:
            try:
                import gpuwrf.physics.rrtmg_lw as rrtmg_lw

                rrtmg_lw.CO2_VMR = original_gases["co2"]
                rrtmg_lw.N2O_VMR = original_gases["n2o"]
                rrtmg_lw.CH4_VMR = original_gases["ch4"]
                rrtmg_lw._CFC_VMR = original_cfc
                if solve_rrtmg_lw_column is not None:
                    solve_rrtmg_lw_column.clear_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
