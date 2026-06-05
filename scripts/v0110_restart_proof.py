#!/usr/bin/env python3
"""Generate the v0.11.0 WRF-compatible restart continuity proof object."""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields, is_dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.noahclassic_surface_hook import NoahClassicLandState, NoahClassicRadiation
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.io.wrfrst_netcdf import (
    CARRY_ARRAY_FIELDS,
    DEFERRED_REGISTRY_RESTART_FIELDS,
    OPTIONAL_CARRY_FIELDS,
    SCHEMA_VERSION,
    STATE_FIELD_ORDER,
    STOCHASTIC_SEED_RESTART_VARIABLES,
    WRF_STANDARD_RESTART_VARIABLES,
    inspect_wrfrst_schema,
    read_wrfrst_carry,
    read_wrfrst_stochastic_seeds,
    write_wrfrst_carry,
)
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import _advance_chunk, _initial_carry_for_run
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


RUN_START = datetime(2026, 6, 3, 0, 0, 0)


def _pattern(shape: tuple[int, ...], dtype, offset: int):
    if np.dtype(dtype) == np.dtype("int32"):
        values = (np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape) + offset) % 31
        return jnp.asarray(values, dtype=dtype)
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    values = values / 1009.0 + offset / 997.0 + 0.125
    return jnp.asarray(values, dtype=dtype)


def _state(grid: GridSpec) -> State:
    return State(
        **{
            field: _pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
            for index, (field, shape) in enumerate(_state_field_shapes(grid).items(), start=1)
        }
    )


def _array(shape: tuple[int, ...], offset: int, *, dtype=jnp.float64):
    return _pattern(shape, dtype, offset)


def _noahmp_land(grid: GridSpec) -> NoahMPLandState:
    xy = (grid.ny, grid.nx)
    soil = (4, grid.ny, grid.nx)
    snow = (3, grid.ny, grid.nx)
    snso = (7, grid.ny, grid.nx)
    return NoahMPLandState(
        tslb=_array(soil, 101),
        smois=_array(soil, 102),
        sh2o=_array(soil, 103),
        smcwtd=_array(xy, 104),
        isnow=_array(xy, 105, dtype=jnp.int32),
        tsno=_array(snow, 106),
        snice=_array(snow, 107),
        snliq=_array(snow, 108),
        zsnso=_array(snso, 109),
        snowh=_array(xy, 110),
        sneqv=_array(xy, 111),
        sneqvo=_array(xy, 112),
        tauss=_array(xy, 113),
        albold=_array(xy, 114),
        tv=_array(xy, 115),
        tg=_array(xy, 116),
        tah=_array(xy, 117),
        eah=_array(xy, 118),
        canliq=_array(xy, 119),
        canice=_array(xy, 120),
        fwet=_array(xy, 121),
        lai=_array(xy, 122),
        sai=_array(xy, 123),
        cm=_array(xy, 124),
        ch=_array(xy, 125),
        t_skin=_array(xy, 126),
        qsfc=_array(xy, 127),
        znt=_array(xy, 128),
        emiss=_array(xy, 129),
        albedo=_array(xy, 130),
        sfcrunoff=_array(xy, 131),
        udrunoff=_array(xy, 132),
    )


def _noahclassic_land(grid: GridSpec) -> NoahClassicLandState:
    xy = (grid.ny, grid.nx)
    trailing_soil = (grid.ny, grid.nx, 4)
    return NoahClassicLandState(
        t1=_array(xy, 201),
        stc=_array(trailing_soil, 202),
        smc=_array(trailing_soil, 203),
        sh2o=_array(trailing_soil, 204),
        cmc=_array(xy, 205),
        sneqv=_array(xy, 206),
        snowh=_array(xy, 207),
        sncovr=_array(xy, 208),
        snotime1=_array(xy, 209),
        ribb=_array(xy, 210),
        flx4=_array(xy, 211),
        fvb=_array(xy, 212),
        fbur=_array(xy, 213),
        fgsn=_array(xy, 214),
        smcrel=_array(trailing_soil, 215),
        xlaidyn=_array(xy, 216),
        hfx=_array(xy, 217),
        qfx=_array(xy, 218),
        lh=_array(xy, 219),
        grdflx=_array(xy, 220),
    )


def _full_synthetic_carry(grid: GridSpec) -> OperationalCarry:
    state = _state(grid)
    xy = (grid.ny, grid.nx)
    return initial_operational_carry(
        state,
        noahmp_land=_noahmp_land(grid),
        noahmp_rad=(_array(xy, 301), _array(xy, 302), _array(xy, 303)),
        cumulus_carry=(_array((grid.nz, grid.ny, grid.nx), 304), _array(xy, 305, dtype=jnp.int32)),
        noahclassic_land=_noahclassic_land(grid),
        noahclassic_rad=NoahClassicRadiation(_array(xy, 306), _array(xy, 307), _array(xy, 308)),
    )


def _seed_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.arange(8, dtype=np.int32) + offset * 1000
        for offset, name in enumerate(STOCHASTIC_SEED_RESTART_VARIABLES, start=1)
    }


def _field_names(obj: Any) -> tuple[str, ...]:
    if obj is None:
        return ()
    if isinstance(obj, State):
        return STATE_FIELD_ORDER
    if isinstance(obj, OperationalCarry):
        return tuple(field.name for field in dataclass_fields(OperationalCarry))
    fields = getattr(obj, "_fields", ())
    if fields:
        return tuple(str(field) for field in fields)
    slots = getattr(obj, "__slots__", ())
    if slots:
        return tuple(str(field) for field in slots)
    if is_dataclass(obj):
        return tuple(field.name for field in dataclass_fields(type(obj)))
    return ()


def _compare_array(left: Any, right: Any) -> dict[str, Any]:
    a = np.asarray(jax.device_get(left))
    b = np.asarray(jax.device_get(right))
    same_shape = a.shape == b.shape
    same_dtype = a.dtype == b.dtype
    bit_identical = bool(same_shape and same_dtype and _same_bytes(a, b))
    max_abs = None
    if same_shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        if a.size:
            max_abs = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
        else:
            max_abs = 0.0
    return {
        "shape": [int(dim) for dim in a.shape],
        "right_shape": [int(dim) for dim in b.shape],
        "dtype": str(a.dtype),
        "right_dtype": str(b.dtype),
        "same_shape": bool(same_shape),
        "same_dtype": bool(same_dtype),
        "bit_identical": bit_identical,
        "max_abs": max_abs,
        "pass": bit_identical,
    }


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    left_bytes = np.ascontiguousarray(left).view(np.uint8)
    right_bytes = np.ascontiguousarray(right).view(np.uint8)
    return bool(np.array_equal(left_bytes, right_bytes))


def _compare_optional(left: Any, right: Any, prefix: str) -> dict[str, dict[str, Any]]:
    if left is None or right is None:
        same_none = left is None and right is None
        return {
            prefix: {
                "bit_identical": bool(same_none),
                "pass": bool(same_none),
                "left_present": left is not None,
                "right_present": right is not None,
            }
        }
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple) or len(left) != len(right):
            return {
                prefix: {
                    "bit_identical": False,
                    "pass": False,
                    "left_len": len(left) if isinstance(left, tuple) else None,
                    "right_len": len(right) if isinstance(right, tuple) else None,
                }
            }
        records: dict[str, dict[str, Any]] = {}
        for index, (lval, rval) in enumerate(zip(left, right, strict=True)):
            records[f"{prefix}[{index}]"] = _compare_array(lval, rval)
        return records
    fields = _field_names(left)
    if not fields or fields != _field_names(right):
        return {
            prefix: {
                "bit_identical": False,
                "pass": False,
                "left_fields": list(fields),
                "right_fields": list(_field_names(right)),
            }
        }
    return {f"{prefix}.{field}": _compare_array(getattr(left, field), getattr(right, field)) for field in fields}


def _compare_carry(left: OperationalCarry, right: OperationalCarry) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for field in STATE_FIELD_ORDER:
        records[f"state.{field}"] = _compare_array(getattr(left.state, field), getattr(right.state, field))
    for field in CARRY_ARRAY_FIELDS:
        records[f"carry.{field}"] = _compare_array(getattr(left, field), getattr(right, field))
    for field in OPTIONAL_CARRY_FIELDS:
        records.update(_compare_optional(getattr(left, field), getattr(right, field), field))
    failed = {name: record for name, record in records.items() if not bool(record["pass"])}
    return {
        "pass": not failed,
        "bit_identical": not failed,
        "field_count": int(len(records)),
        "failed_count": int(len(failed)),
        "failed_fields": sorted(failed),
        "per_field": records,
    }


def _timed_advance(name: str, carry: OperationalCarry, namelist: Any, start_step: int, n_steps: int, cadence: int) -> tuple[OperationalCarry, dict[str, Any]]:
    start = time.perf_counter()
    result = _advance_chunk(
        carry,
        namelist,
        jnp.asarray(int(start_step), dtype=jnp.int32),
        n_steps=int(n_steps),
        cadence=int(cadence),
    )
    block_until_ready(result)
    return result, {
        "name": name,
        "start_step": int(start_step),
        "n_steps": int(n_steps),
        "wall_s": float(time.perf_counter() - start),
    }


def _cpu_full_carry_roundtrip(proof_dir: Path) -> dict[str, Any]:
    grid = GridSpec.canary_3km_template()
    carry = _full_synthetic_carry(grid)
    seed_arrays = _seed_arrays()
    path = proof_dir / "wrfrst_full_carry_roundtrip.nc"
    write_wrfrst_carry(
        carry,
        grid,
        {},
        path,
        valid_time=RUN_START + timedelta(minutes=30),
        run_start=RUN_START,
        step_index=3,
        stochastic_seed_arrays=seed_arrays,
    )
    restored, metadata = read_wrfrst_carry(path)
    restored_seeds = read_wrfrst_stochastic_seeds(path)
    comparison = _compare_carry(carry, restored)
    seed_comparison = {
        name: _compare_array(seed_arrays[name], restored_seeds[name])
        for name in STOCHASTIC_SEED_RESTART_VARIABLES
    }
    schema = inspect_wrfrst_schema(path)
    return {
        "pass": bool(comparison["pass"] and all(record["pass"] for record in seed_comparison.values())),
        "purpose": "structural full-carry NetCDF wrfrst round-trip; not the forecast-continuity acceptance gate",
        "restart_path": str(path),
        "metadata": metadata,
        "comparison": comparison,
        "stochastic_seed_roundtrip": {
            "pass": bool(all(record["pass"] for record in seed_comparison.values())),
            "variables": list(STOCHASTIC_SEED_RESTART_VARIABLES),
            "per_variable": seed_comparison,
        },
        "schema": {
            "schema_version": schema["global_attrs"].get("GPUWRF_WRFRST_SCHEMA_VERSION"),
            "dimension_count": int(len(schema["dimensions"])),
            "variable_count": int(len(schema["variables"])),
            "dimensions": schema["dimensions"],
            "wrf_standard_variables_present": [
                name for name in WRF_STANDARD_RESTART_VARIABLES if name in schema["variables"]
            ],
            "noahmp_wrf_variables_present": [
                name for name in ("TSLB", "SMOIS", "SH2O", "TSNO", "SNICE", "SNLIQ", "ZSNSO")
                if name in schema["variables"]
            ],
            "stochastic_seed_variables_present": [
                name for name in STOCHASTIC_SEED_RESTART_VARIABLES
                if name in schema["variables"]
            ],
            "exact_state_variable_count": int(sum(1 for name in schema["variables"] if name.startswith("GPUWRF_STATE_"))),
            "exact_carry_variable_count": int(sum(1 for name in schema["variables"] if name.startswith("GPUWRF_CARRY_"))),
            "exact_optional_variable_count": int(sum(1 for name in schema["variables"] if name.startswith("GPUWRF_NOAH") or name.startswith("GPUWRF_CUMULUS_"))),
        },
    }


def _real_forecast_continuity(
    proof_dir: Path,
    *,
    run_id: str,
    run_root: Path,
    domain: str,
    split_hour: int,
    final_hour: int,
    dt_s: float,
) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("restart continuity acceptance gate must run on a GPU backend via /tmp/wrf_gpu_run.sh")
    config = DailyPipelineConfig(
        run_id=run_id,
        hours=int(final_hour),
        output_dir=proof_dir / "forecast_work",
        proof_dir=proof_dir,
        run_root=run_root,
        domain=domain,
        dt_s=float(dt_s),
        async_output=False,
        refresh_land_state_hourly=False,
    )
    case, run_dir = _build_real_case(config)
    cadence = int(case.namelist.radiation_cadence_steps)
    steps_per_hour = int(round(3600.0 / float(case.namelist.dt_s)))
    split_steps = int(split_hour) * steps_per_hour
    final_steps = int(final_hour) * steps_per_hour
    if not (0 < split_steps < final_steps):
        raise ValueError("split hour must be between zero and final hour")
    carry0 = _initial_carry_for_run(case.state, case.namelist)
    block_until_ready(carry0)

    whole, whole_timing = _timed_advance("uninterrupted_0_to_m", carry0, case.namelist, 1, final_steps, cadence)
    split_prefix, split_prefix_timing = _timed_advance("split_prefix_0_to_n", carry0, case.namelist, 1, split_steps, cadence)

    checkpoint_path = proof_dir / f"wrfrst_{domain}_hour{split_hour:02d}.nc"
    write_wrfrst_carry(
        split_prefix,
        case.grid,
        case.namelist,
        checkpoint_path,
        valid_time=case.run_start + timedelta(seconds=float(split_steps) * float(case.namelist.dt_s)),
        run_start=case.run_start,
        step_index=split_steps,
        lead_hours=float(split_hour),
    )
    restored, metadata = read_wrfrst_carry(checkpoint_path)
    checkpoint_cmp = _compare_carry(split_prefix, restored)

    continuation_steps = final_steps - split_steps
    split_no_io, split_no_io_timing = _timed_advance(
        "split_no_io_n_to_m",
        split_prefix,
        case.namelist,
        split_steps + 1,
        continuation_steps,
        cadence,
    )
    restarted, restarted_timing = _timed_advance(
        "restart_n_to_m",
        restored,
        case.namelist,
        split_steps + 1,
        continuation_steps,
        cadence,
    )

    whole_vs_split = _compare_carry(whole, split_no_io)
    whole_vs_restart = _compare_carry(whole, restarted)
    split_vs_restart = _compare_carry(split_no_io, restarted)
    schema = inspect_wrfrst_schema(checkpoint_path)
    return {
        "pass": bool(checkpoint_cmp["pass"] and whole_vs_restart["pass"] and split_vs_restart["pass"]),
        "case": {
            "source": case.metadata.get("source"),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "domain": domain,
            "dt_s": float(case.namelist.dt_s),
            "split_hour": int(split_hour),
            "final_hour": int(final_hour),
            "steps_per_hour": int(steps_per_hour),
            "split_steps": int(split_steps),
            "final_steps": int(final_steps),
            "radiation_cadence_steps": int(cadence),
            "physics": case.metadata.get("namelist", {}),
        },
        "restart_path": str(checkpoint_path),
        "restart_metadata": metadata,
        "timings": [whole_timing, split_prefix_timing, split_no_io_timing, restarted_timing],
        "checkpoint_roundtrip": checkpoint_cmp,
        "segmentation_control_whole_vs_split_no_io": whole_vs_split,
        "acceptance_whole_vs_restarted": whole_vs_restart,
        "restart_isolation_split_no_io_vs_restarted": split_vs_restart,
        "schema": {
            "dimensions": schema["dimensions"],
            "variable_count": int(len(schema["variables"])),
            "schema_version": schema["global_attrs"].get("GPUWRF_WRFRST_SCHEMA_VERSION"),
        },
    }


def generate_proof(
    output: Path,
    *,
    run_id: str,
    run_root: Path,
    domain: str,
    split_hour: int,
    final_hour: int,
    dt_s: float,
    skip_forecast: bool,
) -> dict[str, Any]:
    proof_dir = output.parent
    proof_dir.mkdir(parents=True, exist_ok=True)
    structural = _cpu_full_carry_roundtrip(proof_dir)
    forecast = None
    if not skip_forecast:
        forecast = _real_forecast_continuity(
            proof_dir,
            run_id=run_id,
            run_root=run_root,
            domain=domain,
            split_hour=int(split_hour),
            final_hour=int(final_hour),
            dt_s=float(dt_s),
        )
    proof = {
        "artifact": str(output),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "schema_version": SCHEMA_VERSION,
        "jax_backend": jax.default_backend(),
        "visible_gpu": visible_gpu_name(),
        "resource_policy": {
            "required_prefix_for_gpu_commands": "/tmp/wrf_gpu_run.sh",
            "taskset": "taskset -c 0-27",
            "pythonpath": "src",
        },
        "field_set": {
            "state_fields_covered": len(STATE_FIELD_ORDER),
            "state_fields_total": len(STATE_FIELD_ORDER),
            "carry_fields_covered": len(CARRY_ARRAY_FIELDS),
            "optional_carry_groups": list(OPTIONAL_CARRY_FIELDS),
            "wrf_standard_restart_variables": list(WRF_STANDARD_RESTART_VARIABLES),
            "deferred_registry_restart_fields": list(DEFERRED_REGISTRY_RESTART_FIELDS),
        },
        "structural_full_carry_roundtrip": structural,
        "real_forecast_continuity": forecast,
    }
    proof["pass"] = bool(structural["pass"] and forecast is not None and forecast["pass"])
    if skip_forecast:
        proof["pass"] = False
        proof["skip_reason"] = "real forecast continuity gate was skipped; proof is non-acceptance"
    output.write_text(json.dumps(proof, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return proof


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("proofs/v0110/restart_continuity.json"))
    parser.add_argument("--run-id", default="20260521_18z_l3_24h_20260522T133443Z")
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/data/canairy_meteo/runs/wrf_l3"))
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--split-hour", type=int, default=1)
    parser.add_argument("--final-hour", type=int, default=2)
    parser.add_argument("--dt-s", type=float, default=10.0)
    parser.add_argument("--skip-forecast", action="store_true", help="developer-only structural run; proof will be marked non-acceptance")
    args = parser.parse_args()
    proof = generate_proof(
        args.output,
        run_id=args.run_id,
        run_root=args.run_root,
        domain=args.domain,
        split_hour=args.split_hour,
        final_hour=args.final_hour,
        dt_s=args.dt_s,
        skip_forecast=bool(args.skip_forecast),
    )
    print(json.dumps({"pass": proof["pass"], "artifact": str(args.output)}, sort_keys=True))
    return 0 if proof["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
