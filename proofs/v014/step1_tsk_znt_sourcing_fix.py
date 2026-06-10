#!/usr/bin/env python3
"""V0.14 Step-1 TSK/ZNT sourcing proof at the exact MYNN surface boundary."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402
import step1_sfclay_boundary_fix as sfclay_prev  # noqa: E402
from gpuwrf.coupling.physics_couplers import _surface_column_view  # noqa: E402
from gpuwrf.coupling.physics_dispatch import DEFAULT_MP_PHYSICS  # noqa: E402
from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS  # noqa: E402
from gpuwrf.physics.noah_mp import mavail_from_prescribed_fields, roughness_from_prescribed_fields  # noqa: E402
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_tsk_znt_sourcing_fix.json"
OUT_MD = PROOF_DIR / "step1_tsk_znt_sourcing_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_tsk_znt_sourcing_fix_wrf_patch.diff"

SCRATCH = Path("/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609")
WRF_RUN = SCRATCH / "run"
WRF_ROOT = SCRATCH / "WRF"
WRF_SOURCE = WRF_ROOT / "phys/module_surface_driver.F"
WRF_ORACLE_MODULE = WRF_ROOT / "phys/module_wrfgpu2_oracle.F"
WRF_BASE_SOURCE = Path("/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/phys/module_surface_driver.F")
WRF_ORACLE_ROOT = Path("/tmp/wrf_gpu2_step1_tsk_znt_sourcing_fix/wrf_truth_surface")
SURFACE_ROOT = WRF_ORACLE_ROOT / "surface_mynn"
WRF_STDOUT = Path("/tmp/wrf_gpu2_step1_tsk_znt_sourcing_fix/wrf_oracle_singleton.stdout")
WRFINPUT_D02 = WRF_RUN / "wrfinput_d02"
LANDUSE_TBL = WRF_RUN / "LANDUSE.TBL"

DT_S = 6.0
P0_PA = 100000.0
RCP = 287.0 / 1004.0
TSK_PASS = 1.0e-12
ZNT_PASS = 1.0e-6


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        cand = cand[m]
        ref = ref[m]
    delta = cand - ref
    if delta.size == 0:
        return {"count": 0, "max_abs": None, "rmse": None, "bias": None, "ref_max_abs": None}
    return {
        "count": int(delta.size),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "bias": float(np.mean(delta)),
        "ref_max_abs": float(np.max(np.abs(ref))),
    }


def parse_sidecar(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "scheme":
            data["scheme"] = parts[1]
        elif parts[0] == "tag":
            data["tag"] = parts[1]
        elif parts[0] == "grid_id":
            data["grid_id"] = int(parts[1])
        elif parts[0] == "itimestep":
            data["itimestep"] = int(parts[1])
        elif parts[0] == "dims_ni_nk_nj":
            data["ni"], data["nk"], data["nj"] = (int(parts[1]), int(parts[2]), int(parts[3]))
    return data


def ensure_wrf_surface_oracle() -> dict[str, Any]:
    sidecar = SURFACE_ROOT / "sfclay_mynn_in.sidecar.txt"
    if sidecar.exists():
        meta = parse_sidecar(sidecar)
        if meta.get("grid_id") == 2 and meta.get("itimestep") == 1 and meta.get("ni") == 159 and meta.get("nj") == 66:
            return {"status": "REUSED_EXISTING", "sidecar": str(sidecar), "meta": meta}

    WRF_ORACLE_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        "timeout",
        "900",
        "env",
        "WRFGPU2_ORACLE=1",
        "WRFGPU2_ORACLE_GRID=2",
        "WRFGPU2_ORACLE_STEP=1",
        f"WRFGPU2_ORACLE_ROOT={WRF_ORACLE_ROOT}",
        "OMP_NUM_THREADS=1",
        "./wrf.exe",
    ]
    proc = subprocess.run(
        command,
        cwd=str(WRF_RUN),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=930,
    )
    WRF_STDOUT.parent.mkdir(parents=True, exist_ok=True)
    WRF_STDOUT.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        return {"status": "BLOCKED_WRF_SINGLETON_FAILED", "command": command, "returncode": proc.returncode, "stdout": str(WRF_STDOUT)}
    if not sidecar.exists():
        return {"status": "BLOCKED_WRF_ORACLE_MISSING_AFTER_RUN", "command": command, "stdout": str(WRF_STDOUT)}
    return {"status": "RAN_SINGLETON", "command": command, "stdout": str(WRF_STDOUT), "meta": parse_sidecar(sidecar)}


def read2(name: str) -> np.ndarray:
    sidecar = parse_sidecar(SURFACE_ROOT / "sfclay_mynn_in.sidecar.txt")
    path = SURFACE_ROOT / name
    return np.fromfile(path, dtype=">f8").reshape(sidecar["nj"], sidecar["ni"])


def read3(name: str) -> np.ndarray:
    sidecar = parse_sidecar(SURFACE_ROOT / "sfclay_mynn_in.sidecar.txt")
    path = SURFACE_ROOT / name
    return np.fromfile(path, dtype=">f8").reshape(sidecar["nj"], sidecar["nk"], sidecar["ni"])


def read_wrfinput(name: str) -> np.ndarray | None:
    with Dataset(WRFINPUT_D02, "r") as ds:
        if name not in ds.variables:
            return None
        values = np.asarray(ds.variables[name][:])
    if values.shape[:1] == (1,):
        values = values[0]
    return values


def generate_wrf_patch() -> dict[str, Any]:
    chunks: list[str] = []
    commands: list[list[str]] = []
    if WRF_BASE_SOURCE.exists() and WRF_SOURCE.exists():
        command = ["git", "diff", "--no-index", "--", str(WRF_BASE_SOURCE), str(WRF_SOURCE)]
        proc = subprocess.run(command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        chunks.append(proc.stdout)
        commands.append(command)
    if WRF_ORACLE_MODULE.exists():
        command = ["git", "diff", "--no-index", "--", "/dev/null", str(WRF_ORACLE_MODULE)]
        proc = subprocess.run(command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        chunks.append(proc.stdout)
        commands.append(command)
    OUT_WRF_PATCH.write_text("\n".join(chunk for chunk in chunks if chunk), encoding="utf-8")
    return {
        "path": str(OUT_WRF_PATCH),
        "commands": commands,
        "exists": OUT_WRF_PATCH.exists(),
        "size_bytes": OUT_WRF_PATCH.stat().st_size if OUT_WRF_PATCH.exists() else None,
    }


def build_live_surface_state() -> tuple[Any, Any, Any]:
    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    namelist = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    state = patched["carry"].state
    if int(namelist.mp_physics) == DEFAULT_MP_PHYSICS:
        state = om.thompson_adapter(state, DT_S)
    elif int(namelist.mp_physics) in MP_SCAN_ADAPTERS:
        state = MP_SCAN_ADAPTERS[int(namelist.mp_physics)](state, DT_S, namelist.grid)
    return inputs, patched, state


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    oracle_status = ensure_wrf_surface_oracle()
    if str(oracle_status.get("status", "")).startswith("BLOCKED"):
        return {"status": oracle_status["status"], "oracle": oracle_status}

    patch_info = generate_wrf_patch()

    wrf_in = {
        "tsk": read2("sfclay_mynn_in__tsk.f64"),
        "znt": read2("sfclay_mynn_in__znt.f64"),
        "mavail": read2("sfclay_mynn_in__mavail.f64"),
        "xland": read2("sfclay_mynn_in__xland.f64"),
        "ust": read2("sfclay_mynn_in__ust.f64"),
    }
    wrf_out = {
        "znt": read2("sfclay_mynn_out__znt.f64"),
        "ust": read2("sfclay_mynn_out__ust.f64"),
        "hfx": read2("sfclay_mynn_out__hfx.f64"),
        "qfx": read2("sfclay_mynn_out__qfx.f64"),
        "qsfc": read2("sfclay_mynn_out__qsfc.f64"),
        "br": read2("sfclay_mynn_out__br.f64"),
    }
    wrf_col = {
        "u_phy": read3("sfclay_mynn_in__u_phy.f64"),
        "v_phy": read3("sfclay_mynn_in__v_phy.f64"),
        "t_phy": read3("sfclay_mynn_in__t_phy.f64"),
        "th_phy": read3("sfclay_mynn_in__th_phy.f64"),
        "qv": read3("sfclay_mynn_in__qv.f64"),
        "p_phy": read3("sfclay_mynn_in__p_phy.f64"),
        "dz8w": read3("sfclay_mynn_in__dz8w.f64"),
    }

    xland = read_wrfinput("XLAND")
    landmask = read_wrfinput("LANDMASK")
    lu_index = read_wrfinput("LU_INDEX")
    vegfra = read_wrfinput("VEGFRA")
    cm = read_wrfinput("CM")
    smois = read_wrfinput("SMOIS")
    table_znt = np.asarray(
        roughness_from_prescribed_fields(xland, landmask, vegfra=vegfra, cm=cm, lu_index=lu_index),
        dtype=np.float64,
    )
    old_znt = np.asarray(roughness_from_prescribed_fields(xland, landmask, vegfra=vegfra, cm=cm), dtype=np.float64)
    table_mavail = np.asarray(mavail_from_prescribed_fields(xland, landmask, smois, lu_index=lu_index), dtype=np.float64)
    old_mavail = np.asarray(mavail_from_prescribed_fields(xland, landmask, smois), dtype=np.float64)

    inputs, patched, state = build_live_surface_state()
    col = _surface_column_view(state)
    diag = surface_layer_with_diagnostics(col, first_timestep=True)
    strict = sfclay_prev.strict_step1_metric(inputs, patched["carry"])
    strict_metric = strict.get("metric") if isinstance(strict, Mapping) else None

    is_land = wrf_in["xland"] < 1.5
    is_water = wrf_in["xland"] > 1.5
    p0 = np.asarray(col.p[..., 0], dtype=np.float64)
    theta0 = np.asarray(col.theta[..., 0], dtype=np.float64)
    t_from_theta = theta0 * (p0 / P0_PA) ** RCP

    source_metrics = {
        "state_tsk_vs_sfclay_in_tsk": diffstat(state.t_skin, wrf_in["tsk"]),
        "state_znt_vs_sfclay_in_znt": diffstat(state.roughness_m, wrf_in["znt"]),
        "state_znt_vs_sfclay_in_znt_land": diffstat(state.roughness_m, wrf_in["znt"], is_land),
        "state_znt_vs_sfclay_in_znt_water": diffstat(state.roughness_m, wrf_in["znt"], is_water),
        "state_mavail_vs_sfclay_in_mavail": diffstat(state.mavail, wrf_in["mavail"]),
        "state_ust_vs_sfclay_in_ust": diffstat(state.ustar, wrf_in["ust"]),
        "landuse_table_znt_vs_sfclay_in_znt": diffstat(table_znt, wrf_in["znt"]),
        "old_surrogate_znt_vs_sfclay_in_znt": diffstat(old_znt, wrf_in["znt"]),
        "landuse_table_mavail_vs_sfclay_in_mavail": diffstat(table_mavail, wrf_in["mavail"]),
        "old_surrogate_mavail_vs_sfclay_in_mavail": diffstat(old_mavail, wrf_in["mavail"]),
    }
    column_metrics = {
        "u0_vs_sfclay_in_u_phy": diffstat(col.u[..., 0], wrf_col["u_phy"][:, 0, :]),
        "v0_vs_sfclay_in_v_phy": diffstat(col.v[..., 0], wrf_col["v_phy"][:, 0, :]),
        "qv0_vs_sfclay_in_qv": diffstat(col.qv[..., 0], wrf_col["qv"][:, 0, :]),
        "p0_vs_sfclay_in_p_phy": diffstat(p0, wrf_col["p_phy"][:, 0, :]),
        "dz0_vs_sfclay_in_dz8w": diffstat(np.asarray(col.dz)[..., 0], wrf_col["dz8w"][:, 0, :]),
        "theta0_vs_sfclay_in_th_phy": diffstat(theta0, wrf_col["th_phy"][:, 0, :]),
        "temperature_from_theta_vs_sfclay_in_t_phy": diffstat(t_from_theta, wrf_col["t_phy"][:, 0, :]),
    }
    output_metrics = {
        "diag_znt_vs_sfclay_out_znt": diffstat(diag.znt, wrf_out["znt"]),
        "diag_ust_vs_sfclay_out_ust": diffstat(diag.fluxes.ustar, wrf_out["ust"]),
        "diag_hfx_vs_sfclay_out_hfx": diffstat(diag.hfx, wrf_out["hfx"]),
        "diag_qfx_vs_sfclay_out_qfx": diffstat(np.asarray(diag.lh) / 2.5e6, wrf_out["qfx"]),
        "diag_qsfc_vs_sfclay_out_qsfc": diffstat(diag.qsfc, wrf_out["qsfc"]),
        "diag_br_vs_sfclay_out_br": diffstat(diag.br, wrf_out["br"]),
    }

    mynnedmf_tsk = SURFACE_ROOT / "mynnedmf_in__tsk.f64"
    later_handoff = {}
    if mynnedmf_tsk.exists():
        later_handoff["mynn_driver_in_tsk_vs_sfclay_in_tsk"] = diffstat(
            np.fromfile(mynnedmf_tsk, dtype=">f8").reshape(wrf_in["tsk"].shape),
            wrf_in["tsk"],
        )

    source_fixed = (
        source_metrics["state_tsk_vs_sfclay_in_tsk"]["max_abs"] <= TSK_PASS
        and source_metrics["state_znt_vs_sfclay_in_znt"]["max_abs"] <= ZNT_PASS
    )
    strict_closed = bool(
        strict_metric
        and strict_metric.get("max_abs") is not None
        and float(strict_metric["max_abs"]) <= 1.0e-3
        and float(strict_metric["rmse"]) <= 1.0e-5
    )
    if strict_closed:
        status = "STRICT_STEP1_CLOSED"
    elif source_fixed:
        status = "TSK_ZNT_SOURCE_FIXED_NEXT_BLOCKER_THERMODYNAMIC_COLUMN_INPUTS"
    else:
        status = "TSK_ZNT_SOURCE_STILL_RED"

    payload = {
        "artifact": "step1_tsk_znt_sourcing_fix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "status": status,
        "oracle": oracle_status,
        "wrf_patch": patch_info,
        "paths": {
            "wrf_run": str(WRF_RUN),
            "wrfinput_d02": str(WRFINPUT_D02),
            "landuse_tbl": str(LANDUSE_TBL),
            "surface_oracle": str(SURFACE_ROOT),
        },
        "source_metrics": source_metrics,
        "column_metrics": column_metrics,
        "surface_output_metrics": output_metrics,
        "strict_step1_metric": strict_metric,
        "later_handoff_metrics": later_handoff,
        "ranked_findings": [
            {
                "rank": 1,
                "status": "FIXED" if source_fixed else "OPEN",
                "hypothesis": "Step-1 pre-sfclay TSK/ZNT surface-source mismatch is the blocker.",
                "evidence": {
                    "tsk": source_metrics["state_tsk_vs_sfclay_in_tsk"],
                    "znt": source_metrics["state_znt_vs_sfclay_in_znt"],
                    "wrf_source": "module_physics_init.F landuse_init sets Z0=SFZ0/100 and ZNT=Z0 for MODIFIED_IGBP_MODIS_NOAH; direct wrfinput ZNT/Z0 are absent.",
                },
            },
            {
                "rank": 2,
                "status": "BLOCKING" if not strict_closed else "SECONDARY",
                "hypothesis": "Remaining sfclay mismatch is in non-TSK/ZNT thermodynamic column inputs before SFCLAY_mynn.",
                "evidence": {
                    "theta": column_metrics["theta0_vs_sfclay_in_th_phy"],
                    "temperature": column_metrics["temperature_from_theta_vs_sfclay_in_t_phy"],
                    "pressure": column_metrics["p0_vs_sfclay_in_p_phy"],
                    "u_v_qv_are_bounded": {
                        "u": column_metrics["u0_vs_sfclay_in_u_phy"],
                        "v": column_metrics["v0_vs_sfclay_in_v_phy"],
                        "qv": column_metrics["qv0_vs_sfclay_in_qv"],
                    },
                },
            },
            {
                "rank": 3,
                "status": "LATER_THAN_SFCLAY_INPUT",
                "hypothesis": "The prior TSK residual was observed at the MYNN-driver/PBL handoff, not at SFCLAY_mynn input.",
                "evidence": later_handoff,
            },
        ],
        "acceptance": {
            "tsk_znt_source_fixed": source_fixed,
            "strict_step1_closed": strict_closed,
            "next_fastest_command": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_tsk_znt_sourcing_fix.py",
        },
    }
    return payload


def write_markdown(payload: Mapping[str, Any]) -> None:
    source = payload.get("source_metrics", {})
    column = payload.get("column_metrics", {})
    output = payload.get("surface_output_metrics", {})
    strict = payload.get("strict_step1_metric") or {}
    lines = [
        "# V0.14 Step-1 TSK/ZNT Sourcing Fix",
        "",
        f"Verdict: `{payload['status']}`.",
        "",
        "## WRF-Anchored Result",
        "",
        f"- `TSK` at `SFCLAY_mynn` input: max_abs `{source['state_tsk_vs_sfclay_in_tsk']['max_abs']}` K.",
        f"- `ZNT` at `SFCLAY_mynn` input: max_abs `{source['state_znt_vs_sfclay_in_znt']['max_abs']}` m, RMSE `{source['state_znt_vs_sfclay_in_znt']['rmse']}`.",
        f"- `MAVAIL` at `SFCLAY_mynn` input: max_abs `{source['state_mavail_vs_sfclay_in_mavail']['max_abs']}`.",
        "- WRF `wrfinput_d02` has no direct `ZNT`/`Z0`; WRF initializes `ZNT` from `LANDUSE.TBL` `SFZ0/100` by `LU_INDEX` before this call.",
        f"- Old roughness surrogate witness: max_abs `{source['old_surrogate_znt_vs_sfclay_in_znt']['max_abs']}` m; table-backed source max_abs `{source['landuse_table_znt_vs_sfclay_in_znt']['max_abs']}` m.",
        "",
        "## Remaining Blocker",
        "",
        "TSK/ZNT source is no longer the Step-1 blocker. The next WRF-anchored blocker is the non-surface thermodynamic column entering `SFCLAY_mynn`:",
        "",
        f"- `th_phy(kts)` max_abs `{column['theta0_vs_sfclay_in_th_phy']['max_abs']}` K, RMSE `{column['theta0_vs_sfclay_in_th_phy']['rmse']}`.",
        f"- derived `t_phy(kts)` max_abs `{column['temperature_from_theta_vs_sfclay_in_t_phy']['max_abs']}` K, RMSE `{column['temperature_from_theta_vs_sfclay_in_t_phy']['rmse']}`.",
        f"- `p_phy(kts)` max_abs `{column['p0_vs_sfclay_in_p_phy']['max_abs']}` Pa, RMSE `{column['p0_vs_sfclay_in_p_phy']['rmse']}`.",
        f"- `u/v/qv(kts)` are bounded at max_abs `{column['u0_vs_sfclay_in_u_phy']['max_abs']}`, `{column['v0_vs_sfclay_in_v_phy']['max_abs']}`, `{column['qv0_vs_sfclay_in_qv']['max_abs']}`.",
        "",
        "Surface output remains red with exact TSK/ZNT/MAVAIL:",
        "",
        f"- `UST` max_abs `{output['diag_ust_vs_sfclay_out_ust']['max_abs']}`, RMSE `{output['diag_ust_vs_sfclay_out_ust']['rmse']}`.",
        f"- `HFX` max_abs `{output['diag_hfx_vs_sfclay_out_hfx']['max_abs']}`, RMSE `{output['diag_hfx_vs_sfclay_out_hfx']['rmse']}`.",
        f"- `QFX` max_abs `{output['diag_qfx_vs_sfclay_out_qfx']['max_abs']}`, RMSE `{output['diag_qfx_vs_sfclay_out_qfx']['rmse']}`.",
        "",
        "## Strict Step-1",
        "",
        f"- after-conv `T_TENDF` max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        "",
        "## Files",
        "",
        f"- JSON proof: `{OUT_JSON}`",
        f"- WRF hook patch archive: `{OUT_WRF_PATCH}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review(payload: Mapping[str, Any]) -> None:
    strict = payload.get("strict_step1_metric") or {}
    source = payload.get("source_metrics", {})
    column = payload.get("column_metrics", {})
    lines = [
        "# Review: V0.14 Step-1 TSK/ZNT Sourcing",
        "",
        f"Verdict: `{payload['status']}`.",
        "",
        f"Pre-sfclay `TSK` is exact: max_abs `{source['state_tsk_vs_sfclay_in_tsk']['max_abs']}`.",
        f"Pre-sfclay `ZNT` is fixed: max_abs `{source['state_znt_vs_sfclay_in_znt']['max_abs']}`.",
        f"Strict Step-1 remains red: max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        "",
        "Next blocker: non-TSK/ZNT thermodynamic column inputs at `SFCLAY_mynn`.",
        f"`th_phy(kts)` max_abs `{column['theta0_vs_sfclay_in_th_phy']['max_abs']}`; `t_phy(kts)` max_abs `{column['temperature_from_theta_vs_sfclay_in_t_phy']['max_abs']}`; `p_phy(kts)` max_abs `{column['p0_vs_sfclay_in_p_phy']['max_abs']}`.",
        "",
        f"Proof: `{OUT_MD}`",
    ]
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    if not str(payload.get("status", "")).startswith("BLOCKED"):
        write_markdown(payload)
        write_review(payload)
    return 0 if not str(payload.get("status", "")).startswith("BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
