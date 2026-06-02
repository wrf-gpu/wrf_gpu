"""Generate the v0.4.0 S1 wrfinput dynamics proof report."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.init.metgrid_schema import MetEmArtifact, MetgridProjection, metem_field_specs
from gpuwrf.init.real_init.base_state import compute_base_state
from gpuwrf.init.real_init.hydrostatic import balance
from gpuwrf.init.real_init.types import RealInitConfig, WRFINPUT_TOLS
from gpuwrf.init.real_init.vertical_coord import compute_vertical_coord
from gpuwrf.init.real_init.vinterp import vertical_interpolate


RUN_ROOT = Path("/mnt/data/canairy_meteo/runs")
WRF_ROOT = RUN_ROOT / "wrf_l3"
WPS_ROOT = RUN_ROOT / "wps_cases"
OUT = ROOT / "proofs/v040/s1_wrfinput_dynamics_report.json"

FIELDS = (
    "ZNW",
    "ZNU",
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "P_TOP",
    "PB",
    "MUB",
    "PHB",
    "MU",
    "P",
    "PH",
    "T",
    "U",
    "V",
    "W",
    "QVAPOR",
)


def _read_metem(path: Path, domain: str) -> MetEmArtifact:
    with Dataset(str(path)) as ds:
        attrs = ds.__dict__
        projection = MetgridProjection(
            map_proj=int(attrs["MAP_PROJ"]),
            truelat1=float(attrs["TRUELAT1"]),
            truelat2=float(attrs["TRUELAT2"]),
            stand_lon=float(attrs["STAND_LON"]),
            moad_cen_lat=float(attrs["MOAD_CEN_LAT"]),
            pole_lat=float(attrs["POLE_LAT"]),
            pole_lon=float(attrs["POLE_LON"]),
            dx_m=float(attrs["DX"]),
            dy_m=float(attrs["DY"]),
            nx=len(ds.dimensions["west_east"]),
            ny=len(ds.dimensions["south_north"]),
            grid_id=int(attrs.get("grid_id", domain[-1])),
            parent_id=int(attrs.get("parent_id", 1)),
            parent_grid_ratio=int(attrs.get("parent_grid_ratio", 1)),
            i_parent_start=int(attrs.get("i_parent_start", 1)),
            j_parent_start=int(attrs.get("j_parent_start", 1)),
        )
        spec_names = {spec.name for spec in metem_field_specs()}
        arrays = {
            name: np.asarray(ds.variables[name][0])
            for name in spec_names
            if name in ds.variables
        }
        valid_time = "".join(
            ch.decode() if isinstance(ch, bytes) else str(ch)
            for ch in np.asarray(ds.variables["Times"][0])
        ).strip()
    return MetEmArtifact(
        domain=domain,
        valid_time=valid_time,
        projection=projection,
        arrays=arrays,
        provenance={"source": str(path)},
    )


def _config_from_wrfinput(ds: Dataset) -> RealInitConfig:
    return RealInitConfig(
        nz=len(ds.dimensions["bottom_top"]),
        p_top_pa=float(ds.variables["P_TOP"][0]),
        hybrid_opt=int(getattr(ds, "HYBRID_OPT")),
        etac=float(getattr(ds, "ETAC")),
        base_pres=float(ds.variables["P00"][0]),
        base_temp=float(ds.variables["T00"][0]),
        base_lapse=float(ds.variables["TLP"][0]),
        iso_temp=float(ds.variables["TISO"][0]),
        base_pres_strat=float(ds.variables["P_STRAT"][0]),
        base_lapse_strat=float(ds.variables["TLP_STRAT"][0]),
        grid_id=int(getattr(ds, "GRID_ID", 1)),
    )


def _case_time(run_dir: Path) -> tuple[str, str] | None:
    match = re.match(r"^(\d{8})_18z_", run_dir.name)
    if not match:
        return None
    ymd = match.group(1)
    return ymd, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}_18:00:00"


def _select_cases() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for run_dir in sorted(WRF_ROOT.glob("*_18z_l3_24h_*")):
        parsed = _case_time(run_dir)
        if parsed is None:
            continue
        ymd, valid_time = parsed
        wps_dir = WPS_ROOT / f"{ymd}_18z_72h/l3"
        if not wps_dir.exists():
            continue
        for domain in ("d01", "d02", "d03"):
            wrfinput = run_dir / f"wrfinput_{domain}"
            metem = wps_dir / f"met_em.{domain}.{valid_time}.nc"
            if not wrfinput.exists() or not metem.exists():
                break
        else:
            selected.extend(
                {
                    "run_dir": run_dir,
                    "domain": domain,
                    "wrfinput": run_dir / f"wrfinput_{domain}",
                    "metem": wps_dir / f"met_em.{domain}.{valid_time}.nc",
                    "valid_time": valid_time,
                }
                for domain in ("d01", "d02", "d03")
            )
        if len(selected) >= 12:
            break
    if len(selected) < 10:
        raise RuntimeError(f"need >=10 S1 cases, found {len(selected)}")
    return selected[:12]


def _field_map(config: RealInitConfig, vcoord, base, dynamics) -> dict[str, np.ndarray]:
    ny, nx = base.mub.shape
    return {
        "ZNW": vcoord.znw,
        "ZNU": vcoord.znu,
        "C1H": vcoord.c1h,
        "C2H": vcoord.c2h,
        "C3H": vcoord.c3h,
        "C4H": vcoord.c4h,
        "C1F": vcoord.c1f,
        "C2F": vcoord.c2f,
        "C3F": vcoord.c3f,
        "C4F": vcoord.c4f,
        "P_TOP": np.asarray(float(config.p_top_pa)),
        "PB": base.pb,
        "MUB": base.mub,
        "PHB": base.phb,
        "MU": dynamics.mu,
        "P": dynamics.p,
        "PH": dynamics.ph,
        "T": dynamics.theta,
        "U": dynamics.u,
        "V": dynamics.v,
        "W": dynamics.w,
        "QVAPOR": dynamics.qv,
    }


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    diff = a - e
    maxabs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.max(np.abs(e)))
    rel = 0.0 if denom == 0.0 else maxabs / denom
    return {"rmse": rmse, "max_abs": maxabs, "rel_max": rel}


def _compare_case(case: dict[str, Any]) -> dict[str, Any]:
    metem = _read_metem(case["metem"], case["domain"])
    with Dataset(str(case["wrfinput"])) as ds:
        config = _config_from_wrfinput(ds)
        vcoord = compute_vertical_coord(config)
        base = compute_base_state(config, vcoord, metem.arrays["HGT_M"])
        seed = vertical_interpolate(config, vcoord, metem)
        dynamics = balance(config, vcoord, base, seed)
        actuals = _field_map(config, vcoord, base, dynamics)

        fields: dict[str, Any] = {}
        for name in FIELDS:
            expected = np.asarray(ds.variables[name][0])
            metrics = _metrics(actuals[name], expected)
            tol = WRFINPUT_TOLS.get(name)
            if tol is None:
                passed = None
            else:
                passed = metrics["rmse"] <= tol[0] and metrics["max_abs"] <= tol[1]
            fields[name] = {
                **metrics,
                "tol": None if tol is None else {"rmse": tol[0], "max_abs": tol[1]},
                "passed": passed,
            }
        base_constants = {
            "P00": float(ds.variables["P00"][0]),
            "T00": float(ds.variables["T00"][0]),
            "TLP": float(ds.variables["TLP"][0]),
            "TISO": float(ds.variables["TISO"][0]),
            "P_STRAT": float(ds.variables["P_STRAT"][0]),
            "TLP_STRAT": float(ds.variables["TLP_STRAT"][0]),
        }
        attrs = {
            "HYBRID_OPT": int(getattr(ds, "HYBRID_OPT")),
            "ETAC": float(getattr(ds, "ETAC")),
        }
    return {
        "case": case["run_dir"].name,
        "domain": case["domain"],
        "valid_time": case["valid_time"],
        "wrfinput": str(case["wrfinput"]),
        "metem": str(case["metem"]),
        "etac": attrs["ETAC"],
        "hybrid_opt": attrs["HYBRID_OPT"],
        "base_constants": base_constants,
        "fields": fields,
    }


def _aggregate(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_field: dict[str, Any] = {}
    for name in FIELDS:
        items = [case["fields"][name] for case in case_reports]
        tol = next((item["tol"] for item in items if item["tol"] is not None), None)
        max_rmse = max(item["rmse"] for item in items)
        max_abs = max(item["max_abs"] for item in items)
        max_rel = max(item["rel_max"] for item in items)
        failed_cases = [
            f"{case['case']}:{case['domain']}"
            for case in case_reports
            if case["fields"][name]["passed"] is False
        ]
        passed = None if tol is None else not failed_cases
        by_field[name] = {
            "max_rmse": max_rmse,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "tol": tol,
            "passed": passed,
            "failed_cases": failed_cases,
        }
    gated = [v for v in by_field.values() if v["passed"] is not None]
    n_pass = sum(1 for item in gated if item["passed"])
    return {
        "fields": by_field,
        "gated_pass_count": n_pass,
        "gated_field_count": len(gated),
        "overall_pass": n_pass == len(gated),
    }


def main() -> int:
    cases = _select_cases()
    reports = [_compare_case(case) for case in cases]
    aggregate = _aggregate(reports)
    payload = {
        "sprint": "v0.4.0-S1",
        "oracle": "real.exe wrfinput files under /mnt/data/canairy_meteo/runs",
        "case_count": len(reports),
        "domains": sorted({case["domain"] for case in reports}),
        "resource_limits": {"python_prefix_required": "taskset -c 0-3", "jax_platform": "cpu"},
        "tolerance_source": "gpuwrf.init.real_init.types.WRFINPUT_TOLS",
        "overall_pass": aggregate["overall_pass"],
        "aggregate": aggregate,
        "cases": reports,
        "unresolved_residuals": [
            "C1H/C1F max-abs may fail the frozen 1e-5 cap from WRF single-precision "
            "hybrid coefficient differencing; ZNW/ZNU/C3*/C4* and all dynamics fields "
            "are reported separately without masking or tolerance changes."
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "overall_pass": aggregate["overall_pass"]}))
    return 0 if aggregate["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
