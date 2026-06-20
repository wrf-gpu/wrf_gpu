#!/usr/bin/env python3
"""Extract compact v0.18 active full-WRF MP oracle summaries.

The input run directories are intentionally outside git because they contain
NetCDF history files. This script turns those pristine-WRF runs into small JSON
proof objects that preserve the exact module/source hashes, seeded variables,
success line, and two-frame state deltas needed for fail-closed reference gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")

SCHEMES: dict[int, dict[str, str]] = {
    9: {
        "scheme": "Milbrandt-Yau 2-moment",
        "exact_module": "phys/module_mp_milbrandt2mom.F",
        "registry_package": "milbrandt2mom",
    },
    18: {
        "scheme": "NSSL 2-moment 4-ice with predicted CCN",
        "exact_module": "phys/module_mp_nssl_2mom.F",
        "registry_package": "nssl_2mom",
    },
    27: {
        "scheme": "UDM 7-class / UFS double-moment",
        "exact_module": "phys/module_mp_udm.F",
        "registry_package": "udmscheme",
    },
    29: {
        "scheme": "RCON Thompson aerosol-aware variant",
        "exact_module": "phys/module_mp_rcon.F",
        "registry_package": "rcon_mp_scheme",
    },
    40: {
        "scheme": "Morrison aerosol",
        "exact_module": "phys/module_mp_morr_two_moment_aero.F",
        "registry_package": "morr_tm_aero",
    },
    50: {
        "scheme": "P3 1-category",
        "exact_module": "phys/module_mp_p3.F",
        "registry_package": "p3_1category",
    },
    51: {
        "scheme": "P3 1-category plus cloud number",
        "exact_module": "phys/module_mp_p3.F",
        "registry_package": "p3_1category_nc",
    },
    52: {
        "scheme": "P3 2-category",
        "exact_module": "phys/module_mp_p3.F",
        "registry_package": "p3_2category",
    },
    53: {
        "scheme": "P3 1-category 3-moment",
        "exact_module": "phys/module_mp_p3.F",
        "registry_package": "p3_1cat_3mom",
    },
    56: {
        "scheme": "NTU multi-moment",
        "exact_module": "phys/module_mp_ntu.F",
        "registry_package": "ntu",
    },
}

CORE_FIELDS = (
    "T",
    "QVAPOR",
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QSNOW",
    "QGRAUP",
    "QHAIL",
    "QNCLOUD",
    "QNRAIN",
    "QNICE",
    "QNSNOW",
    "QNGRAUPEL",
    "QNHAIL",
    "QNCCN",
    "QNWFA",
    "QNIFA",
    "QNBCA",
    "QNIN",
    "QIR",
    "QIB",
    "QIR2",
    "QIB2",
    "QZI",
    "FI",
    "FS",
    "VI",
    "VS",
    "VG",
    "AI",
    "AS",
    "AG",
    "AH",
    "I3M",
    "RAINNC",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_checksum(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256(path),
    }


def finite_float(value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"non-finite oracle value: {out!r}")
    return out


def stats(array: np.ndarray) -> dict[str, float]:
    return {
        "min": finite_float(np.min(array)),
        "max": finite_float(np.max(array)),
        "mean": finite_float(np.mean(array)),
        "max_abs": finite_float(np.max(np.abs(array))),
    }


def as_array(value: Any) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    return np.asarray(value, dtype=np.float64)


def center_trace(data: np.ndarray) -> dict[str, Any]:
    if data.ndim == 4:
        _, _, ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        return {
            "kind": "vertical_column",
            "y": cy,
            "x": cx,
            "time0": data[0, :, cy, cx].tolist(),
            "time1": data[-1, :, cy, cx].tolist(),
        }
    if data.ndim == 3:
        _, ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        return {
            "kind": "surface_point",
            "y": cy,
            "x": cx,
            "time0": finite_float(data[0, cy, cx]),
            "time1": finite_float(data[-1, cy, cx]),
        }
    return {"kind": "unsupported_rank"}


def decode_times(dataset: Dataset) -> list[str]:
    if "Times" not in dataset.variables:
        return []
    raw = dataset.variables["Times"][:]
    return [bytes(row).decode("ascii").rstrip("\x00") for row in raw]


def variable_summary(dataset: Dataset, name: str) -> dict[str, Any] | None:
    if name not in dataset.variables:
        return None
    variable = dataset.variables[name]
    data = as_array(variable[:])
    if data.shape[0] < 2:
        raise ValueError(f"{name} does not contain two history frames")
    delta = data[-1] - data[0]
    return {
        "shape": list(data.shape),
        "units": str(getattr(variable, "units", "")),
        "description": str(getattr(variable, "description", "")),
        "time0": stats(data[0]),
        "time1": stats(data[-1]),
        "delta": {
            "max_abs": finite_float(np.max(np.abs(delta))),
            "l2": finite_float(np.sqrt(np.mean(delta * delta))),
        },
        "center": center_trace(data),
    }


def summarize_scheme(mp: int, active_root: Path, out_root: Path) -> Path:
    info = SCHEMES[mp]
    run_dir = active_root / f"mp{mp}"
    result_path = run_dir / "active_run_result.json"
    history_path = run_dir / "wrfout_d01_0001-01-01_00:00:00"
    wrf_stdout = run_dir / "wrf_active.stdout"
    wrfinput = run_dir / "wrfinput_d01"
    namelist = run_dir / "namelist.input"

    result = json.loads(result_path.read_text())
    stdout_text = wrf_stdout.read_text(errors="replace")
    success_line = next(
        (line.strip() for line in stdout_text.splitlines() if "SUCCESS COMPLETE WRF" in line),
        "",
    )
    if not result.get("success") or not success_line:
        raise ValueError(f"mp={mp} active WRF oracle did not complete successfully")

    with Dataset(history_path) as dataset:
        variables: dict[str, Any] = {}
        for name in CORE_FIELDS:
            summary = variable_summary(dataset, name)
            if summary is not None:
                variables[name] = summary

        seeded_bases = {item.split("+", 1)[0] for item in result["seeded"]}
        nontrivial_seeded = [
            name
            for name in sorted(seeded_bases)
            if name in variables
            and name not in {"T", "QVAPOR"}
            and variables[name]["delta"]["max_abs"] > 0.0
        ]
        if not nontrivial_seeded:
            raise ValueError(f"mp={mp} has no nontrivial seeded microphysics-field delta")

        payload = {
            "schema": "gpuwrf-v018-active-full-wrf-mp-oracle-v1",
            "oracle_type": "pristine_wrf_full_model_active_driver",
            "endpoint": "ref_with_oracle_fail_closed",
            "bar_met": True,
            "mp_physics": mp,
            "scheme": info["scheme"],
            "exact_module": info["exact_module"],
            "registry_package": info["registry_package"],
            "driver": "phys/module_microphysics_driver.F",
            "run_case": "em_quarter_ss, 1 simulated minute, central active hydrometeor patch",
            "wrf_success": True,
            "success_line": success_line,
            "history_times": decode_times(dataset),
            "seeded_variables": result["seeded"],
            "nontrivial_seeded_microphysics_fields": nontrivial_seeded,
            "source_checksums": [
                source_checksum(WRF_ROOT / info["exact_module"]),
                source_checksum(WRF_ROOT / "phys/module_microphysics_driver.F"),
                source_checksum(WRF_ROOT / "Registry/Registry.EM_COMMON"),
                source_checksum(WRF_ROOT / "run/README.namelist"),
            ],
            "run_artifact_hashes": {
                "wrf_exe": source_checksum(WRF_ROOT / "main/wrf.exe"),
                "wrfinput_d01": source_checksum(wrfinput),
                "namelist_input": source_checksum(namelist),
                "wrf_active_stdout": source_checksum(wrf_stdout),
                "history_netcdf": source_checksum(history_path),
            },
            "variables": variables,
            "notes": [
                "The JSON is the committed proof object; NetCDF files remain in /tmp and are not required for normal tests.",
                "The active patch is deliberately small and central so it exercises the exact WRF module without changing the source tree.",
            ],
        }

    out_dir = out_root / f"mp{mp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-root",
        type=Path,
        default=Path("/tmp/v018_mp_wrf_active_probe"),
        help="Directory containing mp*/ active pristine-WRF runs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where mp*/oracle_summary.json files are written.",
    )
    args = parser.parse_args()

    written = [summarize_scheme(mp, args.active_root, args.out_root) for mp in sorted(SCHEMES)]
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
