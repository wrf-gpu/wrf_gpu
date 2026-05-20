#!/usr/bin/env python3
"""Generate the M5 Thompson analytic column fixture from transcribed WRF formulas."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.physics.thompson_constants import CP, EPS, HGFR, LFUS, LSUB, R1, R_D, RV, T_0, XM0I  # noqa: E402


FIXTURE_ID = "analytic-thompson-column-v1"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
WRF_SOURCE = ROOT.parent / "wrf_gpu" / "sidecar_reports" / "post13_thompson_first_divergence_20260508T224837Z" / "source_snapshots_pre" / "module_mp_thompson.F.pre"


def _rslf(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NumPy RSLF transcription from module_mp_thompson.F.pre lines 5444-5468."""

    x = np.maximum(-80.0, T - 273.16)
    esl = (
        0.611583699e03
        + x
        * (
            0.444606896e02
            + x
            * (
                0.143177157e01
                + x
                * (
                    0.264224321e-1
                    + x
                    * (
                        0.299291081e-3
                        + x
                        * (
                            0.203154182e-5
                            + x * (0.702620698e-8 + x * (0.379534310e-11 + x * -0.321582393e-13))
                        )
                    )
                )
            )
        )
    )
    esl = np.minimum(esl, p * 0.15)
    return 0.622 * esl / (p - esl)


def _rsif(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """NumPy RSIF transcription from module_mp_thompson.F.pre lines 5473-5490."""

    x = np.maximum(-80.0, T - 273.16)
    esi = (
        0.609868993e03
        + x
        * (
            0.499320233e02
            + x
            * (
                0.184672631e01
                + x
                * (
                    0.402737184e-1
                    + x
                    * (
                        0.565392987e-3
                        + x
                        * (
                            0.521693933e-5
                            + x * (0.307839583e-7 + x * (0.105785160e-9 + x * 0.161444444e-12))
                        )
                    )
                )
            )
        )
    )
    esi = np.minimum(esi, p * 0.15)
    return 0.622 * esi / np.maximum(1.0e-4, p - esi)


def _lvap(T: np.ndarray) -> np.ndarray:
    """NumPy lvap formula from module_mp_thompson.F.pre lines 2063 and 3270."""

    return 2.5e6 + (2106.0 - 4218.0) * (T - 273.15)


def _ocp(qv: np.ndarray) -> np.ndarray:
    """NumPy ocp formula from module_mp_thompson.F.pre lines 2061 and 3272."""

    return 1.0 / (CP * (1.0 + 0.887 * qv))


def _rho(p: np.ndarray, T: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """NumPy rho formula from mp_gt_driver line 1270."""

    return 0.622 * p / (R_D * T * (qv + 0.622))


def reference_step_numpy(fields: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Path-B reference body mirroring the JAX Thompson source/sink subset."""

    qv = np.maximum(fields["qv"].copy(), 1.0e-10)
    qc = np.maximum(fields["qc"].copy(), 0.0)
    qr = np.maximum(fields["qr"].copy(), 0.0)
    qi = np.maximum(fields["qi"].copy(), 0.0)
    qs = np.maximum(fields["qs"].copy(), 0.0)
    qg = np.maximum(fields["qg"].copy(), 0.0)
    Ni = np.maximum(fields["Ni"].copy(), 0.0)
    Nr = np.maximum(fields["Nr"].copy(), 0.0)
    T = fields["T"].copy()
    p = fields["p"].copy()

    qvs = _rslf(p, T)
    lvap = _lvap(T)
    ocp = _ocp(qv)
    lvt2 = lvap * lvap * ocp / RV / (T * T)
    clap = (qv - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(3):
        expo = np.exp(lvt2 * clap)
        clap = clap - (qvs * expo - qv + clap) / (qvs * lvt2 * expo + 1.0)
    ssatw = qv / qvs - 1.0
    active = (ssatw > EPS) | ((ssatw < -EPS) & (qc > 0.0))
    clap = np.where(active, clap, 0.0)
    clap = np.where(clap < 0.0, np.maximum(clap, -qc), np.minimum(clap, qv - 1.0e-10))
    qv = qv - clap
    qc = qc + clap
    T = T + lvap * ocp * clap

    ocp = _ocp(qv)
    lvap = _lvap(T)
    lfus2 = LSUB - lvap
    qi_melt = np.where(T > T_0, qi, 0.0)
    qc = qc + qi_melt
    qi = qi - qi_melt
    Ni = np.where(qi_melt > 0.0, 0.0, Ni)
    T = T - LFUS * ocp * qi_melt

    qc_freeze = np.where(T < HGFR, qc, 0.0)
    qc = qc - qc_freeze
    qi = qi + qc_freeze
    Ni = Ni + qc_freeze / XM0I
    T = T + lfus2 * ocp * qc_freeze

    rain_freeze_fraction = np.clip((HGFR - T) / 40.0, 0.0, 1.0)
    rain_freeze = qr * rain_freeze_fraction
    qr = qr - rain_freeze
    qg = qg + rain_freeze
    Nr = Nr * (1.0 - rain_freeze_fraction)
    T = T + lfus2 * ocp * rain_freeze

    warm_fraction = np.clip((T - T_0) / 20.0, 0.0, 1.0)
    qs_melt = qs * warm_fraction
    qg_melt = qg * warm_fraction
    qs = qs - qs_melt
    qg = qg - qg_melt
    qr = qr + qs_melt + qg_melt
    T = T - LFUS * ocp * (qs_melt + qg_melt)

    qvsi = _rsif(p, T)
    existing_ice = qi + qs + qg
    deposition = np.minimum(qv - 1.0e-10, 0.25 * np.maximum(qv - qvsi, 0.0))
    deposition = np.where(T < T_0, deposition, 0.0)
    ice_weight = np.where(existing_ice > R1, qi / np.maximum(existing_ice, R1), 1.0)
    snow_weight = np.where(existing_ice > R1, qs / np.maximum(existing_ice, R1), 0.0)
    graupel_weight = np.where(existing_ice > R1, qg / np.maximum(existing_ice, R1), 0.0)
    qv = qv - deposition
    qi = qi + deposition * ice_weight
    qs = qs + deposition * snow_weight
    qg = qg + deposition * graupel_weight
    Ni = Ni + deposition * ice_weight / XM0I
    T = T + LSUB * ocp * deposition

    qvsi = _rsif(p, T)
    existing_ice = qi + qs + qg
    sublimation = np.minimum(existing_ice, 0.25 * np.maximum(qvsi - qv, 0.0))
    sublimation = np.where(T < T_0, sublimation, 0.0)
    ice_weight = np.where(existing_ice > R1, qi / np.maximum(existing_ice, R1), 0.0)
    snow_weight = np.where(existing_ice > R1, qs / np.maximum(existing_ice, R1), 0.0)
    graupel_weight = np.where(existing_ice > R1, qg / np.maximum(existing_ice, R1), 0.0)
    qi_before_sublimation = qi.copy()
    qv = qv + sublimation
    qi = qi - sublimation * ice_weight
    qs = qs - sublimation * snow_weight
    qg = qg - sublimation * graupel_weight
    Ni = np.maximum(0.0, Ni - Ni * sublimation * ice_weight / np.maximum(qi_before_sublimation, R1))
    T = T - LSUB * ocp * sublimation

    autoconv_source = np.maximum(qc - 1.0e-4, 0.0)
    autoconv = np.minimum(qc, autoconv_source * (1.0 - np.exp(-dt / 900.0)))
    accretion = np.minimum(qc - autoconv, 0.18 * qr * (1.0 - np.exp(-dt / 300.0)))
    transfer = np.maximum(0.0, autoconv + accretion)
    Nr = Nr + autoconv / np.maximum(4.0 / 3.0 * np.pi * 1000.0 * (80.0e-6) ** 3, 1.0e-6)
    qc = qc - transfer
    qr = qr + transfer

    qvs = _rslf(p, T)
    lvap = _lvap(T)
    ocp = _ocp(qv)
    evap = np.minimum(qr, np.minimum(0.20 * np.maximum(qvs - qv, 0.0), qr * (1.0 - np.exp(-dt / 900.0))))
    nr_loss = np.where(qr > 0.0, Nr * evap / np.maximum(qr, R1), 0.0)
    qv = qv + evap
    qr = qr - evap
    Nr = np.maximum(0.0, Nr - nr_loss)
    T = T - lvap * ocp * evap

    qv = np.maximum(qv, 1.0e-10)
    qc = np.where(qc <= R1, 0.0, np.maximum(qc, 0.0))
    qr = np.where(qr <= R1, 0.0, np.maximum(qr, 0.0))
    qi = np.where(qi <= R1, 0.0, np.maximum(qi, 0.0))
    qs = np.where(qs <= R1, 0.0, np.maximum(qs, 0.0))
    qg = np.where(qg <= R1, 0.0, np.maximum(qg, 0.0))
    Ni = np.where(qi <= R1, 0.0, np.maximum(Ni, 0.0))
    Nr = np.where(qr <= R1, 0.0, np.maximum(Nr, 0.0))
    T = np.maximum(T, 50.0)
    return {"qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "Ni": Ni, "Nr": Nr, "T": T, "rho": _rho(p, T, qv)}


def make_scenarios() -> tuple[dict[str, np.ndarray], float]:
    """Constructs the three required maritime, mixed-phase, and precipitating columns."""

    nz = 12
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    dt = 60.0
    p = np.stack(
        [
            96000.0 - 47000.0 * z,
            90000.0 - 52000.0 * z,
            94000.0 - 50000.0 * z,
        ]
    )
    T = np.stack(
        [
            290.0 - 18.0 * z,
            268.0 - 42.0 * z,
            282.0 - 32.0 * z,
        ]
    )
    qvs = _rslf(p, T)
    qv = np.stack([0.995 * qvs[0], 0.92 * qvs[1], 0.88 * qvs[2]])
    qc = np.stack([2.5e-4 * np.exp(-((z - 0.24) / 0.18) ** 2), 9.0e-5 * np.exp(-((z - 0.42) / 0.16) ** 2), 1.6e-4 * np.exp(-((z - 0.34) / 0.20) ** 2)])
    qr = np.stack([2.0e-6 + z * 0.0, 1.0e-6 + z * 0.0, 5.0e-4 * np.exp(-((z - 0.22) / 0.20) ** 2)])
    qi = np.stack([z * 0.0, 7.0e-5 * np.exp(-((z - 0.58) / 0.18) ** 2), 5.0e-5 * np.exp(-((z - 0.72) / 0.16) ** 2)])
    qs = np.stack([z * 0.0, 1.5e-4 * np.exp(-((z - 0.66) / 0.20) ** 2), 2.0e-4 * np.exp(-((z - 0.64) / 0.18) ** 2)])
    qg = np.stack([z * 0.0, 2.0e-5 * np.exp(-((z - 0.52) / 0.14) ** 2), 2.8e-4 * np.exp(-((z - 0.38) / 0.18) ** 2)])
    Ni = np.where(qi > 0.0, 2.0e5, 0.0).astype(np.float64)
    Nr = np.where(qr > 1.0e-8, 8.0e4, 0.0).astype(np.float64)
    fields = {"T": T, "p": p, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "Ni": Ni, "Nr": Nr}
    return fields, dt


def _sha256(path: Path) -> str:
    """Computes a lowercase SHA-256 digest for manifest file entries."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    """Returns the current git revision or branch name for manifest provenance."""

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "worker/gpt/m5-s1-thompson-microphysics-column"


def _variable(name: str, units: str, shape: tuple[int, ...], abs_tol: float, rel_tol: float, rationale: str) -> dict[str, Any]:
    """Builds one manifest variable record without schema drift."""

    return {
        "name": name,
        "units": units,
        "shape": list(shape),
        "staggering": "mass",
        "dtype": "float64",
        "tolerance_abs": float(abs_tol),
        "tolerance_rel": float(rel_tol),
        "tolerance_rationale": rationale,
        "tier_overrides": None,
    }


def write_fixture() -> dict[str, Any]:
    """Writes the sample NPZ and manifest in the M1 schema."""

    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields, dt = make_scenarios()
    output = reference_step_numpy(fields, dt)
    payload = {
        "input_T": fields["T"],
        "input_p": fields["p"],
        "input_rho": _rho(fields["p"], fields["T"], fields["qv"]),
        "input_dt": np.asarray([dt], dtype=np.float64),
    }
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr"):
        payload[f"input_{name}"] = fields[name]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T"):
        payload[f"output_{name}"] = output[name]
    np.savez_compressed(SAMPLE, **payload)
    sample_bytes = SAMPLE.stat().st_size
    if sample_bytes > 100_000:
        raise RuntimeError(f"{SAMPLE} is {sample_bytes} bytes, over the schema limit")

    shape = tuple(fields["T"].shape)
    variables = [
        _variable("input_T", "K", shape, 1.0e-10, 1.0e-12, "fp64 synthetic thermodynamic input"),
        _variable("input_p", "Pa", shape, 1.0e-8, 1.0e-12, "fp64 synthetic pressure input"),
        _variable("input_rho", "kg m-3", shape, 1.0e-12, 1.0e-12, "rho computed with WRF mp_gt_driver formula"),
        _variable("input_dt", "s", (1,), 0.0, 0.0, "static Thompson timestep"),
    ]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        variables.append(_variable(f"input_{name}", "kg kg-1", shape, 1.0e-12, 1.0e-12, "fp64 hydrometeor input"))
    for name in ("Ni", "Nr"):
        variables.append(_variable(f"input_{name}", "kg-1", shape, 1.0e-3, 1.0e-6, "fp64 number concentration input"))
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        variables.append(_variable(f"output_{name}", "kg kg-1", shape, 1.0e-10, 1.0e-8, "ADR-005 hydrometeor tolerance"))
    for name in ("Ni", "Nr"):
        variables.append(_variable(f"output_{name}", "kg-1", shape, 1.0e-3, 1.0e-6, "ADR-005 number concentration tolerance"))
    variables.append(_variable("output_T", "K", shape, 1.0e-8, 1.0e-10, "fp64 latent heating tolerance"))

    manifest = {
        "fixture_id": FIXTURE_ID,
        "source": "analytic",
        "source_commit": "module_mp_thompson.F.pre source-mapped Path B formulas lines 2040-2063, 3024-3273, 3456-3633, 4000-4152, 5444-5490",
        "wrf_version": "v4.7.1",
        "scenario": "three Thompson source/sink columns: maritime shallow cloud, cold mixed-phase, precipitating column; sedimentation disabled by construction",
        "created_utc": "2026-05-20T02:42:06Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "python scripts/m5_generate_thompson_fixture.py",
        "external_uri": None,
        "sample_slice_path": "fixtures/samples/analytic-thompson-column-v1.npz",
        "git_commit": _git_rev(),
        "license_notes": "Synthetic fixture generated in-repository from WRF Thompson source formulas; WRF source itself is not redistributed here.",
        "variables": variables,
        "files": [
            {
                "path": "fixtures/samples/analytic-thompson-column-v1.npz",
                "checksum_sha256": _sha256(SAMPLE),
                "bytes": sample_bytes,
                "external": False,
            }
        ],
    }
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {"sample": str(SAMPLE.relative_to(ROOT)), "manifest": str(MANIFEST.relative_to(ROOT)), "bytes": sample_bytes, "sha256": _sha256(SAMPLE), "path": "B", "wrf_source_exists": WRF_SOURCE.exists()}


def main() -> int:
    """CLI entry point used by the sprint validation command."""

    os.environ.setdefault("JAX_ENABLE_X64", "true")
    record = write_fixture()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
