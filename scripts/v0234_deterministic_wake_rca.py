#!/usr/bin/env python3
"""Authenticate and localize the v0234 deterministic wake first-red.

This is a CPU-only evidence reducer.  It independently re-reads every retained
d03/WRF frame pair through the first red, reports all eight release fields, and
separates the 10-m diagnostic ratio from the prognostic lowest-level wind.  It
does not import JAX, touch the GPU, or modify model bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset

from scripts import v0234_deterministic_wake_admission as admission


MODEL_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
RUN_DIR = admission.LINEAGE_ROOT / (
    "nested_stage_omega_transport_470e6111_deterministic_wake_reference1"
)
BLOCKER = RUN_DIR / "full-run-blocker.json"
FAILURE = RUN_DIR / "failure/failure-proof.json"
FINAL_PAIR = RUN_DIR / "frame-pairs/d03-step-09000.json"
EXPECTED_EVIDENCE = {
    BLOCKER: (
        "728f2f19a9ceca8697c2c7f8a0fc267cf61431e761f214948f5869713224b219",
        "33b4e566c3947274701e70c44292a459da9eab54dd1948ba4899b40bb7333e91",
    ),
    FAILURE: (
        "2a4b1422087c437f7db42668e5a9d2c93564d58c208b0cc326d3880271f30a86",
        "2a7919249c78d5b85f753bfabb310124936a9bda1ac5fe38071d036736826d85",
    ),
    FINAL_PAIR: (
        "86306e3dc629c1e1d945b16fc12c5020067b52ce439a59e75af8083230ef4296",
        None,
    ),
}
FINAL_FRAME_SHA256 = (
    "5402f5b578b49c295737f8b2fc02ebaa810f4f0ba9e7fd49c0fbe07b87692898"
)
WRF_MYNN = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_mynn.F")
WRF_MYNN_SHA256 = (
    "86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3"
)
GPU_SURFACE = admission.REPO_ROOT / "src/gpuwrf/physics/surface_layer.py"
GPU_SURFACE_SHA256 = (
    "d2975455936cffadcaa9f7fcffae31ddb8e7845d9d6f07c782e99cc256dd2995"
)
GPU_COUPLER = admission.REPO_ROOT / "src/gpuwrf/coupling/physics_couplers.py"
GPU_MYNN = admission.REPO_ROOT / "src/gpuwrf/physics/mynn_pbl.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def file_row(path: Path, *, expected: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/symlinked evidence: {path}")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"evidence hash changed: {path}: expected={expected} actual={observed}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": observed,
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def authenticate_self_hashed(
    path: Path, *, file_sha256: str, canonical_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = file_row(path, expected=file_sha256)
    payload = read_json(path)
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != canonical_sha256 or observed != canonical_sha256:
        raise RuntimeError(f"canonical evidence hash changed: {path}")
    row["canonical_sha256"] = observed
    return payload, row


def rmse(candidate: np.ndarray, cpu: np.ndarray) -> float:
    if candidate.shape != cpu.shape:
        raise RuntimeError(f"shape mismatch: {candidate.shape} != {cpu.shape}")
    error = candidate - cpu
    if not bool(np.isfinite(error).all()):
        raise RuntimeError("nonfinite retained frame data")
    return float(np.sqrt(np.mean(error * error, dtype=np.float64)))


def destagger_low_wind(fields: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    u = fields["U"]
    v = fields["V"]
    return (
        0.5 * (u[0, :, :-1] + u[0, :, 1:]),
        0.5 * (v[0, :-1, :] + v[0, 1:, :]),
    )


def ten_m_ratio_decomposition(
    candidate: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Separate scalar 10-m reduction-ratio error from low-wind error."""

    candidate_u, candidate_v = destagger_low_wind(candidate)
    cpu_u, cpu_v = destagger_low_wind(cpu)
    tiny = np.finfo(np.float64).tiny
    candidate_speed = np.sqrt(candidate_u**2 + candidate_v**2)
    cpu_speed = np.sqrt(cpu_u**2 + cpu_v**2)
    candidate_10_speed = np.sqrt(candidate["U10"] ** 2 + candidate["V10"] ** 2)
    cpu_10_speed = np.sqrt(cpu["U10"] ** 2 + cpu["V10"] ** 2)
    candidate_ratio = candidate_10_speed / np.maximum(candidate_speed, tiny)
    cpu_ratio = cpu_10_speed / np.maximum(cpu_speed, tiny)
    ratio_only_v10 = cpu_v * candidate_ratio
    wind_only_v10 = candidate_v * cpu_ratio
    return {
        "actual_v10_rmse": rmse(candidate["V10"], cpu["V10"]),
        "ratio_only_v10_rmse": rmse(ratio_only_v10, cpu["V10"]),
        "wind_only_v10_rmse": rmse(wind_only_v10, cpu["V10"]),
        "candidate_ratio_vs_cpu_ratio_rmse": rmse(candidate_ratio, cpu_ratio),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    if not math.isfinite(value):
        raise RuntimeError("nonfinite correlation")
    return value


def momentum_fingerprint(
    candidate: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_u, candidate_v = destagger_low_wind(candidate)
    cpu_u, cpu_v = destagger_low_wind(cpu)
    u_error = candidate_u - cpu_u
    v_error = candidate_v - cpu_v
    v_full_error = candidate["V"] - cpu["V"]
    ny, nx = v_error.shape
    y, x = np.ogrid[:ny, :nx]
    distance = np.minimum.reduce(
        [
            np.broadcast_to(y, (ny, nx)),
            np.broadcast_to(x, (ny, nx)),
            np.broadcast_to(ny - 1 - y, (ny, nx)),
            np.broadcast_to(nx - 1 - x, (ny, nx)),
        ]
    )
    return {
        "v10_error_vs_lowest_mass_v_error_correlation": _correlation(
            candidate["V10"] - cpu["V10"], v_error
        ),
        "u10_error_vs_lowest_mass_u_error_correlation": _correlation(
            candidate["U10"] - cpu["U10"], u_error
        ),
        "lowest_mass_v_rmse": rmse(candidate_v, cpu_v),
        "lowest_mass_u_rmse": rmse(candidate_u, cpu_u),
        "v_vertical_rmse": [
            float(np.sqrt(np.mean(level * level, dtype=np.float64)))
            for level in v_full_error
        ],
        "lowest_v_ring1_rms": float(
            np.sqrt(np.mean(v_error[distance == 1] ** 2, dtype=np.float64))
        ),
        "lowest_v_interior5plus_rms": float(
            np.sqrt(np.mean(v_error[distance >= 5] ** 2, dtype=np.float64))
        ),
        "v10_ring1_rms": float(
            np.sqrt(
                np.mean(
                    (candidate["V10"] - cpu["V10"])[distance == 1] ** 2,
                    dtype=np.float64,
                )
            )
        ),
        "v10_interior5plus_rms": float(
            np.sqrt(
                np.mean(
                    (candidate["V10"] - cpu["V10"])[distance >= 5] ** 2,
                    dtype=np.float64,
                )
            )
        ),
    }


def read_fields(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with Dataset(path) as dataset:
        fields = {
            name: np.asarray(dataset.variables[name][0], dtype=np.float64)
            for name in STRICT_FIELDS
        }
        attrs = {
            name: (
                dataset.getncattr(name).item()
                if isinstance(dataset.getncattr(name), np.generic)
                else dataset.getncattr(name)
            )
            for name in (
                "DX",
                "GRID_ID",
                "SF_SFCLAY_PHYSICS",
                "SF_SURFACE_PHYSICS",
                "BL_PBL_PHYSICS",
            )
            if name in dataset.ncattrs()
        }
    return fields, attrs


def source_authority() -> dict[str, Any]:
    wrf = file_row(WRF_MYNN, expected=WRF_MYNN_SHA256)
    gpu = file_row(GPU_SURFACE, expected=GPU_SURFACE_SHA256)
    wrf_text = WRF_MYNN.read_text()
    gpu_text = GPU_SURFACE.read_text()
    wrf_tokens = (
        "U10(I)=U1D(I)*PSIX10/PSIX",
        "V10(I)=V1D(I)*PSIX10/PSIX",
        "U10(I)=U1D(I)*log(10./ZNTstoch(I))/log(ZA(I)/ZNTstoch(I))",
        "V10(I)=V1D(I)*log(10./ZNTstoch(I))/log(ZA(I)/ZNTstoch(I))",
    )
    gpu_tokens = (
        "ratio10 = jnp.where((za > 7.0) & (za < 13.0), ratio10_neutral, ratio10_stab)",
        "u10 = u0 * ratio10",
        "v10 = v0 * ratio10",
    )
    if not all(token in wrf_text for token in wrf_tokens):
        raise RuntimeError("pristine WRF MYNN 10-m algebra changed")
    if not all(token in gpu_text for token in gpu_tokens):
        raise RuntimeError("GPU MYNN 10-m algebra changed")
    tree = subprocess.run(
        ("git", "rev-parse", "HEAD:src/gpuwrf"),
        cwd=admission.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tree != MODEL_TREE:
        raise RuntimeError(f"model tree changed: {tree}")
    return {
        "truth": "pristine WRF v4.7.1 source and retained CPU WRF frames",
        "model_tree": tree,
        "wrf_mynn_surface_source": {**wrf, "required_tokens": list(wrf_tokens)},
        "gpu_surface_source": {**gpu, "required_tokens": list(gpu_tokens)},
        "gpu_momentum_sources": {
            "coupler": file_row(GPU_COUPLER),
            "mynn_kernel": file_row(GPU_MYNN),
        },
    }


def _pair_step(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[1])


def _authenticate_terminal_evidence() -> dict[str, Any]:
    blocker, blocker_row = authenticate_self_hashed(
        BLOCKER,
        file_sha256=EXPECTED_EVIDENCE[BLOCKER][0],
        canonical_sha256=EXPECTED_EVIDENCE[BLOCKER][1] or "",
    )
    failure, failure_row = authenticate_self_hashed(
        FAILURE,
        file_sha256=EXPECTED_EVIDENCE[FAILURE][0],
        canonical_sha256=EXPECTED_EVIDENCE[FAILURE][1] or "",
    )
    final_pair_row = file_row(FINAL_PAIR, expected=EXPECTED_EVIDENCE[FINAL_PAIR][0])
    final_pair = read_json(FINAL_PAIR)
    detail = json.loads(str((failure.get("failure") or {}).get("detail", "{}")))
    metric = detail.get("metric_signature") or {}
    fingerprint = detail.get("spatial_fingerprint") or {}
    if (
        blocker.get("verdict") != "FULL_18H_BLOCKED"
        or blocker.get("failure_code") != "INCREMENTAL_FRAME_PAIR"
        or blocker.get("model_or_numerical_edit") is not False
        or failure.get("status") != "ATOMIC_FAILURE_CAPTURED"
        or (failure.get("failure") or {}).get("code") != "INCREMENTAL_FRAME_PAIR"
        or detail.get("classification") != "NEW_SCIENTIFIC_RED_OR_CHANGED_WAKE_SIGNATURE"
        or detail.get("record_and_continue_for_late_ni_only") is not False
        or metric.get("checks", {}).get("V10_inside_authenticated_range") is not False
        or metric.get("checks", {}).get("V_inside_authenticated_envelope") is not True
        or fingerprint.get("policy", {}).get("passed") is not True
        or final_pair.get("candidate", {}).get("sha256") != FINAL_FRAME_SHA256
    ):
        raise RuntimeError("first-red terminal semantics changed")
    return {
        "blocker": blocker_row,
        "failure": failure_row,
        "final_pair": final_pair_row,
        "failure_detail": detail,
    }


def build_payload() -> dict[str, Any]:
    terminal = _authenticate_terminal_evidence()
    authority, authority_row = admission.authenticate_authority()
    pair_paths = sorted((RUN_DIR / "frame-pairs").glob("d03-step-*.json"))
    if [_pair_step(path) for path in pair_paths] != list(range(0, 9001, 200)):
        raise RuntimeError("d03 retained trajectory is not exactly steps 0..9000/200")

    trajectory: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for pair_path in pair_paths:
        pair = read_json(pair_path)
        candidate_path = Path(str(pair.get("candidate", {}).get("path", "")))
        cpu_path = Path(str(pair.get("cpu", {}).get("path", "")))
        candidate_row = file_row(
            candidate_path, expected=str(pair.get("candidate", {}).get("sha256", ""))
        )
        cpu_row = file_row(cpu_path, expected=str(pair.get("cpu", {}).get("sha256", "")))
        candidate, candidate_attrs = read_fields(candidate_path)
        cpu, cpu_attrs = read_fields(cpu_path)
        metrics = {name: rmse(candidate[name], cpu[name]) for name in STRICT_FIELDS}
        recorded = (pair.get("d03_full_pair") or {}).get("strict_rmse") or {}
        if set(recorded) != set(STRICT_FIELDS) or any(
            not math.isclose(metrics[name], float(recorded[name]), rel_tol=0.0, abs_tol=5e-13)
            for name in STRICT_FIELDS
        ):
            raise RuntimeError(f"independent strict RMSE mismatch: {pair_path}")
        step = _pair_step(pair_path)
        fingerprint = momentum_fingerprint(candidate, cpu)
        row = {
            "step": step,
            "valid_time": (pair.get("d03_full_pair") or {}).get("valid_time"),
            "strict_rmse": metrics,
            "v10_vs_low_v_error_correlation": fingerprint[
                "v10_error_vs_lowest_mass_v_error_correlation"
            ],
            "u10_vs_low_u_error_correlation": fingerprint[
                "u10_error_vs_lowest_mass_u_error_correlation"
            ],
            "candidate_frame_sha256": candidate_row["file_sha256"],
            "cpu_frame_sha256": cpu_row["file_sha256"],
        }
        trajectory.append(row)
        retained_rows.append(
            {
                "step": step,
                "pair": file_row(pair_path),
                "candidate": candidate_row,
                "cpu": cpu_row,
            }
        )
        if step in {0, 200, 8800, 9000}:
            selected[str(step)] = {
                "strict_rmse": metrics,
                "momentum_fingerprint": fingerprint,
                "ten_m_ratio_decomposition": (
                    ten_m_ratio_decomposition(candidate, cpu) if step else None
                ),
                "candidate_attributes": candidate_attrs,
                "cpu_attributes": cpu_attrs,
            }

    initial = selected["0"]["strict_rmse"]
    first = selected["200"]
    final_detail = terminal["failure_detail"]
    final_metric = final_detail["metric_signature"]
    final_spatial = final_detail["spatial_fingerprint"]
    v10_excess = (
        float(final_metric["strict_rmse"]["V10"]) - admission.V10_ENVELOPE[1]
    )
    first_decomposition = first["ten_m_ratio_decomposition"]
    if not (
        initial["U"] < 1e-5
        and initial["V"] < 1e-5
        and first_decomposition["wind_only_v10_rmse"]
        > 4.0 * first_decomposition["ratio_only_v10_rmse"]
        and first["momentum_fingerprint"][
            "v10_error_vs_lowest_mass_v_error_correlation"
        ] > 0.9
        and final_spatial.get("policy", {}).get("passed") is True
        and v10_excess > 0.0
    ):
        raise RuntimeError("wake RCA discriminator did not support the localization")

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.deterministic-wake-rca.v1",
        "verdict": (
            "PROGNOSTIC_LOWEST_LEVEL_MOMENTUM_FIRST_INTERVAL_LOCALIZED__"
            "NO_MODEL_FIX_JUSTIFIED"
        ),
        "terminal_evidence": {
            key: value for key, value in terminal.items() if key != "failure_detail"
        },
        "authority": {**authority_row, "authority_sha256": authority["authority_sha256"]},
        "source_authority": source_authority(),
        "retained_trajectory": {
            "domain": "d03",
            "steps": "0..9000 inclusive every 200",
            "frame_pair_count": len(pair_paths),
            "all_eight_fields_independently_recomputed": True,
            "strict_fields": list(STRICT_FIELDS),
            "rows": trajectory,
            "artifact_rows": retained_rows,
            "artifact_inventory_sha256": canonical_digest(retained_rows),
        },
        "selected_discriminators": selected,
        "first_red": {
            "step": 9000,
            "finite": True,
            "static_exact": True,
            "red_fields": final_metric["red_fields"],
            "strict_rmse": final_metric["strict_rmse"],
            "V_authenticated_envelope": list(admission.V_ENVELOPE),
            "V_inside_authenticated_envelope": True,
            "V10_authenticated_range": list(admission.V10_ENVELOPE),
            "V10_excess_above_range": v10_excess,
            "spatial_policy_passed": True,
            "spatial_fingerprint_sha256": final_spatial["fingerprint_sha256"],
            "same_retained_wake_mechanism": True,
            "admission_stopped_fail_closed": True,
            "release_gate_green": False,
            "tolerance_changed": False,
            "waiver_or_reclassification": False,
        },
        "causal_localization": {
            "earliest_authenticated_green_input": (
                "d03 step 0 U/V are effectively exact against CPU WRF"
            ),
            "earliest_authenticated_discrepant_output": (
                "d03 step 200 (00:20) has low-level interior U/V error inherited by U10/V10"
            ),
            "bracket": "first coupled d03 steps 1..200 (00:00..00:20)",
            "v10_is_downstream_of_lowest_level_v": True,
            "surface_ratio_is_not_dominant": True,
            "nest_ring_is_not_dominant": True,
            "exact_remaining_split": (
                "MYNN surface/PBL momentum tendency versus the remaining dry-dycore/"
                "nest-coupled momentum update inside steps 1..200"
            ),
            "model_fix_justified": False,
            "reason_no_fix": (
                "Retained 20-minute frames do not expose operator-boundary tendencies; "
                "editing either branch would outrun WRF-v4 evidence."
            ),
        },
        "exact_next_kimi_action": {
            "owner": "Kimi K3 thinking-max (alternation after this science red)",
            "objective": (
                "Audit this proof, then isolate the first divergent momentum operator "
                "inside d03 steps 1..200 before proposing any model edit."
            ),
            "procedure": [
                "Authenticate this proof, the retained step-0/200 frames, model tree "
                f"{MODEL_TREE}, and staged autotune pin edd3b1271dbc59d2998cfc25a062c54e"
                "d23cce54e4bd740bbafe625f9bcb222f.",
                "Create a new sprint contract before code. Run pristine WRF v4.7.1 for "
                "the real d03 first interval with output-neutral savepoints at the "
                "incoming momentum state, raw MYNN RUBLTEN/RVBLTEN, post-PBL momentum, "
                "and post-dry-dycore/nest momentum.",
                "Run one pinned GPU short replay under canonical lock-v2 PREEMPT, adding "
                "matching output-neutral/source-leaf evidence only; report T,U,V,W,T2,"
                "U10,V10,PSFC and prove no loop host/device transfer in production code.",
                "If step 1 is not divergent, bisect saved steps in 1..200 until the first "
                "divergent update is bracketed. Only WRF-source algebra plus that matching "
                "savepoint may admit a correction."
            ],
            "prediction": (
                "The first material V/V10 error will appear in prognostic low-level "
                "momentum before the surface-layer U10/V10 diagnostic."
            ),
            "falsifier": (
                "Post-momentum GPU and WRF savepoints remain matching while the standalone "
                "10-m diagnostic first diverges; that would reopen surface-layer algebra."
            ),
        },
        "gpu_actions": 0,
        "model_or_numerical_edit": False,
        "release_gate_green": False,
    }
    payload["proof_sha256"] = canonical_digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload()
    atomic_write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "output": str(args.output.resolve()),
                "file_sha256": sha256_file(args.output.resolve()),
                "proof_sha256": payload["proof_sha256"],
                "frame_pair_count": payload["retained_trajectory"]["frame_pair_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
