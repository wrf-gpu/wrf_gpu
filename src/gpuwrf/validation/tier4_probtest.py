"""M6-S7 Tier-4 probtest-style tolerance prototype.

This module deliberately keeps the tolerance freeze separate from held-out
candidate evaluation. The M6 artifact is a prototype based on historical
deterministic Gen2 day-members; the full production ensemble is deferred to M7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, NamedTuple, Sequence

import numpy as np

from gpuwrf.io.gen2_accessor import DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.io.validation import domain_mask, lead_time_slice, load_gen2_var


DEFAULT_GEN2_ROOT = DEFAULT_M6_GEN2_RUN_DIR.parent
DEFAULT_DOMAIN = "d02"
DEFAULT_ENDING_CYCLE = "20260520_18z"
DEFAULT_HELDOUT_CYCLE = "20260519_18z"
DEFAULT_VARIABLES = ("U10", "V10", "T2", "qv2", "precip")
DEFAULT_LEADS_H = (6, 12, 24)
DEFAULT_TOLERANCE_FACTOR = 1.96
PROTOTYPE_LABEL = "M6 prototype; full ensemble at M7"

RUN_DIR_RE = re.compile(r"^(?P<cycle>\d{8}_\d{2}z)_l3_24h_(?P<created>\d{8}T\d{6}Z)$")
SURFACE_VAR_TO_WRF = {"U10": "U10", "V10": "V10", "T2": "T2", "qv2": "Q2"}
VARIABLE_UNITS = {
    "U10": "m s-1",
    "V10": "m s-1",
    "T2": "K",
    "qv2": "kg kg-1",
    "precip": "mm accumulated over lead window",
}


@dataclass(frozen=True)
class HistoricalMember:
    """One selected deterministic Gen2 day-member."""

    cycle: str
    created: str
    path: Path

    @property
    def run_id(self) -> str:
        return self.path.name


class _SurfaceOutputState(NamedTuple):
    u: object
    v: object
    theta: object
    qv: object
    p: object
    dz: object
    t_skin: object
    soil_moisture: object
    xland: object
    lakemask: object
    mavail: object
    roughness_m: object
    ustar: object


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def parse_historical_member(path: str | Path) -> HistoricalMember | None:
    target = Path(path)
    match = RUN_DIR_RE.match(target.name)
    if match is None:
        return None
    return HistoricalMember(match.group("cycle"), match.group("created"), target)


def select_historical_members(
    root: str | Path = DEFAULT_GEN2_ROOT,
    *,
    ending_cycle: str = DEFAULT_ENDING_CYCLE,
    count: int = 10,
    heldout_cycle: str | None = DEFAULT_HELDOUT_CYCLE,
    domain: str = DEFAULT_DOMAIN,
    required_leads_h: Sequence[int] | None = None,
) -> list[Path]:
    """Select latest deterministic wrf_l3 day-members ending at `ending_cycle`.

    Duplicate cycles are resolved by taking the latest created run directory.
    The M6-S2 held-out cycle is excluded by default so the candidate day is not
    part of the tolerance freeze sample.
    """

    root_path = Path(root)
    by_cycle: dict[str, HistoricalMember] = {}
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        member = parse_historical_member(child)
        if member is None:
            continue
        if member.cycle > ending_cycle:
            continue
        if heldout_cycle is not None and member.cycle == heldout_cycle:
            continue
        if required_leads_h is not None and not has_required_history_files(member.path, domain, required_leads_h):
            continue
        previous = by_cycle.get(member.cycle)
        if previous is None or member.created > previous.created:
            by_cycle[member.cycle] = member

    ordered = [by_cycle[cycle] for cycle in sorted(by_cycle)]
    selected = ordered[-count:]
    if len(selected) != count:
        raise ValueError(f"expected {count} historical members, found {len(selected)} under {root_path}")
    if selected[-1].cycle != ending_cycle:
        raise ValueError(f"selected sample does not end at pinned cycle {ending_cycle}: {selected[-1].cycle}")
    return [member.path for member in selected]


def has_required_history_files(path: str | Path, domain: str, leads_h: Sequence[int]) -> bool:
    """Return True when a Gen2 run has real wrfout history for init and leads."""

    try:
        run = Gen2Run(path)
        history = run.history_files(domain)
        required = (0, *[int(item) for item in leads_h])
        for lead_h in required:
            index = lead_time_slice(run, lead_h)
            if index >= len(history):
                return False
            if not history[index].name.startswith(f"wrfout_{domain}_"):
                return False
        return True
    except Exception:
        return False


def available_complete_historical_members(
    root: str | Path = DEFAULT_GEN2_ROOT,
    *,
    ending_cycle: str = DEFAULT_ENDING_CYCLE,
    heldout_cycle: str | None = DEFAULT_HELDOUT_CYCLE,
    domain: str = DEFAULT_DOMAIN,
    required_leads_h: Sequence[int] = DEFAULT_LEADS_H,
) -> list[Path]:
    """List every complete historical wrf_l3 member available for the requested leads."""

    root_path = Path(root)
    by_cycle: dict[str, HistoricalMember] = {}
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        member = parse_historical_member(child)
        if member is None:
            continue
        if member.cycle > ending_cycle:
            continue
        if heldout_cycle is not None and member.cycle == heldout_cycle:
            continue
        if not has_required_history_files(member.path, domain, required_leads_h):
            continue
        previous = by_cycle.get(member.cycle)
        if previous is None or member.created > previous.created:
            by_cycle[member.cycle] = member
    return [by_cycle[cycle].path for cycle in sorted(by_cycle)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
        "sha256": sha256_file(path),
    }


def _cycle_from_run_id(run_id: str) -> str:
    match = RUN_DIR_RE.match(run_id)
    return match.group("cycle") if match is not None else run_id.split("_l3_24h_", 1)[0]


def referenced_member_files(run: Gen2Run, domain: str, leads_h: Sequence[int]) -> list[Path]:
    files: dict[str, Path] = {}
    files["namelist.input"] = run.path / "namelist.input"
    files[f"wrfinput_{domain}"] = run.wrfinput_file(domain)
    for lead_h in (0, *leads_h):
        index = lead_time_slice(run, int(lead_h))
        history = run.history_files(domain)[index]
        files[f"wrfout_{domain}_{lead_h:03d}h"] = history
    return list(files.values())


def build_ensemble_member_manifest(
    member_paths: Sequence[str | Path],
    *,
    domain: str = DEFAULT_DOMAIN,
    leads_h: Sequence[int] = DEFAULT_LEADS_H,
    created_utc: str | None = None,
    ending_cycle: str = DEFAULT_ENDING_CYCLE,
    heldout_cycle: str | None = DEFAULT_HELDOUT_CYCLE,
) -> dict[str, Any]:
    """Build the AC1 manifest with per-member file SHA evidence."""

    members: list[dict[str, Any]] = []
    for index, path in enumerate(member_paths):
        run = Gen2Run(path)
        referenced = [_file_record(item) for item in referenced_member_files(run, domain, leads_h)]
        member_digest = hashlib.sha256(
            json.dumps(
                [{"name": item["name"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]} for item in referenced],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        grid = run.grid(domain)
        times = [item.isoformat() for item in run.time_axis(domain)]
        members.append(
            {
                "member_index": int(index),
                "run_id": run.run_id,
                "cycle": _cycle_from_run_id(run.run_id),
                "path": str(run.path),
                "domain": domain,
                "history_count": len(run.history_files(domain)),
                "history_start_utc": times[0] if times else None,
                "history_end_utc": times[-1] if times else None,
                "grid_shape_yx": [int(grid.mass_ny), int(grid.mass_nx)],
                "referenced_leads_h": [0, *[int(item) for item in leads_h]],
                "referenced_files": referenced,
                "member_sha256": member_digest,
            }
        )

    aggregate_digest = hashlib.sha256(
        json.dumps(
            [{"run_id": item["run_id"], "member_sha256": item["member_sha256"]} for item in members],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_type": "tier4_ensemble_member_manifest",
        "created_utc": created_utc or utc_now_iso(),
        "prototype_label": PROTOTYPE_LABEL,
        "selection_policy": (
            "latest unique deterministic Gen2 wrf_l3 18z day-members ending at pinned "
            f"{ending_cycle}; held-out M6-S2 cycle excluded before tolerance derivation"
        ),
        "ending_cycle": ending_cycle,
        "heldout_cycle_excluded": heldout_cycle,
        "sample_type": "historical deterministic operational sample, not a perturbed ensemble",
        "domain": domain,
        "leads_h": [int(item) for item in leads_h],
        "sample_size": len(members),
        "members": members,
        "aggregate_sha256": aggregate_digest,
    }


def load_member_field(run: Gen2Run, variable: str, lead_h: int, *, domain: str = DEFAULT_DOMAIN) -> np.ndarray:
    """Load one 2-D Gen2 validation field through the shared Gen2 accessor."""

    lead_index = lead_time_slice(run, lead_h)
    if variable in SURFACE_VAR_TO_WRF:
        wrf_name = SURFACE_VAR_TO_WRF[variable]
        return np.asarray(load_gen2_var(run, domain, wrf_name, lead_index), dtype=np.float64)
    if variable != "precip":
        raise KeyError(f"unsupported Tier-4 variable {variable!r}")

    variables = set(run.variables(domain))
    components = ["RAINC", "RAINNC"]
    if "RAINSH" in variables:
        components.append("RAINSH")
    missing = [name for name in components if name not in variables]
    if missing:
        raise KeyError(f"precip components missing for {run.run_id} {domain}: {missing}")
    lead_total = sum(np.asarray(load_gen2_var(run, domain, name, lead_index), dtype=np.float64) for name in components)
    init_total = sum(np.asarray(load_gen2_var(run, domain, name, 0), dtype=np.float64) for name in components)
    return lead_total - init_total


def build_stratum_masks(grid: Any, *, include_canary: bool = True) -> dict[str, np.ndarray]:
    """Return land, sea, and 500 m elevation-band masks from the shared helper."""

    masks: dict[str, np.ndarray] = {}
    if include_canary:
        masks["canary"] = np.asarray(domain_mask(grid, "canary"), dtype=bool)
    for name in ("land", "sea"):
        masks[name] = np.asarray(domain_mask(grid, name), dtype=bool)

    terrain = np.asarray(grid.static_field("HGT"), dtype=np.float64)
    max_band = int(math.floor(float(np.nanmax(terrain)) / 500.0))
    for band in range(max_band + 1):
        name = f"elevation_band_{band}"
        mask = np.asarray(domain_mask(grid, name), dtype=bool)
        if bool(np.any(mask)):
            masks[name] = mask
    return masks


def derive_stratified_tolerance_records(
    samples: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR,
) -> dict[str, dict[str, Any]]:
    """Derive scalar RMSE tolerances from per-grid-cell member variance."""

    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"samples must have shape (member, y, x); got {data.shape}")
    if data.shape[0] < 2:
        raise ValueError("at least two members are required for sample variance")
    variance_grid = np.nanvar(data, axis=0, ddof=1)
    std_grid = np.sqrt(variance_grid)

    records: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != data.shape[1:]:
            raise ValueError(f"mask {name!r} has shape {mask_array.shape}, expected {data.shape[1:]}")
        valid = mask_array & np.isfinite(std_grid)
        values = std_grid[valid]
        if values.size == 0:
            continue
        sigma = float(np.sqrt(np.mean(values * values)))
        records[name] = {
            "tolerance": float(tolerance_factor * sigma),
            "sigma_rms_member_std": sigma,
            "variance_mean": float(np.mean(variance_grid[valid])),
            "std_mean": float(np.mean(values)),
            "std_median": float(np.median(values)),
            "std_p95": float(np.percentile(values, 95.0)),
            "grid_cell_count": int(values.size),
            "finite": True,
        }
    return records


def derive_probtest_tolerances(
    member_paths: Sequence[str | Path],
    *,
    domain: str = DEFAULT_DOMAIN,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    leads_h: Sequence[int] = DEFAULT_LEADS_H,
    tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR,
    member_manifest_path: str = "artifacts/m6/tier4/ensemble_member_manifest.json",
    artifact_paths: Sequence[str] | None = None,
    freeze_time_utc: str | None = None,
) -> dict[str, Any]:
    """Compute AC2/AC3 tolerances from the selected historical sample."""

    if not member_paths:
        raise ValueError("member_paths cannot be empty")
    first_run = Gen2Run(member_paths[0])
    masks = build_stratum_masks(first_run.grid(domain))
    tolerances: dict[str, Any] = {}
    for variable in variables:
        tolerances[variable] = {}
        for lead_h in leads_h:
            fields = [load_member_field(Gen2Run(path), variable, int(lead_h), domain=domain) for path in member_paths]
            samples = np.stack(fields, axis=0)
            records = derive_stratified_tolerance_records(samples, masks, tolerance_factor=tolerance_factor)
            tolerances[variable][f"{int(lead_h)}h"] = records

    return {
        "artifact_type": "tier4_probtest_tolerances",
        "run_id": f"m6_s7_tier4_probtest_{utc_now_iso().replace(':', '').replace('+0000', 'Z')}",
        "status": "PASS",
        "prototype_label": PROTOTYPE_LABEL,
        "domain": domain,
        "sample_size": len(member_paths),
        "variables": list(variables),
        "leads_h": [int(item) for item in leads_h],
        "strata": list(masks.keys()),
        "member_manifest": member_manifest_path,
        "freeze_time_utc": freeze_time_utc or utc_now_iso(),
        "tolerance_factor": float(tolerance_factor),
        "method": {
            "sample_type": "10 deterministic historical Gen2 wrf_l3 day-members, not a perturbed ensemble",
            "variance_estimator": "per-grid-cell sample variance across members with ddof=1",
            "stratum_reduction": "RMS of per-grid-cell member standard deviation; tolerance = k * sigma_rms_member_std",
            "k": float(tolerance_factor),
            "k_rationale": "1.96 approximates a two-sided 95% normal interval for this M6 prototype.",
            "precip_definition": "(RAINC + RAINNC + optional RAINSH at lead) - same components at lead 0",
            "candidate_peek_policy": "tolerances are written before held-out candidate validation",
            "no_min_raw_cap_fudge": True,
        },
        "units": {name: VARIABLE_UNITS[name] for name in variables},
        "heldout_policy": {
            "heldout_cycle": DEFAULT_HELDOUT_CYCLE,
            "reason": "M6-S2 pinned GPU forecast day is excluded from the tolerance sample.",
        },
        "tolerances": tolerances,
        "artifact_paths": list(artifact_paths or []),
    }


def load_candidate_outputs(output_manifest: str | Path, *, root: str | Path | None = None) -> dict[int, Path]:
    manifest_path = Path(output_manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path(root) if root is not None else manifest_path.parent
    outputs: dict[int, Path] = {}
    for item in data.get("outputs", []):
        lead = int(round(float(item["lead_hours"])))
        path = Path(item["path"])
        if not path.is_absolute():
            path = base / path
        outputs[lead] = path
    return outputs


def _derive_surface_from_npz(arrays: dict[str, np.ndarray], run: Gen2Run, domain: str, variable: str) -> np.ndarray:
    import jax.numpy as jnp

    from gpuwrf.io.land_state import load_prescribed_land_state
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

    required = ("U", "V", "T", "QVAPOR", "P", "PH")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"cannot derive {variable}: candidate NPZ missing {missing}")
    land = load_prescribed_land_state(run, domain=domain, time=0)
    u = arrays["U"].astype(np.float64)
    v = arrays["V"].astype(np.float64)
    theta = arrays["T"].astype(np.float64) + 300.0
    qv = arrays["QVAPOR"].astype(np.float64)
    p = arrays["P"].astype(np.float64)
    ph = arrays["PH"].astype(np.float64)
    u_mass = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_mass = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    dz = np.maximum((ph[1:, :, :] - ph[:-1, :, :]) / 9.80665, 1.0)
    ustar = arrays.get("UST", np.zeros(u_mass.shape[1:], dtype=np.float64)).astype(np.float64)
    state = _SurfaceOutputState(
        u=jnp.moveaxis(jnp.asarray(u_mass), 0, -1),
        v=jnp.moveaxis(jnp.asarray(v_mass), 0, -1),
        theta=jnp.moveaxis(jnp.asarray(theta), 0, -1),
        qv=jnp.moveaxis(jnp.asarray(qv), 0, -1),
        p=jnp.moveaxis(jnp.asarray(p), 0, -1),
        dz=jnp.moveaxis(jnp.asarray(dz), 0, -1),
        t_skin=land.t_skin,
        soil_moisture=land.soil_moisture[0],
        xland=land.xland,
        lakemask=land.lakemask,
        mavail=land.mavail,
        roughness_m=land.roughness_m,
        ustar=jnp.asarray(ustar),
    )
    diag = surface_layer_with_diagnostics(state)
    by_var = {
        "U10": np.asarray(diag.u10, dtype=np.float64),
        "V10": np.asarray(diag.v10, dtype=np.float64),
        "T2": np.asarray(diag.t2, dtype=np.float64),
        "qv2": np.asarray(diag.q2, dtype=np.float64),
    }
    return by_var[variable]


def load_candidate_field(
    candidate_path: str | Path,
    variable: str,
    *,
    truth_shape: tuple[int, int],
    heldout_run: Gen2Run,
    domain: str = DEFAULT_DOMAIN,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or derive a candidate GPU field from an M6-S2/S3 NPZ output."""

    path = Path(candidate_path)
    with np.load(path, allow_pickle=False) as npz:
        arrays = {name: np.asarray(npz[name]) for name in npz.files}

    if variable in SURFACE_VAR_TO_WRF:
        key = SURFACE_VAR_TO_WRF[variable]
        if key in arrays:
            return arrays[key].astype(np.float64), {"path": str(path), "key": key, "method": "direct_npz"}
        return _derive_surface_from_npz(arrays, heldout_run, domain, variable), {
            "path": str(path),
            "key": key,
            "method": "derived_with_m6_s3_surface_layer_from_npz_core_state",
        }

    if variable != "precip":
        raise KeyError(f"unsupported candidate variable {variable!r}")
    for key in ("precip", "PRECIP", "RAIN_ACC", "rain_acc"):
        if key in arrays:
            return arrays[key].astype(np.float64), {"path": str(path), "key": key, "method": "direct_npz"}
    component_keys = [name for name in ("rain_acc", "snow_acc", "graupel_acc", "ice_acc") if name in arrays]
    if component_keys:
        total = sum(arrays[name].astype(np.float64) for name in component_keys)
        return total, {"path": str(path), "key": "+".join(component_keys), "method": "accumulator_components"}
    return np.zeros(truth_shape, dtype=np.float64), {
        "path": str(path),
        "key": None,
        "method": "zero_current_m6_gpu_precip_accumulator",
        "limitation": (
            "M6-S2/S3 NPZ outputs do not serialize precipitation accumulators, and the current Thompson "
            "coupler does not update State rain/snow/graupel/ice accumulators. Candidate precip is therefore "
            "the model's current zero-accumulation behavior, not an operational precip product."
        ),
    }


def _score(candidate: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(candidate) & np.isfinite(truth)
    diff = candidate[valid] - truth[valid]
    if diff.size == 0:
        raise ValueError("empty valid score mask")
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "grid_cell_count": int(diff.size),
    }


def validate_heldout_candidate(
    tolerances: dict[str, Any],
    *,
    heldout_run_path: str | Path,
    candidate_output_manifest: str | Path,
    domain: str = DEFAULT_DOMAIN,
    root: str | Path | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Evaluate the held-out M6-S2/S3 GPU candidate against frozen tolerances."""

    if not has_required_history_files(heldout_run_path, domain, [int(item) for item in tolerances["leads_h"]]):
        raise FileNotFoundError(
            f"held-out Gen2 run lacks real wrfout_{domain} history for leads {tolerances['leads_h']}: {heldout_run_path}"
        )
    heldout_run = Gen2Run(heldout_run_path)
    masks = build_stratum_masks(heldout_run.grid(domain))
    outputs = load_candidate_outputs(candidate_output_manifest, root=root)
    variables = list(tolerances["variables"])
    leads_h = [int(item) for item in tolerances["leads_h"]]
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for variable in variables:
        results[variable] = {}
        for lead_h in leads_h:
            if lead_h not in outputs:
                raise FileNotFoundError(f"candidate output manifest has no +{lead_h}h file")
            truth = load_member_field(heldout_run, variable, lead_h, domain=domain)
            candidate, source = load_candidate_field(
                outputs[lead_h],
                variable,
                truth_shape=truth.shape,
                heldout_run=heldout_run,
                domain=domain,
            )
            if candidate.shape != truth.shape:
                raise ValueError(f"{variable} +{lead_h}h candidate shape {candidate.shape} != truth {truth.shape}")
            lead_key = f"{lead_h}h"
            results[variable][lead_key] = {"candidate_source": source, "strata": {}}
            for stratum in tolerances["strata"]:
                if stratum not in masks:
                    continue
                score = _score(candidate, truth, masks[stratum])
                tolerance_record = tolerances["tolerances"][variable][lead_key][stratum]
                tolerance = float(tolerance_record["tolerance"])
                passed = bool(score["rmse"] <= tolerance)
                record = {
                    **score,
                    "tolerance": tolerance,
                    "pass": passed,
                    "units": VARIABLE_UNITS[variable],
                }
                results[variable][lead_key]["strata"][stratum] = record
                if not passed:
                    failures.append(
                        {
                            "variable": variable,
                            "lead_h": int(lead_h),
                            "stratum": stratum,
                            "rmse": score["rmse"],
                            "tolerance": tolerance,
                        }
                    )

    return {
        "artifact_type": "tier4_heldout_candidate_validation",
        "created_utc": created_utc or utc_now_iso(),
        "prototype_label": PROTOTYPE_LABEL,
        "status": "PASS" if not failures else "FAIL",
        "freeze_time_utc": tolerances["freeze_time_utc"],
        "candidate_evaluation_policy": "candidate loaded only after probtest_tolerances.json and tolerance_freeze_report.md are written",
        "heldout_run_id": heldout_run.run_id,
        "heldout_run_path": str(heldout_run.path),
        "candidate_output_manifest": str(candidate_output_manifest),
        "domain": domain,
        "variables": variables,
        "leads_h": leads_h,
        "results": results,
        "failures": failures,
    }


def build_cost_model(
    *,
    member_manifest: dict[str, Any],
    tolerances: dict[str, Any],
    member_manifest_path: str | Path,
    tolerance_path: str | Path,
    freeze_report_path: str | Path,
    candidate_output_manifest: str | Path | None = None,
    spacetime_budget_path: str | Path | None = None,
    m7_sizes: Sequence[int] = (100, 1000),
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build AC4 storage/runtime scaling model."""

    members = member_manifest["members"]
    history_bytes = [sum(int(file["size_bytes"]) for file in member["referenced_files"]) for member in members]
    avg_history_bytes = float(np.mean(history_bytes)) if history_bytes else 0.0
    npz_bytes = 0
    npz_files: list[str] = []
    if candidate_output_manifest is not None and Path(candidate_output_manifest).exists():
        outputs = load_candidate_outputs(candidate_output_manifest, root=Path.cwd())
        for lead_h, path in outputs.items():
            if lead_h in set(tolerances["leads_h"]) and path.exists():
                npz_bytes += path.stat().st_size
                npz_files.append(str(path))

    grid_shape = members[0]["grid_shape_yx"] if members else [0, 0]
    compact_npz_estimate = int(len(tolerances["variables"]) * len(tolerances["leads_h"]) * grid_shape[0] * grid_shape[1] * 4)
    runtime_s: float | None = None
    runtime_source: dict[str, Any] = {
        "status": "BLOCKED_ON_M6_S5_LIFTED_CAP_ARTIFACT",
        "note": "No M6-S5 full-domain batching verdict was visible in this worktree.",
    }
    if spacetime_budget_path is not None and Path(spacetime_budget_path).exists():
        budget = json.loads(Path(spacetime_budget_path).read_text(encoding="utf-8"))
        runtime_s = float(budget.get("extrapolated_24h_wall_s", budget.get("output_run_wall_s")))
        runtime_source = {
            "status": "PROVISIONAL_FROM_M6_S2_SPACETIME_BUDGET",
            "path": str(spacetime_budget_path),
            "wall_time_method": budget.get("wall_time_method"),
            "temporary_bytes_per_step": budget.get("temporary_bytes_per_step"),
            "host_device_transfer_bytes": budget.get("host_device_transfer_bytes"),
            "note": "Replace with M6-S5 lifted-cap config once its verdict artifact lands.",
        }

    scaling: dict[str, Any] = {}
    for size in m7_sizes:
        rel_sigma_uncertainty = math.sqrt(1.0 / (2.0 * (size - 1))) if size > 1 else None
        scaling[str(size)] = {
            "members": int(size),
            "history_references_bytes": int(avg_history_bytes * size),
            "gpu_npz_bytes_if_keep_requested_leads": int(npz_bytes * size) if npz_bytes else None,
            "compact_surface_npz_estimate_bytes": int(compact_npz_estimate * size),
            "serial_single_gpu_runtime_s": float(runtime_s * size) if runtime_s is not None else None,
            "serial_single_gpu_runtime_h": float(runtime_s * size / 3600.0) if runtime_s is not None else None,
            "relative_sigma_estimator_uncertainty": rel_sigma_uncertainty,
        }

    return {
        "artifact_type": "tier4_cost_model",
        "created_utc": created_utc or utc_now_iso(),
        "prototype_label": PROTOTYPE_LABEL,
        "status": "PROVISIONAL" if tolerances.get("status") == "PASS" else "BLOCKED",
        "blockers": list(tolerances.get("blockers", [])),
        "per_member": {
            "referenced_history_and_static_bytes_mean": int(avg_history_bytes),
            "referenced_history_and_static_bytes_by_member": history_bytes,
            "gpu_npz_requested_leads_bytes": int(npz_bytes) if npz_bytes else None,
            "gpu_npz_files_sample": npz_files,
            "compact_surface_npz_estimate_bytes": compact_npz_estimate,
            "manifest_and_report_bytes": int(
                sum(Path(path).stat().st_size for path in (member_manifest_path, tolerance_path, freeze_report_path) if Path(path).exists())
            ),
            "runtime_24h_gpu_s": runtime_s,
            "runtime_source": runtime_source,
        },
        "m7_scaling": scaling,
        "recommended_m7_ensemble_size": 100,
        "recommendation": (
            "Use 100 members for the first M7 full-ensemble dispatch: it reduces the idealized sigma-estimator "
            "relative uncertainty from about 24% at n=10 to about 7%, while keeping single-GPU serial runtime "
            "and storage practical. Reserve 1000 members for precipitation/regime-tail hardening after M7 cost gates pass."
        ),
        "statistical_confidence_tradeoff": {
            "n10_relative_sigma_estimator_uncertainty": math.sqrt(1.0 / (2.0 * (10 - 1))),
            "n100_relative_sigma_estimator_uncertainty": math.sqrt(1.0 / (2.0 * (100 - 1))),
            "n1000_relative_sigma_estimator_uncertainty": math.sqrt(1.0 / (2.0 * (1000 - 1))),
            "formula": "sqrt(1 / (2 * (n - 1))) for normal-sample standard deviation uncertainty",
        },
    }


def write_tolerance_freeze_report(
    path: str | Path,
    *,
    tolerances: dict[str, Any],
    member_manifest: dict[str, Any],
    cost_model_path: str | Path,
) -> None:
    members = ", ".join(member["cycle"] for member in member_manifest["members"])
    blocker_text = ""
    if tolerances.get("blockers"):
        blocker_lines = "\n".join(f"- {item}" for item in tolerances["blockers"])
        blocker_text = f"\n## Blockers\n\n{blocker_lines}\n"
    text = f"""# M6-S7 Tier-4 Probtest Tolerance Freeze

Status: {tolerances["status"]}
Prototype label: {PROTOTYPE_LABEL}
Freeze time UTC: {tolerances["freeze_time_utc"]}

## Choices Frozen Before Held-Out Candidate

- Sample: {member_manifest["sample_size"]} deterministic historical Gen2 wrf_l3 day-members ({members}); this is not a perturbed ensemble.
- Held-out exclusion: {tolerances["heldout_policy"]["heldout_cycle"]} was excluded before tolerance derivation because it is the M6-S2 pinned GPU candidate day.
- Variables: {", ".join(tolerances["variables"])}
- Leads: {", ".join(str(item) + "h" for item in tolerances["leads_h"])}
- Strata: {", ".join(tolerances["strata"])}
- Tolerance factor: k = {tolerances["tolerance_factor"]} ({tolerances["method"]["k_rationale"]})
- Variance method: {tolerances["method"]["variance_estimator"]}; {tolerances["method"]["stratum_reduction"]}.
- Precipitation: {tolerances["method"]["precip_definition"]}.

## Prototype Limits

This is an M6 prototype only; full ensemble at M7. Ten deterministic day-members give a useful operational spread estimate, but precipitation tails and humidity regime coverage remain weak. The cost model in `{cost_model_path}` gates M7 ensemble dispatch and currently treats the M6-S5 lifted-cap runtime as provisional unless the S5 verdict artifact is present.

## Candidate Separation

No held-out candidate field is needed to compute this file. Candidate validation is written separately after this report and `probtest_tolerances.json` exist, preserving the no-after-failure tolerance rule.
{blocker_text}
"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


__all__ = [
    "DEFAULT_DOMAIN",
    "DEFAULT_ENDING_CYCLE",
    "DEFAULT_GEN2_ROOT",
    "DEFAULT_HELDOUT_CYCLE",
    "DEFAULT_LEADS_H",
    "DEFAULT_TOLERANCE_FACTOR",
    "DEFAULT_VARIABLES",
    "PROTOTYPE_LABEL",
    "available_complete_historical_members",
    "build_cost_model",
    "build_ensemble_member_manifest",
    "build_stratum_masks",
    "derive_probtest_tolerances",
    "derive_stratified_tolerance_records",
    "has_required_history_files",
    "load_candidate_outputs",
    "load_member_field",
    "select_historical_members",
    "validate_heldout_candidate",
    "write_json",
    "write_tolerance_freeze_report",
]
