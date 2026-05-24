"""Schema objects for WRF small-step savepoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


SCHEMA_VERSION = "m6b0r-savepoint-v1"
SAVEPOINT_FORMAT = "hdf5-savepoint-v1"
VALID_STAGGERS = {"mass", "u", "v", "w", "eta-half", "eta-full", "scalar"}
VALID_BOUNDARIES = {
    "calc_coef_w_pre",
    "calc_coef_w_post",
    "small_step_prep_post",
    "advance_mu_t_pre",
    "advance_mu_t_post",
    "advance_uv_post",
    "advance_w_rhs_ready",
    "advance_w_raw_w",
    "advance_w_tridiag_fwd",
    "advance_w_tridiag_back",
    "advance_w_rayleigh",
    "advance_w_ph_final",
    "calc_p_rho_post",
    "small_step_finish_post",
    "acoustic_substep_boundary",
    "rk_stage_boundary",
    # M6B0 compatibility aliases.
    "coefficient_construction",
    "acoustic_substep_start",
    "acoustic_substep_end",
    "rk_stage_end",
}
TOLERANCE_LADDER_PATH = Path(__file__).with_name("tolerance_ladder.json")


@dataclass(frozen=True)
class VariableMetadata:
    """Metadata for one savepoint variable."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    stagger: str
    units: str
    provenance: str
    role: Literal["input", "expected", "diagnostic"] = "input"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VariableMetadata.name must be non-empty")
        if self.stagger not in VALID_STAGGERS:
            raise ValueError(f"invalid stagger for {self.name}: {self.stagger}")
        if not self.units:
            raise ValueError(f"VariableMetadata.units must be non-empty for {self.name}")
        if not self.provenance:
            raise ValueError(f"VariableMetadata.provenance must be non-empty for {self.name}")
        np.dtype(self.dtype)
        if any(int(dim) < 0 for dim in self.shape):
            raise ValueError(f"VariableMetadata.shape must be non-negative for {self.name}")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stagger": self.stagger,
            "units": self.units,
            "provenance": self.provenance,
            "role": self.role,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "VariableMetadata":
        return cls(
            name=str(payload["name"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            stagger=str(payload["stagger"]),
            units=str(payload["units"]),
            provenance=str(payload["provenance"]),
            role=payload.get("role", "input"),
        )


@dataclass(frozen=True)
class SavepointMetadata:
    """Run and operator metadata required to compare WRF and JAX savepoints."""

    run_id: str
    wrf_version: str
    wrf_commit: str
    namelist_hash: str
    source_path: str
    domain_index: int
    tier: str
    operator: str
    boundary: str
    dt_seconds: float
    rk_stage_index: int
    acoustic_substep_index: int
    map_factors: dict[str, Any]
    vertical_grid: dict[str, Any]
    variables: dict[str, VariableMetadata]
    schema_version: str = SCHEMA_VERSION
    file_format: str = SAVEPOINT_FORMAT
    sanitizer_mode: Literal["off"] = "off"
    created_utc: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.file_format != SAVEPOINT_FORMAT:
            raise ValueError(f"unsupported file_format: {self.file_format}")
        for name in ("run_id", "wrf_version", "wrf_commit", "namelist_hash", "tier", "operator", "boundary"):
            if not str(getattr(self, name)):
                raise ValueError(f"SavepointMetadata.{name} must be non-empty")
        if self.boundary not in VALID_BOUNDARIES:
            raise ValueError(f"unsupported savepoint boundary: {self.boundary}")
        if int(self.domain_index) < 1:
            raise ValueError("domain_index must be >= 1")
        if float(self.dt_seconds) <= 0.0:
            raise ValueError("dt_seconds must be positive")
        if int(self.rk_stage_index) < 0 or int(self.acoustic_substep_index) < 0:
            raise ValueError("stage and substep indexes must be non-negative")
        if self.sanitizer_mode != "off":
            raise ValueError("M6B0 savepoints only accept sanitizer_mode='off'")
        if not self.map_factors:
            raise ValueError("map_factors metadata is required")
        if not self.vertical_grid:
            raise ValueError("vertical_grid metadata is required")
        if not self.variables:
            raise ValueError("at least one variable is required")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "file_format": self.file_format,
            "run_id": self.run_id,
            "wrf_version": self.wrf_version,
            "wrf_commit": self.wrf_commit,
            "namelist_hash": self.namelist_hash,
            "source_path": self.source_path,
            "domain_index": self.domain_index,
            "tier": self.tier,
            "operator": self.operator,
            "boundary": self.boundary,
            "dt_seconds": self.dt_seconds,
            "rk_stage_index": self.rk_stage_index,
            "acoustic_substep_index": self.acoustic_substep_index,
            "map_factors": self.map_factors,
            "vertical_grid": self.vertical_grid,
            "variables": {name: item.to_json() for name, item in self.variables.items()},
            "sanitizer_mode": self.sanitizer_mode,
            "created_utc": self.created_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SavepointMetadata":
        variables = {
            str(name): VariableMetadata.from_json(item)
            for name, item in dict(payload["variables"]).items()
        }
        return cls(
            schema_version=str(payload["schema_version"]),
            file_format=str(payload["file_format"]),
            run_id=str(payload["run_id"]),
            wrf_version=str(payload["wrf_version"]),
            wrf_commit=str(payload["wrf_commit"]),
            namelist_hash=str(payload["namelist_hash"]),
            source_path=str(payload.get("source_path", "")),
            domain_index=int(payload["domain_index"]),
            tier=str(payload["tier"]),
            operator=str(payload["operator"]),
            boundary=str(payload["boundary"]),
            dt_seconds=float(payload["dt_seconds"]),
            rk_stage_index=int(payload["rk_stage_index"]),
            acoustic_substep_index=int(payload["acoustic_substep_index"]),
            map_factors=dict(payload["map_factors"]),
            vertical_grid=dict(payload["vertical_grid"]),
            variables=variables,
            sanitizer_mode=payload.get("sanitizer_mode", "off"),
            created_utc=str(payload.get("created_utc", "")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class Savepoint:
    """A metadata object plus named NumPy arrays."""

    metadata: SavepointMetadata
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> None:
        missing = set(self.metadata.variables) - set(self.arrays)
        extra = set(self.arrays) - set(self.metadata.variables)
        if missing:
            raise ValueError(f"savepoint missing arrays: {sorted(missing)}")
        if extra:
            raise ValueError(f"savepoint has arrays without metadata: {sorted(extra)}")
        for name, variable in self.metadata.variables.items():
            array = np.asarray(self.arrays[name])
            if tuple(array.shape) != tuple(variable.shape):
                raise ValueError(f"{name} shape mismatch: {array.shape} != {variable.shape}")
            if np.dtype(array.dtype) != np.dtype(variable.dtype):
                raise ValueError(f"{name} dtype mismatch: {array.dtype} != {variable.dtype}")


def load_tolerance_ladder(path: str | Path = TOLERANCE_LADDER_PATH) -> dict[str, Any]:
    """Load the committed machine-readable comparison tolerance ladder."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "fields" not in payload or not isinstance(payload["fields"], dict):
        raise ValueError("tolerance ladder must contain a fields object")
    factor = float(payload.get("perturbation_rule", {}).get("minimum_factor_over_tolerance", 10.0))
    if factor < 10.0:
        raise ValueError("perturbation magnitude rule must be at least 10x tolerance")
    for field_name, entry in payload["fields"].items():
        for required in ("units", "dtype", "abs", "rel", "ulp", "accumulation_exception"):
            if required not in entry:
                raise ValueError(f"tolerance ladder field {field_name} missing {required}")
        if entry["abs"] is None and entry["rel"] is None and entry["ulp"] is None:
            raise ValueError(f"tolerance ladder field {field_name} has no threshold")
    return payload


def tolerance_for_field(field_name: str, path: str | Path = TOLERANCE_LADDER_PATH) -> dict[str, Any]:
    """Return one field tolerance entry from the ladder."""

    ladder = load_tolerance_ladder(path)
    try:
        return dict(ladder["fields"][field_name])
    except KeyError as exc:
        raise KeyError(f"no tolerance entry for field: {field_name}") from exc
