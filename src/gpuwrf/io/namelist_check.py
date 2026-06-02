"""Fail-fast support check for WRF namelist/config options.

This module is deliberately conservative: a selected nonzero physics/dynamics
option is accepted only when this branch has a faithful implementation path for
that option. Disabled options (usually ``0``) remain valid for dry/operator gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SupportedOption:
    """One supported-option registry entry."""

    key: str
    supported_values: frozenset[Any]
    implemented: str
    action: str


@dataclass(frozen=True)
class UnsupportedSelection:
    """A selected namelist/config value outside the faithful registry."""

    key: str
    location: str
    value: Any
    supported_values: tuple[Any, ...]
    implemented: str
    action: str
    domain_index: int | None = None


class UnsupportedNamelistOption(ValueError):
    """Raised when a namelist/config selects an unsupported option."""

    def __init__(self, selections: list[UnsupportedSelection]) -> None:
        self.selections = tuple(selections)
        super().__init__(_format_error(selections))


SUPPORTED_OPTIONS: dict[str, SupportedOption] = {
    # Physics suite wired in runtime.operational_mode and coupling.physics_couplers.
    "mp_physics": SupportedOption(
        key="mp_physics",
        supported_values=frozenset({0, 8}),
        implemented="0=disabled, 8=Thompson microphysics",
        action="Use mp_physics=8 for Thompson or 0 for a dry/no-microphysics gate.",
    ),
    "cu_physics": SupportedOption(
        key="cu_physics",
        supported_values=frozenset({0}),
        implemented="0=disabled; KF(1) is not implemented on this branch",
        action="Set cu_physics=0 or merge a faithful KF(1) implementation before enabling cumulus.",
    ),
    "bl_pbl_physics": SupportedOption(
        key="bl_pbl_physics",
        supported_values=frozenset({0, 5}),
        implemented="0=disabled, 5=MYNN PBL",
        action="Use bl_pbl_physics=5 for MYNN or 0 when PBL is intentionally disabled.",
    ),
    "sf_sfclay_physics": SupportedOption(
        key="sf_sfclay_physics",
        supported_values=frozenset({0, 5}),
        implemented="0=disabled, 5=MYNN revised surface layer / sfclayrev path",
        action="Use sf_sfclay_physics=5 for the MYNN-sfclayrev path or 0 when disabled.",
    ),
    "sf_surface_physics": SupportedOption(
        key="sf_surface_physics",
        supported_values=frozenset({0, 4}),
        implemented="0=disabled, 4=Noah-MP land surface",
        action="Use sf_surface_physics=4 for Noah-MP or 0 when the land surface is disabled.",
    ),
    "ra_sw_physics": SupportedOption(
        key="ra_sw_physics",
        supported_values=frozenset({0, 4}),
        implemented="0=disabled, 4=RRTMG shortwave",
        action="Use ra_sw_physics=4 for RRTMG SW or 0 when radiation is disabled.",
    ),
    "ra_lw_physics": SupportedOption(
        key="ra_lw_physics",
        supported_values=frozenset({0, 4}),
        implemented="0=disabled, 4=RRTMG longwave",
        action="Use ra_lw_physics=4 for RRTMG LW or 0 when radiation is disabled.",
    ),
    # Runtime/dynamics controls exposed by OperationalNamelist.
    "rk_order": SupportedOption(
        key="rk_order",
        supported_values=frozenset({3}),
        implemented="3=WRF RK3 outer loop",
        action="Use rk_order=3; other RK orders are not wired faithfully.",
    ),
    "diff_6th_opt": SupportedOption(
        key="diff_6th_opt",
        supported_values=frozenset({0, 2}),
        implemented="0=off, 2=WRF monotonic sixth-order horizontal filter",
        action="Use diff_6th_opt=2 for the operational filter or 0 when disabled.",
    ),
    "diff_opt": SupportedOption(
        key="diff_opt",
        supported_values=frozenset({0, 2}),
        implemented="0=off, 2=constant-K diffusion path when configured",
        action="Use diff_opt=2/km_opt=1 only with the implemented constant-K path, or 0.",
    ),
    "km_opt": SupportedOption(
        key="km_opt",
        supported_values=frozenset({0, 1}),
        implemented="0=off, 1=constant-K coefficient path",
        action="Use km_opt=1 with diff_opt=2 for constant-K diffusion, or 0.",
    ),
    "w_damping": SupportedOption(
        key="w_damping",
        supported_values=frozenset({0, 1}),
        implemented="0=off, 1=WRF vertical-CFL w damping",
        action="Use w_damping=1 or 0; no other WRF w_damping option is implemented.",
    ),
    "damp_opt": SupportedOption(
        key="damp_opt",
        supported_values=frozenset({0, 3}),
        implemented="0=off, 3=upper-level Rayleigh w damping",
        action="Use damp_opt=3 for the implemented Rayleigh path or 0.",
    ),
    "sf_urban_physics": SupportedOption(
        key="sf_urban_physics",
        supported_values=frozenset({0}),
        implemented="0=disabled; urban canopy physics is not implemented",
        action="Set sf_urban_physics=0 for this branch.",
    ),
}


def validate_supported_namelist(config: Any) -> None:
    """Raise if ``config`` selects a physics/dynamics option outside the registry.

    ``config`` may be a flat mapping, a nested WRF-style mapping such as
    ``{"physics": {"mp_physics": [8, 8]}}``, an object/dataclass with matching
    attributes, or a path to a simple WRF namelist file. Missing keys are ignored:
    this checker validates selected options, not namelist completeness.
    """

    config_obj = _coerce_config(config)
    failures: list[UnsupportedSelection] = []
    for key, spec in SUPPORTED_OPTIONS.items():
        found = _lookup(config_obj, key)
        if found is None:
            continue
        location, raw = found
        values = _domain_values(raw)
        for idx, value in enumerate(values):
            normalized = _normalize_value(value)
            if normalized in spec.supported_values:
                continue
            failures.append(
                UnsupportedSelection(
                    key=key,
                    location=location,
                    value=normalized,
                    supported_values=tuple(sorted(spec.supported_values, key=repr)),
                    implemented=spec.implemented,
                    action=spec.action,
                    domain_index=idx + 1 if len(values) > 1 else None,
                )
            )
    if failures:
        raise UnsupportedNamelistOption(failures)


def _coerce_config(config: Any) -> Any:
    if isinstance(config, Path):
        return _parse_wrf_namelist(config.read_text())
    if isinstance(config, str):
        if "\n" in config or config.lstrip().startswith("&"):
            return _parse_wrf_namelist(config)
        path = Path(config)
        if path.exists():
            return _parse_wrf_namelist(path.read_text())
    return config


def _lookup(config: Any, key: str) -> tuple[str, Any] | None:
    if isinstance(config, Mapping):
        if key in config:
            return key, config[key]
        for section, values in config.items():
            if isinstance(values, Mapping) and key in values:
                return f"{section}.{key}", values[key]
        return None
    if hasattr(config, key):
        return key, getattr(config, key)
    fields = getattr(config, "__dataclass_fields__", {})
    if key in fields:
        return key, getattr(config, key)
    return None


def _domain_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        arr = np.asarray(value)
        return arr.reshape(-1).tolist()
    return [value]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip().strip("'\"")
        lower = text.lower()
        if lower in {".true.", "true", "t"}:
            return True
        if lower in {".false.", "false", "f"}:
            return False
        try:
            number = float(text.replace("d", "e").replace("D", "e"))
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _parse_wrf_namelist(text: str) -> dict[str, dict[str, list[Any]]]:
    sections: dict[str, dict[str, list[Any]]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            current = line[1:].strip().lower()
            sections.setdefault(current, {})
            continue
        if line == "/":
            current = None
            continue
        if current is None or "=" not in line:
            continue
        key, rhs = line.split("=", 1)
        sections[current][key.strip().lower()] = _split_values(rhs)
    return sections


def _split_values(rhs: str) -> list[Any]:
    values: list[Any] = []
    for token in rhs.rstrip(",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(_normalize_value(token))
    return values


def _format_error(selections: list[UnsupportedSelection]) -> str:
    lines = ["Unsupported namelist/config option(s) for the GPU-WRF faithful path:"]
    for item in selections:
        domain = f" domain {item.domain_index}" if item.domain_index is not None else ""
        supported = ", ".join(repr(v) for v in item.supported_values)
        lines.append(
            f"- {item.location}{domain} selected {item.value!r}; supported values: "
            f"{supported}. Implemented: {item.implemented}. Action: {item.action}"
        )
    return "\n".join(lines)


__all__ = [
    "SUPPORTED_OPTIONS",
    "SupportedOption",
    "UnsupportedNamelistOption",
    "UnsupportedSelection",
    "validate_supported_namelist",
]
