"""Fail-fast support check for WRF namelist/config options.

For v0.6.0, the physics values accepted here are the frozen interface matrix,
not a claim that every scheme is already wired into the operational dispatcher.
Unsupported option numbers still fail closed loudly. Per-scheme lanes must pass
their WRF savepoint parity gates before a non-Thompson suite can be used for an
integrated forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gpuwrf.contracts.physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_RA_LW_PHYSICS,
    ACCEPTED_RA_SW_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
)
from gpuwrf.io.wrf_scheme_catalog import WRF_PARAM_LABEL, wrf_scheme_name


@dataclass(frozen=True)
class SupportedOption:
    """One supported-option registry entry."""

    key: str
    supported_values: frozenset[Any]
    implemented: str
    action: str


@dataclass(frozen=True)
class UnsupportedSelection:
    """A selected namelist/config value outside the faithful registry.

    ``outcome`` records *why* the value was rejected, so the caller (and the
    formatted message) can distinguish a recognized WRF v4 scheme that is not
    yet implemented in the GPU port from a value that is not a valid WRF option:

    * ``"not_yet_implemented"`` -- a recognized WRF v4 scheme that the port does
      not yet wire (``wrf_scheme`` names it);
    * ``"invalid_wrf_option"`` -- not a recognized WRF v4 option at all;
    * ``"unsupported"`` -- generic rejection for keys without a WRF v4 catalog
      (e.g. a structural pairing constraint).
    """

    key: str
    location: str
    value: Any
    supported_values: tuple[Any, ...]
    implemented: str
    action: str
    domain_index: int | None = None
    outcome: str = "unsupported"
    wrf_scheme: str | None = None


class UnsupportedNamelistOption(ValueError):
    """Raised when a namelist/config selects an unsupported option."""

    def __init__(self, selections: list[UnsupportedSelection]) -> None:
        self.selections = tuple(selections)
        super().__init__(_format_error(selections))


SUPPORTED_OPTIONS: dict[str, SupportedOption] = {
    # Physics suite wired in runtime.operational_mode and coupling.physics_couplers.
    "mp_physics": SupportedOption(
        key="mp_physics",
        supported_values=frozenset(ACCEPTED_MP_PHYSICS),
        implemented="0=disabled/passive qv, 1=Kessler, 2=Purdue-Lin, 3=WSM3, 4=WSM5, 6=WSM6, 8=Thompson, 10=Morrison, 16=WDM6",
        action="Use one of the frozen v0.6.0 microphysics options; all other MP options remain unsupported.",
    ),
    "cu_physics": SupportedOption(
        key="cu_physics",
        supported_values=frozenset(ACCEPTED_CU_PHYSICS),
        implemented=(
            "0=disabled, 1=Kain-Fritsch, 2=Betts-Miller-Janjic (fp64 savepoint-parity), "
            "6=Tiedtke (all GPU-operational, scan-wired); "
            "3=Grell-Freitas (savepoint-parity CPU reference; GPU closure-ensemble batch TODO), "
            "16=New Tiedtke (accepted, NOT separately source-gated; fail-closed in the GPU scan) "
            "-- 3/16 are selectable for reference but fail-closed in the operational GPU scan"
        ),
        action=(
            "Use cu_physics=0/1/2/6 for the operational GPU scan; "
            "3=Grell-Freitas and 16=New Tiedtke remain fail-closed (not scan-wired)."
        ),
    ),
    "bl_pbl_physics": SupportedOption(
        key="bl_pbl_physics",
        supported_values=frozenset(ACCEPTED_BL_PBL_PHYSICS),
        implemented=(
            "0=disabled, 1=YSU, 5=MYNN, 7=ACM2, 8=BouLac (all GPU-operational, scan-wired); "
            "2=MYJ (savepoint-parity-proven CPU reference, NOT yet GPU-scan-wired -- "
            "selectable for reference but fail-closed in the operational GPU scan, GPU-batching TODO)"
        ),
        action=(
            "Use bl_pbl_physics=0/1/5/7/8 for the operational GPU scan; 2=MYJ remains "
            "CPU-reference-only and must pair with sf_sfclay_physics=2. "
            "All other PBL options remain unsupported. "
            "Pair with the matching surface layer (MYNN<->5, ACM2<->7/1, YSU<->1, MYJ<->2)."
        ),
    ),
    "sf_sfclay_physics": SupportedOption(
        key="sf_sfclay_physics",
        supported_values=frozenset(ACCEPTED_SF_SFCLAY_PHYSICS),
        implemented=(
            "0=disabled, 1=revised-MM5, 5=MYNN surface layer, 7=Pleim-Xiu surface layer "
            "(all GPU-operational, scan-wired); "
            "2=Janjic Eta (savepoint-parity-proven CPU reference, NOT yet GPU-scan-wired -- "
            "selectable for reference but fail-closed in the operational GPU scan, GPU-batching TODO)"
        ),
        action=(
            "Use sf_sfclay_physics=0/1/5/7 for the operational GPU scan; 2=Janjic Eta remains "
            "CPU-reference-only and must pair with bl_pbl_physics=2. "
            "All other sfclay options remain unsupported. "
            "Use the PBL-compatible partner (MYNN-SL 5<->MYNN PBL 5, Pleim-Xiu 7<->ACM2 7, Janjic 2<->MYJ 2)."
        ),
    ),
    "sf_surface_physics": SupportedOption(
        key="sf_surface_physics",
        supported_values=frozenset(ACCEPTED_SF_SURFACE_PHYSICS),
        implemented="0=disabled, 2=Noah classic (explicit static/land bundle), 4=Noah-MP "
        "(set use_noahmp=True) -- both GPU-operational, scan-wired",
        action="Use one of the frozen v0.6.0 land-surface options; all other land-surface options remain unsupported.",
    ),
    "ra_sw_physics": SupportedOption(
        key="ra_sw_physics",
        supported_values=frozenset(ACCEPTED_RA_SW_PHYSICS),
        implemented="0=disabled, 1=Dudhia shortwave (isolated-savepoint parity-proven + accepted, "
        "NOT operational-scan-wired), 4=RRTMG shortwave (GPU-operational; the operational radiation "
        "slot runs RRTMG)",
        action="Use ra_sw_physics=4 for the operational RRTMG SW path, or 0 when radiation is "
        "disabled. ra_sw_physics=1 (Dudhia) passes its isolated WRF-savepoint gate but is not yet "
        "selectable by the operational scan (post-0.9.0 radiation-family dispatch).",
    ),
    "ra_lw_physics": SupportedOption(
        key="ra_lw_physics",
        supported_values=frozenset(ACCEPTED_RA_LW_PHYSICS),
        implemented="0=disabled, 1=RRTM longwave (isolated-savepoint parity-proven + accepted, "
        "NOT operational-scan-wired; host-NumPy single-column kernel), 4=RRTMG longwave "
        "(GPU-operational; the operational radiation slot runs RRTMG)",
        action="Use ra_lw_physics=4 for the operational RRTMG LW path, or 0 when radiation is "
        "disabled. ra_lw_physics=1 (classic RRTM) passes its isolated WRF-savepoint gate but is not "
        "yet selectable by the operational scan (post-0.9.0 jit/vmap rewrite + radiation dispatch).",
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
        supported_values=frozenset({0, 1, 2}),
        implemented="0=off, 1=coordinate-surface (eta) horizontal diffusion, "
        "2=physical-level constant-K diffusion path",
        action="Use diff_opt=1/km_opt=4 for the real-data default 2-D Smagorinsky, "
        "diff_opt=2/km_opt=1 for the constant-K path, or 0.",
    ),
    "km_opt": SupportedOption(
        key="km_opt",
        supported_values=frozenset({0, 1, 4}),
        implemented="0=off, 1=constant-K coefficient, 4=2-D Smagorinsky horizontal "
        "eddy viscosity (vertical mixing from the PBL scheme)",
        action="Use km_opt=4 with diff_opt=1 for the real-data default 2-D "
        "Smagorinsky, km_opt=1 with diff_opt=2 for constant-K, or 0.",
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
            outcome, wrf_scheme = _classify_rejection(key, normalized)
            failures.append(
                UnsupportedSelection(
                    key=key,
                    location=location,
                    value=normalized,
                    supported_values=_sorted_supported_values(spec.supported_values),
                    implemented=spec.implemented,
                    action=spec.action,
                    domain_index=idx + 1 if len(values) > 1 else None,
                    outcome=outcome,
                    wrf_scheme=wrf_scheme,
                )
            )
    failures.extend(_myj_pairing_failures(config_obj))
    if failures:
        raise UnsupportedNamelistOption(failures)


def _classify_rejection(key: str, value: Any) -> tuple[str, str | None]:
    """Classify a rejected ``key=value`` against the full WRF v4 catalog.

    Returns ``(outcome, wrf_scheme_name)``:

    * ``("not_yet_implemented", "<scheme name>")`` -- ``value`` is a recognized
      WRF v4 option for ``key`` that the GPU port does not yet implement /
      operationally wire;
    * ``("invalid_wrf_option", None)`` -- ``value`` is not a recognized WRF v4
      option for ``key`` (and ``key`` has a WRF v4 catalog);
    * ``("unsupported", None)`` -- ``key`` has no WRF v4 catalog (no enumeration
      to check against, e.g. a structural-only control).
    """

    if not isinstance(value, int):
        # Non-integer selections (e.g. a stray string) cannot be a WRF code.
        scheme = wrf_scheme_name(key, value) if isinstance(value, (int, float)) else None
        if scheme is not None:
            return "not_yet_implemented", scheme.name
        return "invalid_wrf_option", None

    scheme = wrf_scheme_name(key, value)
    if scheme is not None:
        return "not_yet_implemented", scheme.name
    if key in WRF_PARAM_LABEL:
        return "invalid_wrf_option", None
    return "unsupported", None


def _myj_pairing_failures(config: Any) -> list[UnsupportedSelection]:
    bl_found = _lookup(config, "bl_pbl_physics")
    sf_found = _lookup(config, "sf_sfclay_physics")
    if bl_found is None and sf_found is None:
        return []

    bl_location = bl_found[0] if bl_found is not None else "bl_pbl_physics"
    sf_location = sf_found[0] if sf_found is not None else "sf_sfclay_physics"
    bl_values = [_normalize_value(v) for v in _domain_values(bl_found[1])] if bl_found is not None else [None]
    sf_values = [_normalize_value(v) for v in _domain_values(sf_found[1])] if sf_found is not None else [None]
    ndom = max(len(bl_values), len(sf_values))

    failures: list[UnsupportedSelection] = []
    for idx in range(ndom):
        bl = bl_values[idx] if idx < len(bl_values) else bl_values[-1]
        sf = sf_values[idx] if idx < len(sf_values) else sf_values[-1]
        if (bl == 2) == (sf == 2):
            continue
        if bl != 2 and sf != 2:
            continue
        failures.append(
            UnsupportedSelection(
                key="myj_pairing",
                location=f"{bl_location}/{sf_location}",
                value={"bl_pbl_physics": bl, "sf_sfclay_physics": sf},
                supported_values=("bl_pbl_physics=2 with sf_sfclay_physics=2",),
                implemented="MYJ PBL and Janjic Eta surface layer are a mandatory WRF pair",
                action="Select both option values as 2, or select neither as 2.",
                domain_index=idx + 1 if ndom > 1 else None,
            )
        )
    return failures


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


def _sorted_supported_values(values: frozenset[Any]) -> tuple[Any, ...]:
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=repr))


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
        values.extend(_expand_repeat(token))
    return values


def _expand_repeat(token: str) -> list[Any]:
    """Expand a Fortran namelist repeat-count token ``N*value`` -> N copies.

    WRF namelists very commonly use ``3*8`` to mean ``8, 8, 8`` across max_dom
    domains. The ``*`` is the Fortran repeat operator, not multiplication.
    Tokens without a valid ``N*`` prefix are returned as a single normalized
    value unchanged (a bare ``*`` Fortran "no value" marker is dropped).
    """

    if "*" not in token:
        return [_normalize_value(token)]
    count_str, _, value_str = token.partition("*")
    count_str = count_str.strip()
    value_str = value_str.strip()
    if not count_str.isdigit():
        # Not a repeat count (e.g. an unexpected expression); keep verbatim.
        return [_normalize_value(token)]
    count = int(count_str)
    if not value_str:
        # Fortran ``N*`` with no value = "keep N defaults": nothing to validate.
        return []
    return [_normalize_value(value_str)] * count


def _format_error(selections: list[UnsupportedSelection]) -> str:
    lines = ["Unsupported namelist/config option(s) for the GPU-WRF faithful path:"]
    for item in selections:
        lines.append(_format_selection(item))
    return "\n".join(lines)


def _format_selection(item: UnsupportedSelection) -> str:
    domain = f" domain {item.domain_index}" if item.domain_index is not None else ""
    supported = ", ".join(repr(v) for v in item.supported_values)

    if item.outcome == "not_yet_implemented":
        label = WRF_PARAM_LABEL.get(item.key, item.key)
        return (
            f"- {item.location}{domain}={item.value} ({item.wrf_scheme}): recognized WRF v4 "
            f"{label} scheme, NOT YET IMPLEMENTED in the GPU port. "
            f"Supported {item.key} values: {supported}. "
            f"Implemented: {item.implemented}. Action: {item.action}"
        )
    if item.outcome == "invalid_wrf_option":
        label = WRF_PARAM_LABEL.get(item.key, item.key)
        return (
            f"- {item.location}{domain}={item.value} is not a recognized WRF v4 {label} option. "
            f"Supported {item.key} values: {supported}. "
            f"Implemented: {item.implemented}. Action: {item.action}"
        )
    # Generic / structural rejection (no WRF v4 catalog for this key).
    return (
        f"- {item.location}{domain} selected {item.value!r}; supported values: "
        f"{supported}. Implemented: {item.implemented}. Action: {item.action}"
    )


__all__ = [
    "SUPPORTED_OPTIONS",
    "SupportedOption",
    "UnsupportedNamelistOption",
    "UnsupportedSelection",
    "validate_supported_namelist",
]
