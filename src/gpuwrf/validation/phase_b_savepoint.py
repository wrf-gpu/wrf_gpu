"""Phase-B savepoint loader / validator (Gate-1 frozen interface).

This is the thin, frozen surface the physics lanes (B1 Thompson, B2 surface +
MYNN, B3 RRTMG) use to READ WRF-oracle savepoints, and that the WRF-oracle
factory uses to know the accepted operator/boundary names and tolerance ladder.

It deliberately *reuses* the committed dycore savepoint machinery
(:mod:`gpuwrf.validation.savepoint_schema` + :mod:`gpuwrf.validation.savepoint_io`)
rather than inventing a parallel format.  Phase B only adds:

* the physics operator / boundary names the dycore schema does not enumerate,
* per-field tolerance bands (tight transcription vs operational RMSE),
* the per-scheme physical-activation floors so a validator applies the same
  "physically-inactive != missing operator" rule as the diagnostic harness,
* a ``source_run_id`` accessor and a checksum-verifying loader.

No model code lives here — it is contract/validation glue only.  Importing this
module must NOT pull in JAX or touch the GPU (NumPy + h5py only), so lanes can use
it in pure-CPU CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.validation.savepoint_schema import (
    Savepoint,
    SavepointMetadata,
    load_tolerance_ladder,
)


# ---------------------------------------------------------------------------
# Phase-B operator / boundary names (union with the frozen dycore sets)
# ---------------------------------------------------------------------------

# WRF physics operator names the oracle factory may emit.  These EXTEND, not
# replace, ``savepoint_schema.VALID_OPERATORS`` (which is frozen for the dycore).
PHASE_B_OPERATORS: frozenset[str] = frozenset(
    {
        # B1 Thompson microphysics
        "mp_gt_driver",
        "thompson_microphysics",
        # B2 surface layer + MYNN PBL
        "sfclay",
        "surface_layer",
        "mynn_surface",
        "mynn_bl_driver",
        "mynn_pbl",
        # B3 RRTMG radiation + land/diurnal driver
        "radiation_driver",
        "rrtmg_sw",
        "rrtmg_lw",
        # B4 lateral boundary / static (boundary application is dycore-adjacent)
        "lateral_boundary",
    }
)

# WRF physics savepoint boundary (operator-internal cut) names.
PHASE_B_BOUNDARIES: frozenset[str] = frozenset(
    {
        # B1
        "mp_gt_driver_pre",
        "mp_gt_driver_post",
        "mp_thompson_process_boundary",
        # B2 surface
        "sfclay_pre",
        "sfclay_post",
        # B2 MYNN internal process boundaries (mym_* + tendencies)
        "mym_level2",
        "mym_length",
        "mym_turbulence",
        "mym_predict",
        "mynn_tendencies",
        "mynn_bl_driver_pre",
        "mynn_bl_driver_post",
        # B3
        "radiation_driver_pre",
        "radiation_driver_post",
        "rrtmg_sw_pre",
        "rrtmg_sw_post",
        "rrtmg_lw_pre",
        "rrtmg_lw_post",
        # B4
        "lateral_boundary_pre",
        "lateral_boundary_post",
    }
)


# ---------------------------------------------------------------------------
# Frozen Phase-B tolerance bands (see contracts/phase_b/savepoint_schema.md §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToleranceBand:
    """One field's two-regime tolerance: tight transcription + operational RMSE."""

    units: str
    transcription_abs: float
    transcription_rel: float
    operational_rmse: float
    rmse_units: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "transcription_abs": self.transcription_abs,
            "transcription_rel": self.transcription_rel,
            "operational_rmse": self.operational_rmse,
            "rmse_units": self.rmse_units or self.units,
        }


PHASE_B_TOLERANCES: dict[str, ToleranceBand] = {
    # B1 microphysics mass species (kg/kg)
    "qv": ToleranceBand("kg kg^-1", 1e-9, 1e-6, 1e-4),
    "qc": ToleranceBand("kg kg^-1", 1e-9, 1e-6, 1e-4),
    "qr": ToleranceBand("kg kg^-1", 1e-9, 1e-6, 1e-4),
    "qi": ToleranceBand("kg kg^-1", 1e-9, 1e-6, 1e-4),
    "qs": ToleranceBand("kg kg^-1", 1e-9, 1e-6, 1e-4),
    "qg": ToleranceBand("kg kg^-1", 1e-9, 1e-6, 1e-4),
    # B1 number concentrations (m^-3) — operational band is relative (10%)
    "Ni": ToleranceBand("m^-3", 1e-3, 1e-4, 0.10),
    "Nr": ToleranceBand("m^-3", 1e-3, 1e-4, 0.10),
    "Ns": ToleranceBand("m^-3", 1e-3, 1e-4, 0.10),
    "Ng": ToleranceBand("m^-3", 1e-3, 1e-4, 0.10),
    # B2 surface stability / flux handles
    "ustar": ToleranceBand("m s^-1", 1e-7, 1e-7, 0.05),
    "theta_flux": ToleranceBand("K m s^-1", 1e-7, 1e-6, 30.0, "W m^-2 (via HFX)"),
    "qv_flux": ToleranceBand("kg kg^-1 m s^-1", 1e-7, 1e-6, 30.0, "W m^-2 (via LH)"),
    "tau_u": ToleranceBand("m2 s^-2", 1e-7, 1e-6, 0.05),
    "tau_v": ToleranceBand("m2 s^-2", 1e-7, 1e-6, 0.05),
    "rhosfc": ToleranceBand("kg m^-3", 1e-7, 1e-7, 0.02),
    # B3 radiation surface diagnostics
    "SWDOWN": ToleranceBand("W m^-2", 1e-4, 1e-5, 20.0),
    "GLW": ToleranceBand("W m^-2", 1e-4, 1e-5, 20.0),
    # operational diagnostic set
    "HFX": ToleranceBand("W m^-2", 1e-4, 1e-5, 30.0),
    "LH": ToleranceBand("W m^-2", 1e-4, 1e-5, 30.0),
    "PBLH": ToleranceBand("m", 1e-2, 1e-4, 150.0),
    "TSK": ToleranceBand("K", 1e-5, 1e-6, 0.0),  # data-replayed -> bitwise
    "T2": ToleranceBand("K", 1e-5, 1e-6, 1.5),
    "U10": ToleranceBand("m s^-1", 1e-5, 1e-6, 1.5),
    "V10": ToleranceBand("m s^-1", 1e-5, 1e-6, 1.5),
    "PSFC": ToleranceBand("Pa", 1e-3, 1e-7, 50.0),
    "theta": ToleranceBand("K", 1e-6, 1e-7, 1.5),
}


# ---------------------------------------------------------------------------
# Physical-activation floors (mirror diagnostics.comprehensive_harness)
# ---------------------------------------------------------------------------
#
# A WRF oracle column may legitimately show zero scheme output where the scheme
# is physically inactive.  A lane validator uses these to apply the
# "physically-inactive != missing" rule: a zero scheme output is only a FAILURE
# if the driving input cleared the floor.
PHASE_B_ACTIVATION_FLOORS: dict[str, float] = {
    "microphysics_condensate_kg_kg": 1.0e-8,
    "microphysics_vapour_kg_kg": 1.0e-6,
    "surface_wind_m_s": 1.0e-3,
    "pbl_tke_m2_s2": 1.0e-6,
    "radiation_coszen": 1.0e-3,
    "dycore_perturbation": 1.0e-6,
}


def activation_floor_for(key: str) -> float:
    """Return the physical-activation floor for one scheme-input key."""

    try:
        return PHASE_B_ACTIVATION_FLOORS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no activation floor for {key!r}") from exc


def phase_b_tolerance(field_name: str) -> ToleranceBand:
    """Return the frozen Phase-B tolerance band for a field, by name."""

    try:
        return PHASE_B_TOLERANCES[field_name]
    except KeyError as exc:
        raise KeyError(f"no Phase-B tolerance band for field {field_name!r}") from exc


def is_accepted_operator(operator: str) -> bool:
    """True if ``operator`` is a frozen dycore OR Phase-B physics operator."""

    from gpuwrf.validation.savepoint_schema import VALID_OPERATORS

    return operator in VALID_OPERATORS or operator in PHASE_B_OPERATORS


def is_accepted_boundary(boundary: str) -> bool:
    """True if ``boundary`` is a frozen dycore OR Phase-B physics boundary."""

    from gpuwrf.validation.savepoint_schema import VALID_BOUNDARIES

    return boundary in VALID_BOUNDARIES or boundary in PHASE_B_BOUNDARIES


# ---------------------------------------------------------------------------
# Loader / validator
# ---------------------------------------------------------------------------


def source_run_id(metadata: SavepointMetadata) -> str:
    """Stable source-run identifier for a savepoint's originating WRF run."""

    return "|".join(
        str(part)
        for part in (
            metadata.run_id,
            metadata.wrf_version,
            metadata.wrf_commit,
            metadata.namelist_hash,
            metadata.source_path,
            metadata.domain_index,
        )
    )


def load_phase_b_savepoint(path: str | Path, *, verify_checksum: bool = True) -> Savepoint:
    """Load + validate a WRF-oracle savepoint for a physics lane.

    Verifies the payload checksum (unless explicitly disabled), the array
    shape/dtype contract (:meth:`Savepoint.validate`), and that the operator /
    boundary are in the accepted (dycore ∪ Phase-B) sets.  Raises ``ValueError``
    on any mismatch so a lane never silently validates against a corrupt or
    misclassified oracle file.

    h5py is imported lazily so importing this module stays dependency-light.
    """

    import h5py  # local import: keep module import lightweight
    import json
    import hashlib

    from gpuwrf.validation.savepoint_io import (
        METADATA_ATTR,
        PAYLOAD_SHA256_ATTR,
        FIELDS_GROUP,
        _payload_digest,
    )

    target = Path(path)
    with h5py.File(target, "r") as handle:
        metadata = SavepointMetadata.from_json(json.loads(handle.attrs[METADATA_ATTR]))
        arrays = {name: np.asarray(handle[FIELDS_GROUP][name]) for name in handle[FIELDS_GROUP]}
        stored_digest = str(handle.attrs.get(PAYLOAD_SHA256_ATTR, ""))

    savepoint = Savepoint(metadata=metadata, arrays=arrays)
    savepoint.validate()

    if not is_accepted_operator(metadata.operator):
        raise ValueError(f"savepoint operator {metadata.operator!r} not in accepted Phase-B/dycore set")
    if not is_accepted_boundary(metadata.boundary):
        raise ValueError(f"savepoint boundary {metadata.boundary!r} not in accepted Phase-B/dycore set")

    if verify_checksum and stored_digest:
        recomputed = _payload_digest(metadata, arrays)
        if recomputed != stored_digest:
            raise ValueError(
                f"savepoint checksum mismatch for {target}: "
                f"stored={stored_digest[:16]}… recomputed={recomputed[:16]}…"
            )
    return savepoint


def phase_b_tolerance_ladder() -> dict[str, Any]:
    """Return the committed dycore ladder merged with the Phase-B physics bands.

    The dycore ladder is loaded/validated from ``tolerance_ladder.json``; the
    Phase-B physics + operational-diagnostic bands are merged in under a
    ``phase_b_fields`` key so the dycore entries stay untouched.
    """

    ladder = load_tolerance_ladder()
    ladder = dict(ladder)
    ladder["phase_b_fields"] = {name: band.as_json() for name, band in PHASE_B_TOLERANCES.items()}
    ladder["phase_b_activation_floors"] = dict(PHASE_B_ACTIVATION_FLOORS)
    return ladder


__all__ = [
    "PHASE_B_OPERATORS",
    "PHASE_B_BOUNDARIES",
    "PHASE_B_TOLERANCES",
    "PHASE_B_ACTIVATION_FLOORS",
    "ToleranceBand",
    "activation_floor_for",
    "phase_b_tolerance",
    "is_accepted_operator",
    "is_accepted_boundary",
    "source_run_id",
    "load_phase_b_savepoint",
    "phase_b_tolerance_ladder",
]
