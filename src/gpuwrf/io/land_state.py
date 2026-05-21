"""Read-only Gen2 prescribed land-state access for M6-S3."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.validation import load_gen2_var
from gpuwrf.physics.noah_mp import PrescribedNoahMPState, prescribe_noah_mp_state


LAND_STATE_VARIABLES = (
    "XLAND",
    "LANDMASK",
    "LAKEMASK",
    "IVGTYP",
    "ISLTYP",
    "LU_INDEX",
    "SST",
    "TSK",
    "SMOIS",
    "SH2O",
    "TSLB",
    "VEGFRA",
    "CM",
    "CH",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(field: Any) -> dict[str, Any]:
    data = np.asarray(field)
    finite = np.isfinite(data)
    if not finite.any():
        return {"shape": list(data.shape), "dtype": str(data.dtype), "finite": False}
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "finite": bool(finite.all()),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
    }


def load_prescribed_land_state(run: Gen2Run, domain: str = "d02", time: int = 0) -> PrescribedNoahMPState:
    """Load the Option-A Noah-MP prescribed lower boundary via validation I/O."""

    loaded = {name: load_gen2_var(run, domain, name, time=time) for name in LAND_STATE_VARIABLES if name in run.variables(domain)}
    missing = sorted(set(LAND_STATE_VARIABLES) - set(loaded))
    required = {"XLAND", "LANDMASK", "IVGTYP", "ISLTYP", "LU_INDEX", "SST", "TSK", "SMOIS", "SH2O", "TSLB"}
    absent_required = sorted(required - set(loaded))
    if absent_required:
        raise KeyError(f"required Gen2 land-state variables missing for {domain}: {absent_required}")
    source = {
        "run_id": run.run_id,
        "domain": domain,
        "time_index": int(time),
        "source_file": str(run._file_for_time(domain, time)),  # metadata only; value reads above use validation I/O.
        "missing_optional_variables": missing,
        "roughness_note": "ZNT absent; roughness_m derived from CM when usable, otherwise VEGFRA/land-water surrogate.",
    }
    return prescribe_noah_mp_state(
        t_skin=loaded["TSK"],
        smois=loaded["SMOIS"],
        sh2o=loaded["SH2O"],
        tslb=loaded["TSLB"],
        xland=loaded["XLAND"],
        landmask=loaded["LANDMASK"],
        lakemask=loaded.get("LAKEMASK", np.zeros_like(np.asarray(loaded["XLAND"]))),
        ivgtyp=loaded["IVGTYP"],
        isltyp=loaded["ISLTYP"],
        lu_index=loaded["LU_INDEX"],
        sst=loaded["SST"],
        vegfra=loaded.get("VEGFRA"),
        cm=loaded.get("CM"),
        source=source,
    )


def build_land_state_manifest(
    run: Gen2Run,
    domain: str = "d02",
    time: int = 0,
    state: PrescribedNoahMPState | None = None,
) -> dict[str, Any]:
    """Return a compact provenance manifest for the prescribed land state."""

    land = load_prescribed_land_state(run, domain, time) if state is None else state
    source_file = Path(land.source["source_file"])
    variables = {name: {"available": name in run.variables(domain)} for name in LAND_STATE_VARIABLES}
    summaries = {
        "t_skin": _summary(land.t_skin),
        "soil_moisture": _summary(land.soil_moisture),
        "soil_liquid": _summary(land.soil_liquid),
        "soil_temperature": _summary(land.soil_temperature),
        "xland": _summary(land.xland),
        "landmask": _summary(land.landmask),
        "roughness_m": _summary(land.roughness_m),
        "mavail": _summary(land.mavail),
    }
    return {
        "artifact_type": "land_state_manifest",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "domain": domain,
        "time_index": int(time),
        "source_file": str(source_file),
        "source_sha256": _sha256(source_file),
        "variables": variables,
        "summaries": summaries,
        "roughness_derivation": land.source.get("roughness_note"),
        "read_only_source_root": "/mnt/data/canairy_meteo",
        "artifact_paths": [],
    }


__all__ = ["LAND_STATE_VARIABLES", "build_land_state_manifest", "load_prescribed_land_state"]
