"""WRF auxinput4 lower boundary, loaded once and held between read alarms.

Specification: Registry.EM SST and Registry.EM_COMMON XICE/VEGFRA/ALBBCK;
share/mediation_integrate.F med_before_solve_io; phys/module_surface_driver.F
SST_UPDATE (water TSK and top TSLB, only for 250 < SST < 350 K).
No interpolation: WRF reads the next record before the first solve at/after
the auxiliary alarm. An output written at the alarm time still has the fields
used by the preceding step. All operands remain on device during stepping.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from netCDF4 import chartostring


class LowerBoundary(NamedTuple):
    activation_steps: jax.Array  # zero-based step-entry indices
    sst: jax.Array              # (record, ny, nx), K, WRF REAL
    seaice: jax.Array           # fraction (these operational cases are ice-free)
    vegfra: jax.Array           # percent, not [0,1]
    albbck: jax.Array           # background albedo


def _domain_value(values, name, index, default=None):
    value = values.get(name, default)
    if isinstance(value, (tuple, list)):
        return value[min(index, len(value) - 1)] if value else default
    return value


def load_lower_boundary(run_dir, namelist, domain, *, run_start, dt_s, shape):
    """Read aux4 outside jit; return None without file access for sst_update=0.

    Sea-ice creation/removal needs a prognostic ice land tile, which the retained
    Noah-MP hook does not implement. Reject it before a forecast rather than
    silently using water physics over ice. This path wires the ice-free WN3
    cases; it does not claim coupled-ocean or sea-ice support.
    """
    index = int(domain[1:]) - 1
    physics = namelist.get("physics", {})
    enabled = int(_domain_value(physics, "sst_update", index, 0))
    if enabled == 0:
        return None
    if enabled != 1:
        raise ValueError(f"{domain}: sst_update must be 0 or 1")
    tc = namelist.get("time_control", {})
    if int(_domain_value(tc, "io_form_auxinput4", index, 0)) != 2:
        raise ValueError(f"{domain}: sst_update=1 requires io_form_auxinput4=2 (NetCDF)")
    if int(_domain_value(physics, "sst_skin", index, 0)) or int(_domain_value(physics, "sf_lake_physics", index, 0)):
        raise ValueError(f"{domain}: aux4 SST with sst_skin/lake physics is not wired")
    seconds = sum(float(_domain_value(tc, f"auxinput4_interval_{key}", index, 0)) * factor
                  for key, factor in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)))
    if seconds == 0:
        seconds = float(_domain_value(tc, "auxinput4_interval", index, 0)) * 60
    if seconds <= 0:
        raise ValueError(f"{domain}: sst_update=1 requires a positive auxinput4 interval")
    template = str(tc.get("auxinput4_inname", "wrflowinp_d<domain>"))
    filename = template.replace("<domain>", f"{index + 1:02d}")
    if "<" in filename:
        raise ValueError(f"{domain}: unresolved auxinput4 filename {filename!r}")
    path = Path(run_dir) / filename
    with Dataset(path, "r") as ds:
        times = [datetime.strptime(str(t), "%Y-%m-%d_%H:%M:%S") for t in chartostring(ds["Times"][:])]
        start = run_start.replace(tzinfo=None)
        offsets = np.array([(t - start).total_seconds() for t in times])
        if len(offsets) < 1 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError(f"{path}: records must start at run start and increase strictly")
        if len(offsets) > 1 and not np.all(offsets == np.arange(len(offsets)) * seconds):
            raise ValueError(f"{path}: record times must match auxinput4 interval {seconds}s")
        fields = []
        for key in ("SST", "SEAICE", "VEGFRA", "ALBBCK"):
            raw = np.ma.asarray(ds[key][:])
            if np.ma.getmaskarray(raw).any():
                raise ValueError(f"{path}: {key} contains missing data")
            a = np.asarray(raw, dtype=np.float32)
            if a.shape != (len(offsets), *shape) or not np.isfinite(a).all():
                raise ValueError(f"{path}: {key} must be finite (record,{shape}), got {a.shape}")
            if key == "SEAICE" and np.any(a != 0):
                raise ValueError(f"{path}: nonzero SEAICE needs the unimplemented prognostic ice tile")
            fields.append(jnp.asarray(a))
    steps = np.ceil(offsets / float(dt_s)).astype(np.int32)
    return LowerBoundary(jnp.asarray(steps), *fields)


def lower_boundary_fields(boundary, entry_step):
    """Select the last record activated at this zero-based step entry."""
    index = jnp.sum(jnp.asarray(entry_step, dtype=jnp.int32) >= boundary.activation_steps) - 1
    # The initial record is active at step 0; no value clamps or repairs.
    return {key: jax.lax.dynamic_index_in_dim(value, index, keepdims=False)
            for key, value in zip(("SST", "SEAICE", "VEGFRA", "ALBBCK"), boundary[1:])}


def apply_lower_boundary(carry, boundary, step_index):
    """Apply surface_driver's prescribed ocean SST at one-based step entry.

    Return (carry, fields); the caller passes VEGFRA/100 to Noah-MP's static
    input view. ALBBCK is a held input/history field: Noah-MP computes its own
    radiative albedo, independently of that WRF background field.
    """
    if boundary is None:
        return carry, None
    fields = lower_boundary_fields(boundary, step_index - 1)
    return apply_lower_boundary_fields(carry, fields), fields


def apply_lower_boundary_fields(carry, fields):
    """Surface-driver phase: radiation has already consumed the previous TSK."""
    state = carry.state
    sst = fields["SST"].astype(state.t_skin.dtype)
    water = (state.xland > 1.5) & (sst > 250.0) & (sst < 350.0)
    tsk = jnp.where(water, sst, state.t_skin)
    state = state.replace(t_skin=tsk)
    land = carry.noahmp_land
    if land is not None:
        top = jnp.where(water, sst.astype(land.tslb.dtype), land.tslb[0])
        land = land.replace(tslb=land.tslb.at[0].set(top),
                            t_skin=jnp.where(water, sst.astype(land.t_skin.dtype), land.t_skin))
    return carry.replace(state=state, noahmp_land=land)


def lower_boundary_history(boundary, own_step):
    """Held fields used in the last completed step, for history diagnostics."""
    if boundary is None:
        return {}
    fields = lower_boundary_fields(boundary, max(int(own_step) - 1, 0))
    if int(own_step) == 0:
        # WRF writes initial history before the first aux4 read. SST/SEAICE/
        # VEGFRA record0 are unchanged wrfinput fields; ALBBCK is first replaced
        # by LANDUSE initialization, so leave the initial writer value intact.
        fields.pop("ALBBCK")
    return fields
